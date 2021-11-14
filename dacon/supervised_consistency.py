import os

import numpy as np
import torch
import torch.nn as nn
from apex import amp
from tensorboardX import SummaryWriter
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup

from dacon.dataset import DaconDataset
from dacon.model import MultiTaskNet
from snippext.dataset import SnippextDataset, get_tokenizer
from snippext.train_util import *


def train(model, train_set, optimizer, scheduler=None, batch_size=32, fp16=False):
    """Perfrom one epoch of the training process.

    Args:
        model (MultiTaskNet): the current model state
        train_set (SnippextDataset): the training dataset
        optimizer: the optimizer for training (e.g., Adam)
        batch_size (int, optional): the batch size
        fp16 (boolean): whether to use fp16

    Returns:
        None
    """
    iterator = data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=DaconDataset.pad,
    )

    tagging_criterion = nn.CrossEntropyLoss(ignore_index=0)
    classifier_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    model.train()
    for i, batch in enumerate(iterator):
        # for monitoring
        orig_batch, aug_batch = batch
        (
            orig_words,
            x,
            orig_is_heads,
            tags,
            orig_mask,
            y,
            orig_seqlens,
            orig_taskname,
        ) = orig_batch

        taskname = orig_taskname[0]
        _y = y

        if "tagging" in taskname:
            criterion = tagging_criterion
        elif "sts-b" in taskname:
            criterion = regression_criterion
        else:
            criterion = classifier_criterion

        # forward
        optimizer.zero_grad()

        orig_logits, orig_y, _ = model(x, y, task=taskname)
        orig_logits = (
            orig_logits.view(-1)
            if "sts-b" in taskname
            else orig_logits.view(-1, orig_logits.shape[-1])
        )
        orig_y = orig_y.view(-1)
        orig_ce_loss = criterion(orig_logits, orig_y)

        loss = orig_ce_loss

        for aug_sample in aug_batch:
            (
                aug_words,
                aug_x,
                aug_is_heads,
                tags,
                aug_mask,
                y,
                aug_seqlens,
                aug_taskname,
            ) = aug_sample
            aug_logits, aug_y, _ = model(aug_x, y, task=taskname)
            aug_logits = (
                aug_logits.view(-1)
                if "sts-b" in taskname
                else aug_logits.view(-1, aug_logits.shape[-1])
            )
            aug_y = aug_y.view(-1)
            aug_ce_loss = criterion(aug_logits, aug_y)
            loss += aug_ce_loss

        # back propagation
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i == 0:
            print("=====sanity check======")
            print("-----original data-----")
            print("orig_words: ", orig_words[0])
            print("x:", x.cpu().numpy()[0][: orig_seqlens[0]])
            print(
                "orig_tokens: ",
                get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[
                    : orig_seqlens[0]
                ],
            )
            print("orig_is_heads: ", orig_is_heads[0])
            y_sample = _y.cpu().numpy()[0]
            if np.isscalar(y_sample):
                print("y :", y_sample)
            else:
                print("y: ", y_sample[: aug_seqlens[0]])
            print("tags: ", tags[0])
            print("orig_mask: ", orig_mask[0])
            print("orig_seqlen: ", orig_seqlens[0])
            print("task_name: ", taskname)
            print("-----augmented data----")
            print("aug_words: ", aug_words[0])
            print("aug_x: ", x.cpu().numpy()[0][: aug_seqlens[0]])
            print(
                "aug_tokens: ",
                get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[
                    : aug_seqlens[0]
                ],
            )
            print("aug_is_heads: ", aug_is_heads[0])
            y_sample = _y.cpu().numpy()[0]
            if np.isscalar(y_sample):
                print("y: ", y_sample)
            else:
                print("y: ", y_sample[: aug_seqlens[0]])
            print("tags: ", tags[0])
            print("aug_mask: ", aug_mask[0])
            print("aug_seqlen: ", aug_seqlens[0])
            print("task_name: ", taskname)
            print("----aug_distribution---")
            print("aug_distribution: ", model.aug_distribution)
            print("=======================")

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, task: {taskname}, loss: {loss.item()}")
            del loss


def initialize_and_train(
    task_config,
    train_raw_set,
    valid_set,
    test_set,
    train_dataset_class,
    vocab,
    hp,
    run_tag,
):
    """The train process.

    Args:
        task_config (dictionary): the configuration of the task
        train_raw_set (str): path to the training set
        valid_set (DittoDataset or TextCLSDataset): the validation set
        test_set (DittoDataset or TextCLSDataset): the test set
        train_dataset_class (DaconDataset or DaconTextCLSDataset): the dataset class for training data
        vocab: the vocab for the model
        hp (Namespace): the parsed hyper-parameters
        run_tag (string): the tag of the run (for logging purpose)

    Returns:
        None
    """
    # initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskNet([task_config], device, lm=hp.lm, bert_path=hp.bert_path)

    # create DaconDataset or DaconTextCLSDataset for train data
    train_set = train_dataset_class(
        source=train_raw_set,
        vocab=vocab,
        taskname=task_config["name"],
        lm=hp.lm,
        max_len=hp.max_len,
        size=hp.size,
    )

    # create iterators for validation and test data
    valid_iter = data.DataLoader(
        dataset=valid_set,
        batch_size=hp.batch_size * 4,
        shuffle=False,
        num_workers=0,
        collate_fn=SnippextDataset.pad,
    )

    test_iter = data.DataLoader(
        dataset=test_set,
        batch_size=hp.batch_size * 4,
        shuffle=False,
        num_workers=0,
        collate_fn=SnippextDataset.pad,
    )

    if device == "cpu":
        optimizer = AdamW(model.parameters(), lr=hp.lr)
    else:
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        if hp.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # learning rate scheduler
    num_steps = (len(train_set) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 10, num_training_steps=num_steps
    )

    # create logging directory
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    epoch = 1
    while epoch <= hp.n_epochs:
        train(
            model,
            train_set,
            optimizer,
            scheduler=scheduler,
            batch_size=hp.batch_size,
            fp16=hp.fp16,
        )

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(
            epoch,
            model,
            task_config["name"],
            valid_iter,
            valid_set,
            test_iter,
            test_set,
            writer,
            run_tag,
        )

        # if dev_f1 > 1e-6:
        epoch += 1
        if hp.save_model:
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                torch.save(model.state_dict(), run_tag + "_dev.pt")
            if test_f1 > best_test_f1:
                best_test_f1 = dev_f1
                torch.save(model.state_dict(), run_tag + "_test.pt")

    writer.close()
