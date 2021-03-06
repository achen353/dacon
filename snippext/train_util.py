import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from transformers.data import glue_compute_metrics, glue_processors


def eval_classifier(model, iterator, threshold=None, get_threshold=False):
    """Evaluate a classification model state on a dev/test set.

    Args:
        model (MultiTaskNet): the model state
        iterator (DataLoader): a batch iterator of the dev/test set
        threshold (float, optional): the cut-off threshold for binary cls
        get_threshold (boolean, optional): return the selected threshold if True

    Returns:
        float: Precision (or accuracy if more than 2 classes)
        float: Recall (or accuracy if more than 2 classes)
        float: F1 (or macro F1 if more than 2 classes)
        float: The Loss
        float: The cut-off threshold
    """
    model.eval()

    Y = []
    Y_hat = []
    Y_prob = []
    loss_list = []
    total_size = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            _, x, _, _, _, y, _, taskname = batch
            taskname = taskname[0]
            logits, y1, y_hat = model(x, y, task=taskname)
            logits = logits.view(-1, logits.shape[-1])
            y1 = y1.view(-1)
            if "sts-b" in taskname.lower():
                loss = nn.MSELoss()(logits, y1)
            else:
                loss = nn.CrossEntropyLoss()(logits, y1)

            loss_list.append(loss.item() * y.shape[0])
            total_size += y.shape[0]

            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Y_prob.extend(logits.softmax(dim=-1).max(dim=-1)[0].cpu().numpy().tolist())

    loss = sum(loss_list) / total_size

    print("=============%s==================" % taskname)

    # for glue
    if taskname in glue_processors:
        Y_hat = np.array(Y_hat).squeeze()
        Y = np.array(Y)
        result = glue_compute_metrics(taskname, Y_hat, Y)
        result["loss"] = loss
        print(result)
        return result
    elif taskname[:5] == "glue_":
        task = taskname.split("_")[1].lower()
        Y_hat = np.array(Y_hat).squeeze()
        Y = np.array(Y)
        result = glue_compute_metrics(task, Y_hat, Y)
        result["loss"] = loss
        print(result)
        return result
    else:
        num_classes = len(set(Y))
        # Binary classification
        if num_classes <= 2:
            accuracy = metrics.accuracy_score(Y, Y_hat)
            precision = metrics.precision_score(Y, Y_hat)
            recall = metrics.recall_score(Y, Y_hat)
            f1 = metrics.f1_score(Y, Y_hat)
            if any(
                [prefix in taskname for prefix in ["cleaning_", "em_"]]
            ):  # handle imbalance:
                max_f1 = f1
                if threshold is None:
                    for th in np.arange(0.0, 1.0, 0.005):
                        Y_hat = [y if p > th else 0 for (y, p) in zip(Y_hat, Y_prob)]
                        f1 = metrics.f1_score(Y, Y_hat)
                        if f1 > max_f1:
                            max_f1 = f1
                            accuracy = metrics.accuracy_score(Y, Y_hat)
                            precision = metrics.precision_score(Y, Y_hat)
                            recall = metrics.recall_score(Y, Y_hat)
                            threshold = th
                    f1 = max_f1
                else:
                    Y_hat = [y if p > threshold else 0 for (y, p) in zip(Y_hat, Y_prob)]
                    accuracy = metrics.accuracy_score(Y, Y_hat)
                    precision = metrics.precision_score(Y, Y_hat)
                    recall = metrics.recall_score(Y, Y_hat)
                    f1 = metrics.f1_score(Y, Y_hat)

            print("accuracy=%.3f" % accuracy)
            print("precision=%.3f" % precision)
            print("recall=%.3f" % recall)
            print("f1=%.3f" % f1)
            print("======================================")
            if get_threshold:
                return accuracy, precision, recall, f1, loss, threshold
            else:
                return accuracy, precision, recall, f1, loss
        else:
            accuracy = metrics.accuracy_score(Y, Y_hat)
            f1 = metrics.f1_score(Y, Y_hat, average="macro")
            precision = recall = accuracy  # We might just not return anything
            print("accuracy=%.3f" % accuracy)
            print("macro_f1=%.3f" % f1)
            print("======================================")
            return accuracy, f1, loss


def eval_on_task(
    epoch,
    model,
    task,
    valid_iter,
    valid_dataset,
    test_iter,
    test_dataset,
    writer,
    run_tag,
):
    """Run the eval function on the dev/test datasets and log the results.

    Args:
        epoch (int): the epoch number of the training process
        model (MultiTaskNet): the model state
        task (str): the task name to be evaluated
        valid_iter (DataLoader): the dev set iterator
        valid_dataset (Dataset): the dev dataset
        test_iter (DataLoader): the test set iterator
        test_dataset (Datset): the test dataset
        writer (SummaryWriter): the logging writer for tensorboard
        run_tag (str): the tag of the run

    Returns:
        float: dev F1
        float: test F1
    """
    t_prec = t_recall = t_f1 = t_loss = None
    if any([prefix in task for prefix in ["cleaning_", "em_"]]):  # handle imbalance:
        print("Validation:")
        acc, prec, recall, f1, v_loss, th = eval_classifier(
            model, valid_iter, get_threshold=True
        )
        print("Test:")
        # t_acc, t_prec, t_recall, t_f1, t_loss = eval_classifier(model, test_iter)
        t_acc, t_prec, t_recall, t_f1, t_loss = eval_classifier(
            model, test_iter, threshold=th
        )
        scalars = {
            "acc": acc,
            "precision": prec,
            "recall": recall,
            "f1": f1,
            "v_loss": v_loss,
            "t_acc": t_acc,
            "t_precision": t_prec,
            "t_recall": t_recall,
            "t_f1": t_f1,
            "t_loss": t_loss,
        }
    else:
        print("Validation:")
        v_output = eval_classifier(model, valid_iter)

        if test_iter is not None:
            print("Test:")
            t_output = eval_classifier(model, test_iter)

        if len(v_output) == 5:
            acc, prec, recall, f1, v_loss = v_output
            t_acc, t_prec, t_recall, t_f1, t_loss = t_output
            scalars = {
                "acc": acc,
                "precision": prec,
                "recall": recall,
                "f1": f1,
                "v_loss": v_loss,
                "t_acc": t_acc,
                "t_precision": t_prec,
                "t_recall": t_recall,
                "t_f1": t_f1,
                "t_loss": t_loss,
            }
        else:
            acc, f1, v_loss = v_output
            t_acc, t_f1, t_loss = t_output
            scalars = {
                "acc": acc,
                "f1": f1,
                "v_loss": v_loss,
                "t_acc": t_acc,
                "t_f1": t_f1,
                "t_loss": t_loss,
            }

    # logging
    writer.add_scalars(run_tag, scalars, epoch)
    return f1, t_f1
