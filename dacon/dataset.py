from augment import Augmenter
from transformers import InputExample

from ditto.dataset import DittoDataset
from rotom.dataset import TextCLSDataset
from snippext.dataset import get_tokenizer


class DaconDataset(DittoDataset):
    def __init__(
        self,
        source,
        vocab,
        taskname,
        max_len=512,
        lm="distilbert",
        size=None,
        augment_op=None,
        balance=False,
    ):
        super().__init__(
            source=source,
            vocab=vocab,
            taskname=taskname,
            max_len=max_len,
            lm=lm,
            size=size,
            balance=balance,
        )
        # augmentation op
        self.augment_op = augment_op
        if augment_op:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


class DaconTextCLSDataset(TextCLSDataset):
    def __init__(
        self,
        path,
        vocab,
        taskname,
        max_len=512,
        lm="distilbert",
        augment_op=None,
        size=None,
    ):
        self.taskname = taskname
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = get_tokenizer(lm=lm)
        # read path
        self.examples = []
        for uid, line in enumerate(open(path)):
            LL = line.strip().split("\t")
            if len(LL) == 2:
                e = InputExample(uid, LL[0], None, LL[1])
            elif len(LL) == 3:
                e = InputExample(uid, LL[0], LL[1], LL[2])
            self.examples.append(e)
            if size is not None and len(self.examples) >= size:
                break

        # vocab
        if None in self.vocab:
            # regression task
            self.tag2idx = self.idx2tag = None
        else:
            self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab)}
            self.idx2tag = {idx: tag for idx, tag in enumerate(self.vocab)}

        # augmentation
        self.augmenter = Augmenter()
        self.augment_op = augment_op
