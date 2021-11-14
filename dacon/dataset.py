from transformers import InputExample

from dacon.augment import Augmenter
from ditto.dataset import DittoDataset
from rotom.dataset import TextCLSDataset
from snippext.dataset import SnippextDataset, get_tokenizer


class DaconDataset(DittoDataset):
    def __init__(
        self,
        source,
        vocab,
        taskname,
        max_len=512,
        lm="distilbert",
        size=None,
        augmenter=Augmenter(),
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
        self.augmenter = augmenter

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.

        Args:
            idx (int): the element index
        Returns:
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        if self.balance:
            if idx < len(self.pos_sents):
                idx = self.pos_sents[idx]
            else:
                N = len(self.pos_sents)
                idx -= N
                new_idx = self.neg_sents[
                    (idx + self.neg_cnt[idx] * N) % len(self.neg_sents)
                ]
                self.neg_cnt[idx] += 1
                idx = new_idx

        orig_words, tags = self.sents[idx], self.tags_li[idx]
        aug_words = self.augmenter.augment_sent(orig_words)

        # encode orig_words to token_ids
        if " [SEP] " in orig_words:
            sents = orig_words.split(" [SEP] ")
            x = self.tokenizer.encode(
                text=sents[0],
                text_pair=sents[1],
                add_special_tokens=True,
                truncation="longest_first",
                max_length=self.max_len,
            )
        else:
            x = self.tokenizer.encode(
                text=orig_words,
                add_special_tokens=True,
                truncation="longest_first",
                max_length=self.max_len,
            )

        # encode aug_words to token_ids
        if " [SEP] " in aug_words:
            sents = aug_words.split(" [SEP] ")
            try:
                aug_x = self.tokenizer.encode(
                    text=sents[0],
                    text_pair=sents[1],
                    add_special_tokens=True,
                    truncation="longest_first",
                    max_length=self.max_len,
                )
            except:
                print(
                    "Unable to encode augmented data '{}' and '{}'. Falling back to original data.".format(
                        sents[0], sents[1]
                    )
                )
                sents = orig_words.split(" [SEP] ")
                aug_x = self.tokenizer.encode(
                    text=sents[0],
                    text_pair=sents[1],
                    add_special_tokens=True,
                    truncation="longest_first",
                    max_length=self.max_len,
                )
        else:
            try:
                aug_x = self.tokenizer.encode(
                    text=aug_words,
                    add_special_tokens=True,
                    truncation="longest_first",
                    max_length=self.max_len,
                )
            except:
                print("Unable to encode augmented data. Falling back to original data.")
                aug_x = self.tokenizer.encode(
                    text=orig_words,
                    add_special_tokens=True,
                    truncation="longest_first",
                    max_length=self.max_len,
                )

        y = self.tag2idx[tags]  # label
        orig_is_heads, orig_mask, orig_seqlen = [1] * len(x), [1] * len(x), len(x)
        aug_is_heads, aug_mask, aug_seqlen = (
            [1] * len(aug_x),
            [1] * len(aug_x),
            len(aug_x),
        )

        assert (
            len(x) == len(orig_mask) == len(orig_is_heads)
        ), f"len(x) = {len(x)}, len(y) = {len(y)}, len(orig_is_heads) = {len(orig_is_heads)}"

        assert (
            len(aug_x) == len(aug_mask) == len(aug_is_heads)
        ), f"len(aug_x)={len(aug_x)}, len(y)={len(y)}, len(aug_is_heads) = {len(aug_is_heads)}"

        return (
            orig_words,
            x,
            orig_is_heads,
            tags,
            orig_mask,
            y,
            orig_seqlen,
            self.taskname,
        ), (
            aug_words,
            aug_x,
            aug_is_heads,
            tags,
            aug_mask,
            y,
            aug_seqlen,
            self.taskname,
        )


class DaconBaseDataset(SnippextDataset):
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.

        Args:
            idx (int): the element index

        Returns:
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        if self.augment_op is not None:
            return self.get(idx, [(self.augment_op, [])])
        else:
            return self.get(idx, [])

    def get(self, idx, ops=[], aug_idx=None):
        """Return the ith item of in the dataset and apply the transformation ops.

        The ops are of the form [(op_1, args_1), ..., (op_k, args_k)]

        Args:
            idx (int): the element index
            ops (list): the list of operators

        Returns:
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        orig_text_a = self.examples[idx].text_a
        orig_text_b = self.examples[idx].text_b
        if orig_text_b and not orig_text_b.strip():
            orig_text_b = None
        label = self.examples[idx].label

        orig_combined = (
            orig_text_a + " [SEP] " + orig_text_b if orig_text_b else orig_text_a
        )
        aug_combined = orig_combined

        if self.augmenter:
            aug_combined = self.augmenter.augment_sent(aug_combined)

        if " [SEP] " in aug_combined:
            aug_text_a, aug_text_b = aug_combined.split(" [SEP] ")
        else:
            aug_text_a, aug_text_b = aug_combined, None

        x = self.tokenizer.encode(
            text=orig_text_a,
            text_pair=orig_text_b,
            add_special_tokens=True,
            truncation=True,
            truncation_strategy="longest_first",
            max_length=self.max_len,
        )

        aug_x = self.tokenizer.encode(
            text=aug_text_a,
            text_pair=aug_text_b,
            add_special_tokens=True,
            truncation=True,
            truncation_strategy="longest_first",
            max_length=self.max_len,
        )

        if self.tag2idx:
            # regression
            y = float(label)
        else:
            y = self.tag2idx[label] if label in self.tag2idx else 0

        orig_is_heads, aug_is_heads = [1] * len(x), [1] * len(aug_x)
        orig_mask, aug_mask = [1] * len(x), [1] * len(aug_x)
        orig_seqlen, aug_seqlen = len(x), len(aug_x)

        assert (
            len(x) == len(orig_mask) == len(orig_is_heads)
        ), f"len(x) = {len(x)}, len(y)={len(y)}, len(orig_is_heads)={len(orig_is_heads)}"

        assert (
            len(aug_x) == len(aug_mask) == len(aug_is_heads)
        ), f"len(aug_x) = {len(aug_x)}, len(y)={len(y)}, len(aug_is_heads)={len(aug_is_heads)}"

        orig_words = (
            orig_text_a + " [SEP] " + orig_text_b if orig_text_b else orig_text_a
        )
        aug_words = aug_text_a + " [SEP] " + aug_text_b if aug_text_b else aug_text_a

        return (
            orig_words,
            x,
            orig_is_heads,
            label,
            orig_mask,
            y,
            orig_seqlen,
            self.taskname,
        ), (
            aug_words,
            x,
            aug_is_heads,
            label,
            aug_mask,
            y,
            aug_seqlen,
            self.taskname,
        )


class DaconTextCLSDataset(TextCLSDataset):
    def __init__(
        self,
        path,
        vocab,
        taskname,
        max_len=512,
        lm="distilbert",
        augmenter=Augmenter(),
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
        self.augmenter = augmenter
