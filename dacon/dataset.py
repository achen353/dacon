import numpy as np
import torch
from transformers import InputExample

from dacon.augment import Augmenter
from ditto.dataset import DittoDataset
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
        aug_words_list = self.augmenter.augment_sent(orig_words)

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

        y = self.tag2idx[tags]  # label
        orig_is_heads, orig_mask, orig_seqlen = [1] * len(x), [1] * len(x), len(x)

        assert (
            len(x) == len(orig_mask) == len(orig_is_heads)
        ), f"len(x) = {len(x)}, len(y) = {len(y)}, len(orig_is_heads) = {len(orig_is_heads)}"

        # encode aug_words to token_ids
        aug_x_results = []
        for aug_words in aug_words_list:
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
                    print(
                        "Unable to encode augmented data. Falling back to original data."
                    )
                    aug_x = self.tokenizer.encode(
                        text=orig_words,
                        add_special_tokens=True,
                        truncation="longest_first",
                        max_length=self.max_len,
                    )

            aug_is_heads, aug_mask, aug_seqlen = (
                [1] * len(aug_x),
                [1] * len(aug_x),
                len(aug_x),
            )

            assert (
                len(aug_x) == len(aug_mask) == len(aug_is_heads)
            ), f"len(aug_x)={len(aug_x)}, len(y)={len(y)}, len(aug_is_heads) = {len(aug_is_heads)}"

            aug_x_results.append(
                (
                    aug_words,
                    aug_x,
                    aug_is_heads,
                    tags,
                    aug_mask,
                    y,
                    aug_seqlen,
                    self.taskname,
                )
            )

        return (
            orig_words,
            x,
            orig_is_heads,
            tags,
            orig_mask,
            y,
            orig_seqlen,
            self.taskname,
        ), aug_x_results

    @staticmethod
    def pad(batch):
        """Pads to the longest sample

        Args:
            batch:

        Returns:
            return words, f(x), is_heads, tags, f(mask), f(y), seqlens, name
        """
        f_orig = lambda x: [sample[0][x] for sample in batch]
        g_orig = lambda x, seqlen, val: [
            sample[0][x] + [val] * (seqlen - len(sample[0][x])) for sample in batch
        ]

        def f_aug(x):
            output = []
            for sample in batch:
                for aug_x_result in sample[1]:
                    output.append(aug_x_result[x])
            return output

        def g_aug(x, seqlen_list, val):
            output = []
            for sample in batch:
                for i, aug_x_result in enumerate(sample[1]):
                    output.append(
                        aug_x_result[x] + [val] * seqlen_list[i] - len(aug_x_result[x])
                    )

        # get task name
        tags, name = f_orig(3), f_orig(7)

        orig_seqlens = f_orig(6)
        orig_maxlen = np.array(orig_seqlens).max()
        orig_words, orig_is_heads, x, orig_mask = (
            f_orig(0),
            f_orig(2),
            g_orig(1, orig_maxlen, 0),
            g_orig(4, orig_maxlen, 1),
        )
        y = g_orig(5, orig_maxlen, 0) if "_tagging" in name[0] else f_orig(5)

        aug_seqlens_list = f_aug(6)
        aug_maxlen_list = [
            np.array(aug_seqlens).max() for aug_seqlens in aug_seqlens_list
        ]
        aug_words_list, aug_is_heads_list, aug_x_list, aug_mask_list = (
            f_aug(0),
            f_aug(2),
            g_aug(1, aug_maxlen_list, 0),
            g_aug(4, aug_maxlen_list, 1),
        )
        aug_y_list = g_aug(5, aug_maxlen_list, 0) if "_tagging" in name[0] else f_aug(5)

        y = torch.Tensor(y) if isinstance(y[0], float) else torch.LongTensor(y)
        aug_y_list = [
            (
                torch.Tensor(aug_y)
                if isinstance(aug_y[0], float)
                else torch.LongTensor(aug_y)
            )
            for aug_y in aug_y_list
        ]

        aug_tags, aug_name = [tags] * len(aug_x_list), [name] * len(aug_x_list)

        t = torch.LongTensor

        aug_x_list = [t(aug_x) for aug_x in aug_x_list]
        aug_mask_list = [t(aug_mask) for aug_mask in aug_mask_list]

        aug_x_results = list(
            zip(
                aug_words_list,
                aug_x_list,
                aug_is_heads_list,
                aug_tags,
                aug_mask_list,
                aug_y_list,
                aug_seqlens_list,
                aug_name,
            )
        )

        return (
            orig_words,
            t(x),
            orig_is_heads,
            tags,
            t(orig_mask),
            y,
            orig_seqlens,
            name,
        ), aug_x_results


class DaconBaseDataset(SnippextDataset):
    def __init__(
        self,
        source,
        vocab,
        taskname,
        max_len=512,
        lm="bert",
        augmenter=Augmenter(),
        size=None,
    ):
        super().__init__(
            source=source,
            vocab=vocab,
            taskname=taskname,
            max_len=max_len,
            lm=lm,
            size=size,
        )
        self.augmenter = augmenter

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.

        Args:
            idx (int): the element index

        Returns:
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
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

        aug_combined_list = self.augmenter.augment_sent(orig_combined)

        x = self.tokenizer.encode(
            text=orig_text_a,
            text_pair=orig_text_b,
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

        orig_is_heads, orig_mask, orig_seqlen = [1] * len(x), [1] * len(x), len(x)

        assert (
            len(x) == len(orig_mask) == len(orig_is_heads)
        ), f"len(x) = {len(x)}, len(y)={len(y)}, len(orig_is_heads)={len(orig_is_heads)}"

        orig_words = (
            orig_text_a + " [SEP] " + orig_text_b if orig_text_b else orig_text_a
        )

        aug_x_results = []
        for aug_combined in aug_combined_list:
            if " [SEP] " in aug_combined:
                aug_text_a, aug_text_b = aug_combined.split(" [SEP] ")
            else:
                aug_text_a, aug_text_b = aug_combined, None

            aug_x = self.tokenizer.encode(
                text=aug_text_a,
                text_pair=aug_text_b,
                add_special_tokens=True,
                truncation=True,
                truncation_strategy="longest_first",
                max_length=self.max_len,
            )

            aug_is_heads, aug_mask, aug_seqlen = (
                [1] * len(aug_x),
                [1] * len(aug_x),
                len(aug_x),
            )

            assert (
                len(aug_x) == len(aug_mask) == len(aug_is_heads)
            ), f"len(aug_x) = {len(aug_x)}, len(y)={len(y)}, len(aug_is_heads)={len(aug_is_heads)}"

            aug_words = (
                aug_text_a + " [SEP] " + aug_text_b if aug_text_b else aug_text_a
            )

            aug_x_results.append(
                (
                    aug_words,
                    x,
                    aug_is_heads,
                    label,
                    aug_mask,
                    y,
                    aug_seqlen,
                    self.taskname,
                )
            )

        return (
            orig_words,
            x,
            orig_is_heads,
            label,
            orig_mask,
            y,
            orig_seqlen,
            self.taskname,
        ), aug_x_results


class DaconTextCLSDataset(DaconBaseDataset):
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
