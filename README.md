# DACon
Code for CS 8803 DMM Project: Data Augmentation for Entity Matching using Consistency Learning

## Requirements

* Python 3.7+
* PyTorch 1.10.0+cu111: default CUDA version 11.1 (change the `--find-links` in `requirements.txt` for other versions)
* Transformers 4.12.3
* NVIDIA Apex (fp16 training): requires Nvidia graphic card

Install required packages
```
pip install -r requirements.txt
# Apex requires CUDA-supported graphic cards (Nvidia graphic cards)
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Experiment Results

### F1 scores on all EM datasets
|         Method          | Abt-Buy | Amazon-Google | DBLP-ACM (clean/dirty) | DBLP-Scholar (clean/dirty) | Walmart-Amazon (clean/dirty) |
|:-----------------------:|:-------:|:-------------:|:----------------------:|:--------------------------:|:----------------------------:|
|      DM + RoBERTa       |  85.71  |     82.35     |      97.92/97.92       |        92.63/91.49         |         72.73/67.92          |
|         RoBERTa         |  80.49  |     65.02     |      98.97/96.63       |        92.18/92.30         |         71.50/73.55          |
|          InvDA          |  84.26  |     59.70     |      96.97/96.51       |        91.99/91.69         |         71.85/73.48          |
|          Rotom          |  76.34  |     62.38     |      96.76/97.16       |        91.80/91.63         |         66.28/76.66          |
|       Rotom + SSL       |  81.89  |     62.34     |      98.09/97.20       |        92.88/92.97         |         72.19/71.55          |
|     DACon Baseline      |  81.06  |     62.95     |      96.73/96.98       |        92.42/92.00         |         73.71/71.91          |
|    DACon One-to-Many    |  80.21  |     61.06     |      96.96/97.19       |        91.94/91.45         |         76.88/75.14          |
| DACon Fixed Consistency |  83.60  |     62.87     |      92.29/96.40       |        91.33/91.69         |         78.93/74.18          |
|    DACon Consistency    |  80.81  |     60.30     |      97.29/95.95       |        92.02/91.86         |         74.60/72.11          |

### Average training time with different training + validation size
|         Method          |  300   |  450   |  600   |  780   |
|:-----------------------:|:------:|:------:|:------:|:------:|
|      DM + RoBERTa       | 65.17  | 90.46  | 114.28 | 139.11 |
|         RoBERTa         | 100.67 | 111.24 | 121.17 | 132.65 |
|          InvDA          | 112.45 | 130.90 | 148.07 | 166.12 |
|          Rotom          | 165.73 | 216.40 | 263.81 | 313.18 |
|       Rotom + SSL       | 165.73 | 216.17 | 260.64 | 313.51 |
|     DACon Baseline      | 168.78 | 182.93 | 196.23 | 211.42 |
|    DACon One-to-Many    | 190.14 | 216.42 | 238.87 | 264.89 |
| DACon Fixed Consistency | 190.93 | 216.88 | 240.85 | 266.48 |
|    DACon Consistency    | 185.49 | 218.16 | 240.49 | 266.91 |

See the figures [here](exp_figures).

## Model Training

To train a model with Rotom:
```
CUDA_VISIBLE_DEVICES=0 python train_any.py \
  --task em_DBLP-ACM \
  --size 300 \
  --logdir results_em/ \
  --finetuning \
  --batch_size 32 \
  --lr 3e-5 \
  --n_epochs 20 \
  --max_len 128 \
  --fp16 \
  --lm roberta \
  --da dacon_baseline \
  --balance \
  --run_id 0
```

The current version supports 3 tasks: entity matching (EM), error detection (EDT), and text classification (TextCLS). The supported tasks are:

| Type    | Dataset Names                                                        | taskname pattern                         |
|---------|----------------------------------------------------------------------|------------------------------------------|
| EM      | Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-GoogleScholar, Walmart-Amazon | em_{dataset}, e.g., em_DBLP-ACM          |
| EDT     | beers, hospital, movies, rayyan, tax                                 | cleaning_{dataset}, e.g., cleaning_beers |
| TextCLS | AG, AMAZON2, AMAZON5, ATIS, IMDB, SNIPS, SST-2, SST-5, TREC          | textcls_{dataset}, e.g., textcls_AG      |
| TextCLS, splits from [Hu et al.](https://arxiv.org/pdf/1910.12795.pdf) | IMDB, SST-5, TREC | compare1_{dataset}, e.g., compare1_IMDB |
| TextCLS, splits from [Kumar et al.](https://arxiv.org/pdf/2003.02245.pdf) | ATIS, SST-2, TREC | compare2_{dataset}, e.g., compare2_ATIS |

Parameters:
* ``--task``: the taskname pattern specified following the above table
* ``--size``: the dataset size (optional). If not specified, the entire dataset will be used. The size ranges are {300, 450, 600, 750} for EM, {50, 100, 150, 200} For EDT, and {100, 300, 500} for TextCLS
* ``--logdir``: the path for TensorBoard logging (F1, acc, precision, and recall)
* ``--finetuning``: always keep this flag on
* ``--batch_size``, ``--lr``, ``--max_len``, ``--n_epochs``: the batch size, learning rate, max sequence length, and the number of epochs for model training
* ``--fp16``: whether to use half-precision for training
* ``--lm``: the language model to fine-tune. We currently support bert, distilbert, and roberta
* ``--balance``: a special option for binary classification (EM and EDT) with skewed labels (#positive labels >> #negative labels). If this flag is on, then the training process will up-sample the positive labels
* ``--warmup``: (new) if this flag is on with SSL, then first warm up the model by training it on labeled data only before running SSL. Only support EM for now.
* ``--run_id``: the integer ID of the run e.g., {0, 1, 2, ...}
* ``--da``: the data augmentation method (See table below)

|                      Method                       |                                                                              Operator Name(s)                                                                             |
|:-------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|           No DA (simply LM fine-tuning)           |                                                                                    None                                                                                   |
|          Regular transformation-based DA          | ``['del', 'drop_col', 'append_col', 'swap', 'ins']`` for EM/EDT <br> ``['del', 'token_del_tfidf', 'token_del', 'shuffle', 'token_repl', 'token_repl_tfidf']`` for TextCLS |
|                Inversed DA (InvDA)                |                                                                                 t5 / invda                                                                                |
|       Rotom (w/o semi-supervised learning)        |                                                                         auto_filter_weight_no_ssl                                                                         |
|        Rotom (w. semi-supervised learning)        |                                                                             auto_filter_weight                                                                            |
|                  DACon Baseline                   |                                                                             dacon_baseline                                                                                |
|                 DACon One-to-Many                 |                                                                             dacon_one_to_many                                                                             |
|              DACon Fixed Consistency              |                                                               dacon_fixed_consistency                                                                       |
|                 DACon Consistency                 |                                                               dacon_consistency                                                                             |

For the invda fine-tuning, see ``invda/README.md``.


## Experiment scripts

All experiment scripts are available in ``scripts/``. To run the experiments for a task (em, cleaning, or textcls):
```
python scripts/run_all_em.py
```
