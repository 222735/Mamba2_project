# Mamba-2 reconstruction evaluation

This artifact compares direct generation and State-Consistent Inference on **100 English news passages from the Reuters corpus used in CoNLL-2003**. Each input has 256 model tokens. Both methods used the same decoder. The artifact evaluates saved outputs from those runs and reports paired reconstruction scores.

## Run

From the repository directory, run the following with Python 3.8 or later. Evaluation uses the standard library and runs on a CPU.

```sh
python evaluate.py
```

For a paired bootstrap interval on the mean ROUGE-1 F1 difference, keep both methods' scores together when resampling each input. The number of resamples and the random seed can be set on the command line.

```sh
python evaluate.py --bootstrap 2000 --seed 123
```

## Results on 100 news passages

| Metric | Direct | State-Consistent Inference | Difference |
| --- | ---: | ---: | ---: |
| ROUGE-1 precision | 0.4469 | 0.8782 | +0.4313 |
| ROUGE-1 recall | 0.2313 | 0.6375 | +0.4062 |
| ROUGE-1 F1 | 0.2674 | 0.7229 | +0.4555 |
| ROUGE-2 F1 | 0.1959 | 0.6630 | +0.4671 |
| ROUGE-L F1 | 0.2442 | 0.6801 | +0.4359 |
| Token-set F1 | 0.2856 | 0.7749 | +0.4893 |
| BGE cosine | 0.6714 | 0.9115 | +0.2401 |
| BERTScore F1 | 0.4969 | 0.8794 | +0.3824 |

## Data and scoring

- [selection.json](selection.json) records the sample indices and random seed used to select the 100 passages.
- [paired_evaluation.csv](paired_evaluation.csv) contains one row per selected input. The prefix `direct_` denotes direct generation and `full_` denotes State-Consistent Inference. Each pair was matched to the same reference text before export.
- [evaluate.py](evaluate.py) computes the five ROUGE columns from the reference length, output length, matching unigram and bigram counts, and longest common subsequence length. The word counts are for the ROUGE tokenizer and may differ from the model's token counts. Text was lowercased and tokenized as in `rouge-score 0.1.2` with stemming enabled. Unigram and bigram overlap use clipped multiplicities. The five per-input ROUGE scores were checked against the archived scorer output for all 200 reconstructions.
- For token-set F1, BGE cosine and BERTScore F1, the CSV contains the original per-input scores. The evaluator computes each method's arithmetic mean across inputs separately for every metric. The optional interval describes variation across the 100 inputs.

The passages used for this demonstration come from English CoNLL-2003. The public artifact distributes numeric scoring records and the selection manifest. Reuters distributes the underlying corpus under separate terms. [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/) · [Reuters corpus access and terms](https://trec.nist.gov/data/reuters/reuters.html)
