# Mamba-2 reconstruction evaluation

This artifact compares direct generation and State-Consistent Inference on **100 English news passages from the Reuters corpus used in CoNLL-2003**. Each input has 256 model tokens. Both methods used the same decoder. The artifact evaluates saved outputs from those runs and reports paired reconstruction scores.

## Start from a passage

`review_input.py` runs the public Mamba-2 backbone on text you supply. In a Python environment with PyTorch, install the optional dependencies and try a short passage on CPU. The first live run downloads about 516 MB of backbone weights. A local copy of the same model can be passed with `--model-dir`. [Backbone and model card](https://huggingface.co/AntonV/mamba2-130m-av/tree/9f0dd145df8556bbf2de843811ccc6b49d27024d)

```sh
python -m pip install torch==2.10.0 transformers==4.57.6 rouge-score==0.1.2
python review_input.py --text "The morning train reached the station at nine."
```

For a new passage, the live pass reports the token count, number of layers and whether the model produced a cache. The reconstruction evidence below comes from a recorded run of the research decoder. This release does not run that decoder on arbitrary text.

### Inspect the Figure 4 example

The archived run for passage 68 contains four outputs. ROUGE-1 F1 was 0.5510 for direct generation, 0.7359 after candidate search, 0.7381 after local repair, and 0.7359 after the EOS check. To display the archived scores and output lengths using only the Python standard library, run

```sh
python review_input.py --replay-only
```

If you have the original text of passage 68, save it as `case68.txt` in UTF-8 and run

```sh
python review_input.py --text-file case68.txt
```

The script matches the reference by SHA-256, restores the original 256-token segmentation and runs the backbone. Separately, it reads the four **historical** output texts from the saved trace, verifies their hashes and recalculates ROUGE-1 F1 and ROUGE-L F1 directly against your reference. Add `--replay-only` to skip the backbone pass. `case68.json` stores the outputs as character spans in the reference plus literal edits, so readers supply their own copy of the Reuters passage. The live cache is not used to produce the archived outputs.

## Evaluate all 100 passages

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
