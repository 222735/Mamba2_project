# Mamba-2 text reconstruction

Run text reconstruction on 100 English news passages from Reuters / CoNLL-2003. Each input contains 256 model tokens. Direct generation and State-Consistent Inference use the same trained decoder.

```text
Input text and its token sequence
  → Mamba-2 encoder
  → Terminal SSM and convolution states
  → Trained reconstruction decoder
  → Direct generation / State-Consistent Inference
  → Reconstructed text
  → Per-input scores and mean scores
```

State-Consistent Inference performs candidate search, state-consistency verification, local repair and EOS checking. Candidate selection uses model outputs and the input's terminal cache. Reference text is used for input verification and final scoring.

## Install

Use Python 3.11 on Linux. GPU generation requires a CUDA-enabled PyTorch installation. The bundled PyTorch implementation also supports CPU execution.

```sh
git clone https://github.com/222735/Mamba2_project.git
cd Mamba2_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_decoder.py
```

The trained decoder download is about 516 MB. The downloader verifies the archive and model files using the SHA-256 checksums in `decoder_manifest.json`. The first experiment run downloads the public encoder. Full scoring also downloads the BGE and BERTScore models.

## Run all 100 inputs

Run these commands from the repository directory:

```sh
python run_experiment.py --limit 100 --output results/run.json --resume
python score_predictions.py --input results/run.json --output results/metrics.json
```

Generation saves each completed input. Repeat the generation command to resume an interrupted run. Saved rows are checked against the dataset, weights, source code and runtime identities before reuse.

To try one input first:

```sh
python run_experiment.py --limit 1 --output results/example.json --resume
python score_predictions.py --input results/example.json --output results/example_metrics.json
```

Add `--device cpu` to the generation command for CPU execution. Scoring uses CPU by default; `--device cuda` selects GPU scoring. Add `--lexical-only` to the scoring command to compute ROUGE and token-set F1 without the semantic models.

## Inputs and outputs

`data/samples.jsonl` contains one input per line with `sample_id`, `reference` and `input_ids`. Each supplied token sequence has length 256. The original token sequence is included because decoding a stream fragment and re-tokenizing it can change token boundaries. The runner checks that the sequence decodes to the supplied text.

`results/run.json` contains the newly generated direct and optimized predictions, generated tokens, candidate traces and measured costs. `results/metrics.json` contains per-input scores and arithmetic means across the inputs in that run.

Scoring includes ROUGE-1 precision, recall and F1, ROUGE-2 F1, ROUGE-L F1, model-token-set F1, BGE cosine similarity and BERTScore F1. ROUGE uses stemming. BGE uses `BAAI/bge-large-en-v1.5`. BERTScore uses `sentence-transformers/all-mpnet-base-v2`, layer 12, without baseline rescaling. Semantic scoring uses FP32 and batch size 1.

## Files

```text
data/samples.jsonl                    100 inputs
download_decoder.py                  Trained decoder download
decoder_manifest.json                Weight URL and checksums
run_experiment.py                    Input encoding and reconstruction
score_predictions.py                 Scoring generated predictions
requirements.txt                     Python dependencies
src/inference.py                     Inference policy and cache backend
src/ouroboros/encode_dataset.py       Terminal-state encoding
src/ouroboros/models/                Model implementation
src/topk_prefix_compat_20261014.py     Exact-tie compatibility helper
```

Downloaded weights are stored in `checkpoints/decoder/`. Experiment outputs are stored in `results/`.

The encoder is [`AntonV/mamba2-130m-av`](https://huggingface.co/AntonV/mamba2-130m-av), pinned to revision `9f0dd145df8556bbf2de843811ccc6b49d27024d`. The decoder download supplies the trained weights. Generation uses zero-initialized encoder states, FP32, TF32 disabled, BOS/EOS token 0, a 256-token output cap and independent cache copies for rollouts.

## Attribution

The samples contain English news text from the Reuters corpus used for CoNLL-2003. See the [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/) and [Reuters access and terms](https://trec.nist.gov/data/reuters/reuters.html). Corpus terms apply to the sample text. Upstream code notices and the Apache-2.0 license are included in `NOTICE.txt` and `LICENSE-APACHE-2.0.txt`.
