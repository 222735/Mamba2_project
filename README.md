# Mamba-2 text reconstruction

This reviewer artifact runs direct generation and State-Consistent Inference on **100 English news passages from Reuters / CoNLL-2003**, with 256 model tokens per input. Both methods use the same trained decoder. The runner creates terminal SSM and convolution states from each supplied input, independently generates both reconstructions, and saves the new predictions for scoring.

## Run the experiment

Use Python 3.11 on Linux with a CUDA-enabled PyTorch installation for GPU inference. About 3 GiB of free GPU memory is a practical starting point for serial FP32 generation. The CPU fallback is available but substantially slower. Dependency versions below match the research environment for PyTorch and Transformers. Optional Mamba CUDA kernels are unnecessary for the bundled PyTorch implementation.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_decoder.py
python run_experiment.py --limit 1 --output results/smoke.json
python score_predictions.py --input results/smoke.json --output results/smoke_metrics.json
```

The first run downloads the public backbone and the semantic scoring models. The trained decoder is supplied as a GitHub release asset, with file and archive SHA-256 checks. To evaluate all 100 inputs, run

```sh
python run_experiment.py --limit 100 --output results/run.json --resume
python score_predictions.py --input results/run.json --output results/metrics.json
```

Each completed input is saved atomically. If interrupted, repeat the same generation command with `--resume`. The runner verifies dataset, decoder, backbone, source and runtime identities before reusing rows. It never reads the historical scoring CSV to generate predictions. The reference text is used only to check input tokenization before inference and to score the completed predictions. Candidate selection uses the terminal cache and model outputs.

For CPU generation, add `--device cpu`. To score lexical metrics without downloading semantic models, add `--lexical-only` to `score_predictions.py`. The full scorer uses ROUGE with stemming and macro averaging, model-token-set F1, BGE cosine (`BAAI/bge-large-en-v1.5`) and BERTScore F1 (`sentence-transformers/all-mpnet-base-v2`, layer 12, no baseline rescaling). Semantic scoring uses FP32, batch size 1 and CPU by default. `--device cuda` selects GPU scoring.

## Release validation

The released runner was executed end to end on sample 0 using the research server. Both newly generated prediction texts matched the archived run exactly. All lexical and semantic metrics were recomputed; the optimized ROUGE-1 F1 was 0.8725 for this input. Resuming the completed one-input run preserved every output. `validation.json` records the checks and source identities. This acceptance check regenerated one input; the table below summarizes the historical 100-input run.

## Files

- `data/samples.jsonl` contains the 100 plaintext references, original 256-token input sequences and sample IDs. Exact input tokens avoid changes caused by decoding and re-tokenizing a stream fragment.
- `download_decoder.py` downloads the trained decoder. `decoder_manifest.json` identifies its release asset and checksums. Model weights are placed in `checkpoints/decoder`.
- `run_experiment.py` runs input encoding, direct generation and the complete candidate-search, state-verification, local-repair and EOS-check procedure.
- `src/inference.py` contains the inference policy and cache backend. `src/ouroboros/` contains the model implementation and state encoder. `src/topk_prefix_compat_20261014.py` preserves the verified exact-tie handling.
- `score_predictions.py` evaluates the newly generated texts and writes per-input metrics and means.
- `results/run.json` is created by generation. Each row contains both predictions, generated token IDs, candidate trace and measured costs. `results/metrics.json` is created by scoring.
- `selection.json`, `paired_evaluation.csv` and `evaluate.py` retain the historical scoring records and their CPU-only evaluator.
- `review_input.py` and `case68.json` retain the earlier single-case archival demonstration.

## Model and protocol

The encoder is [`AntonV/mamba2-130m-av`](https://huggingface.co/AntonV/mamba2-130m-av), pinned to revision `9f0dd145df8556bbf2de843811ccc6b49d27024d`. The decoder asset is the trained original architecture after the additional training stage, used by the published research run. No training is performed by this artifact.

Inputs use zero-initialized encoder states. Generation uses FP32, disables TF32, starts from BOS 0 and stops on EOS 0 or the 256-token cap. Each rollout receives an independent cache copy. The full method preserves the verified research policy in `EngineArgs`. Inference receives a cache produced by the supplied public sample on the reviewer's own model instance.

The expected results below describe the historical outputs for these 100 inputs. A fresh run writes its own measurements. Hardware, numerical libraries and exact-score ties can affect regenerated predictions. A one-input smoke test is available before committing to the full serial run.

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

## Data source and attribution

The passages are English news text from the Reuters corpus used for CoNLL-2003. See the [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/) and [Reuters access and terms](https://trec.nist.gov/data/reuters/reuters.html). Corpus terms apply to the sample text. Upstream code notices and the Apache-2.0 text are included in `NOTICE.txt` and `LICENSE-APACHE-2.0.txt`.
