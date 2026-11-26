# Paper experiment details

This document records the configuration and supplementary results reported in the paper. The main evaluation uses 10,000 English documents from eight Pile-uncopyrighted domains. The repository's runnable example dataset contains 100 Reuters / CoNLL-2003 news passages. These are different evaluation sets; running the examples produces scores for those 100 inputs, not a reproduction of the paper's 10,000-document aggregate. The defense study below uses 48 development inputs.

## Models and training

The frozen target is [AntonV/mamba2-130m-av](https://huggingface.co/AntonV/mamba2-130m-av/tree/9f0dd145df8556bbf2de843811ccc6b49d27024d). Inversion decoders use its pretrained initialization and `Mamba2ForCausalLM`: 24 layers, hidden width 768, expansion width 1,536, 24 heads of width 64, state size 128, convolution width 4, one SSM group, and tied embeddings for 50,288 tokens. Token 0 is used as BOS, EOS, and padding. Target encoding adds no special tokens.

Training uses the [Pile-uncopyrighted training split](https://huggingface.co/datasets/monology/pile-uncopyrighted) and next-token cross-entropy including EOS.

| Decoder input length | Training |
| --- | --- |
| 4, 8, 16, 32, 64, 128 tokens | 5,000 AdamW steps on the first 600 downloaded documents; batch size 1; constant learning rate 1e-5 |
| 256 tokens, first stage | 6,000 AdamW steps on a random 20,000-document subset; batch size 4; learning rate 1e-5; weight decay 0 |
| 256 tokens, second stage | 3,000 Adafactor steps on distinct segments from the first 20,000 streamed documents; batch size 1; one fixed-order pass; target learning rate 1e-5; weight decay 0 |

The Adafactor stage uses 10 warm-up steps, clipping 1.0, FP32 parameters with BF16 autocasting, sampling seed 20261010, and training seed 20261009. The final 256-token decoder has 9,000 total updates and weight SHA-256 prefix `9ecdb98306fef0c8`; the C0 diagnostic decoder is the checkpoint after 6,000 updates.

Checkpoint identifiers in the experiment records are `sweep_csN/step_5000` for N=4,8,16,32; `v2fix_csN/step_5000` for N=64,128; `broad_training_20261010_clean/step_3000` for the final 256-token decoder; and `m2_v2_fix256/step_6000` for C0. These identifiers describe the experiment records, not additional files required by the example runner. The bundled weight manifest identifies the supplied decoder.

## Inference and development selection

The reported inference environment uses an A800 80 GB PCIe GPU, batch size 1, FP32 with TF32 disabled, PyTorch 2.3.1+cu121, Transformers 4.44.2, and four CPU threads. Direct and State-Consistent Inference use the same fixed decoder, inputs, caches, and 256-token limit in the main comparison. Candidate rollouts use independent cache copies.

A separate set of 241 sequences supports development and diagnosis. On C0's first 24 development inputs, layer 19 maximizes 32-branch SSM-MSE-selection ROUGE-1 F1 across layers 0–23. Threshold candidates include zero, observed nonnegative relative repair gains, and the next representable value above their maximum. Maximizing F1, favoring larger thresholds within 1e-12, selects tau=0.027218597339261106. Maximizing F1 subject to a precision loss no greater than 0.02 selects a 64-token extension cap. These settings are fixed before evaluation.

The likelihood policy B1 sums content and EOS log probabilities, applies a probability floor of 1e-12, and divides by max(m,1), where m is the content length. Length-limited outputs omit EOS. Candidate replacement requires strict improvement over the fallback candidate, and the 64-token extension cap remains. B1 uses likelihood for intermediate choices and final selection.

## Main evaluation data

The main evaluation draws from the Pile-uncopyrighted test split at revision `3be90335b66f24456a5d6659d9c8d208c0357119`.

| Domain | Documents |
| --- | ---: |
| News / current affairs | 1,000 |
| General web text | 1,000 |
| Law | 1,000 |
| Patents | 1,000 |
| Science | 1,500 |
| Biomedicine | 1,500 |
| Encyclopedic knowledge | 1,500 |
| Question answering | 1,500 |
| Total | 10,000 |

Seed 20261120 selects within domain quotas by normalized-text hash. Filters require 200–20,000 English words and at least 256 model tokens. Normalized full texts and first-256-token sequences are exactly deduplicated. Domains follow Pile source labels, except for rule-based news selection within Pile-CC. Near-duplicate and training-pool overlap audits remain incomplete; target-pretraining overlap is unverified.

Each document supplies one 256-token segment and its 4-, 8-, 16-, 32-, 64-, and 128-token prefixes. Each input length therefore covers the same 10,000 documents. The 256-token inputs also serve for ablations and timing.

## Scoring

For reference r_i, reconstruction c_i, and metric q, report the arithmetic mean `(1/n) * sum_i q(r_i,c_i)` over the inputs in the relevant evaluation. The main evaluation uses n=10,000; the defense study uses n=48. Precision, recall, and F1 are averaged separately; aggregate F1 is not calculated from aggregate precision and recall.

- ROUGE-1/2 measures unigram/bigram overlap; ROUGE-L uses longest common subsequences. The implementation uses `rouge-score` with stemming.
- Token-set F1 uses sets A and B of token IDs, excluding special tokens and repetitions: `2 * |A intersection B| / (|A| + |B|)`. The score is zero if either set is empty.
- BGE similarity is the cosine of the reference and reconstruction embeddings: `dot(e(r), e(c)) / (norm(e(r)) * norm(e(c)))`. The model is `BAAI/bge-large-en-v1.5`.
- BERTScore matches contextual token embeddings. The model is `sentence-transformers/all-mpnet-base-v2`, layer 12, without IDF weighting or baseline rescaling.

Semantic scoring uses `bert-score` and `sentence-transformers`, batch size 1, and FP32 without TF32 or autocasting. The repository's scoring script provides executable metric definitions.

## Direct recovery across input lengths

All rows below summarize the same 10,000 source documents with length-specific decoders.

| Tokens | ROUGE-1 F1 | ROUGE-L F1 | BGE | BERTScore F1 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.9826 | 0.9817 | 0.9923 | 0.9938 |
| 8 | 0.9760 | 0.9754 | 0.9879 | 0.9805 |
| 16 | 0.9767 | 0.9766 | 0.9885 | 0.9844 |
| 32 | 0.9394 | 0.9382 | 0.9690 | 0.9550 |
| 64 | 0.8725 | 0.8690 | 0.9409 | 0.9259 |
| 128 | 0.7229 | 0.7105 | 0.8785 | 0.8289 |
| 256 | 0.2768 | 0.2499 | 0.6702 | 0.5131 |

## Timing and allocated memory

Timings synchronize the GPU and cover generation, prefix replay, and verification where applicable. Direct includes initial target encoding; the other methods start from prepared caches. Loading and final scoring are excluded. Per-input memory counters include resident models, including both target and decoder for B1, B2, and the full method. Peak memory is maximum allocated GPU memory across inputs, not total GPU capacity or total system memory.

| Method | Mean (s/input) | Median (s/input) | P95 (s/input) | Peak allocated memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Direct | 3.29 | 4.66 | 5.09 | 2209.72 |
| Likelihood policy (B1) | 225.06 | 219.89 | 295.50 | 1041.41 |
| 64-branch search (B2) | 195.62 | 196.48 | 263.68 | 2228.38 |
| State-Consistent Inference | 237.30 | 238.85 | 297.62 | 2228.38 |

## Cache perturbation and continuation utility

This study uses the first 48 development inputs. All terminal SSM and convolution states are transformed once, and the attacker and target receive the same transformed cache. Computation uses FP32 with fixed decoder weights.

For a state tensor T, Gaussian noise has standard deviation `alpha * sqrt(mean(T^2))`, with alpha in {0.001, 0.01, 0.1}. Symmetric b-bit quantization uses `qmax = 2^(b-1) - 1` and `s = max(abs(T)) / qmax`: round T/s, clip to [-qmax,qmax], and rescale by s. Groups span the last two SSM dimensions or the last convolution dimension. Each group has its own scale; zero groups remain zero.

Continuation perplexity is `exp(-sum_i sum_t log p(u_i,t | C'_i, u_i,0:t-1) / (48 * 32))`, for i=1..48 and t=1..32. Here C'_i is the transformed cache and u_i is the reference continuation. The true first token u_i,0 warms up the state without scoring; the next 32 tokens are scored with teacher forcing. Lower perplexity is better. These historical continuation streams may cross document boundaries.

The following table uses an unchanged verifier and the first noise-0.01 realization.

| Cache | ROUGE-1 F1 | ROUGE-L F1 | BGE | BERTScore F1 | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.6706 | 0.6212 | 0.9079 | 0.8735 | 23.315 |
| Noise 0.001 | 0.6665 | 0.6121 | 0.9076 | 0.8729 | 23.318 |
| Noise 0.01 | 0.6751 | 0.6263 | 0.9143 | 0.8797 | 23.376 |
| Noise 0.1 | 0.4607 | 0.4033 | 0.8285 | 0.7528 | 29.540 |
| 8-bit quantization | 0.6005 | 0.5431 | 0.8888 | 0.8282 | 25.101 |
| 4-bit quantization | 0.1032 | 0.0829 | 0.5876 | 0.4240 | 171.400 |

Across three noise-0.01 realizations, ROUGE-1 F1 is 0.6751, 0.6727, and 0.6791 (clean: 0.6706). Applying the same quantization to candidate layer-19 states, using candidate-derived scales, yields ROUGE-1 F1 of 0.6008 at 8 bits and 0.1026 at 4 bits.
