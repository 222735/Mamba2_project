"""Score newly generated texts, with the research ROUGE and semantic settings."""
import argparse
import gc
import json
from pathlib import Path
from statistics import fmean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/run.json"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lexical-only", action="store_true", help="Skip the two semantic models")
    cli = parser.parse_args()
    result = json.loads(cli.input.read_text(encoding="utf-8"))
    rows = result["rows"]
    if not result["complete"] or len(rows) != result["requested_samples"]:
        raise ValueError("Generation must finish the requested sample count before scoring")
    import torch
    from rouge_score.rouge_scorer import RougeScorer
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-av",
        revision="9f0dd145df8556bbf2de843811ccc6b49d27024d")
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    references = [row["reference"].strip() for row in rows]
    texts = {"direct": [row["direct"]["text"].strip() for row in rows],
             "full": [row["full"]["reconstruction"].strip() for row in rows]}
    scores = {method: {} for method in texts}
    for method, predictions in texts.items():
        per_row = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        for rouge in ("rouge1", "rouge2", "rougeL"):
            for suffix, attribute in (("p", "precision"), ("r", "recall"), ("f1", "fmeasure")):
                scores[method][f"{rouge}_{suffix}"] = [getattr(r[rouge], attribute) for r in per_row]
        values = []
        for ref, pred in zip(references, predictions):
            left = set(tokenizer.encode(ref, add_special_tokens=False))
            right = set(tokenizer.encode(pred, add_special_tokens=False))
            values.append(2 * len(left & right) / (len(left) + len(right)) if left and right else 0.0)
        scores[method]["token_set_f1"] = values
    if not cli.lexical_only:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=cli.device).float().eval()
        with torch.inference_mode():
            vectors = model.encode(references, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
            for method, predictions in texts.items():
                candidate_vectors = model.encode(predictions, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
                scores[method]["bge_cosine"] = (vectors * candidate_vectors).sum(axis=1).tolist()
        del model, vectors, candidate_vectors
        gc.collect()
        if cli.device == "cuda":
            torch.cuda.empty_cache()
        from bert_score import BERTScorer
        model = BERTScorer(model_type="sentence-transformers/all-mpnet-base-v2", num_layers=12,
                           lang="en", device=cli.device, batch_size=1, rescale_with_baseline=False)
        with torch.inference_mode():
            for method, predictions in texts.items():
                values = model.score(predictions, references, batch_size=1)
                for suffix, tensor in zip(("p", "r", "f1"), values):
                    scores[method][f"bertscore_{suffix}"] = tensor.cpu().tolist()
    means = {method: {metric: fmean(values) for metric, values in metrics.items()}
             for method, metrics in scores.items()}
    report = {"n": len(rows), "sample_ids": [r["sample_id"] for r in rows],
              "generation_identity": result["identity"], "scoring_device": cli.device,
              "semantic_batch_size": 1, "rouge_stemming": True,
              "aggregation": "arithmetic mean over input samples", "means": means, "per_input": scores}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cli.output.with_name(cli.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(cli.output)
    for metric in means["direct"]:
        a, b = means["direct"][metric], means["full"][metric]
        print(f"{metric:<18} direct={a:.4f}  full={b:.4f}  delta={b-a:+.4f}")


if __name__ == "__main__":
    main()
