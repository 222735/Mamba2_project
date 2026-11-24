"""Inspect paired reconstruction scores for the released news passage sample."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import fmean
from random import Random

HERE = Path(__file__).resolve().parent
LABELS = ("ROUGE-1 P", "ROUGE-1 R", "ROUGE-1 F1", "ROUGE-2 F1",
          "ROUGE-L F1", "Token-set F1", "BGE cosine", "BERTScore F1")


def overlap_counts(reference_tokens, candidate_tokens):
    """Compute sufficient statistics on two *already normalized* ROUGE token lists."""
    def overlap(left, right):
        return sum((Counter(left) & Counter(right)).values())

    lcs = [0] * (len(candidate_tokens) + 1)
    for ref in reference_tokens:
        previous = 0
        for j, token in enumerate(candidate_tokens, start=1):
            old = lcs[j]
            lcs[j] = previous + 1 if ref == token else max(old, lcs[j - 1])
            previous = old
    return {
        "words": len(candidate_tokens),
        "unigrams": overlap(reference_tokens, candidate_tokens),
        "bigrams": overlap(zip(reference_tokens, reference_tokens[1:]),
                           zip(candidate_tokens, candidate_tokens[1:])),
        "lcs": lcs[-1],
    }


def per_input(row, method):
    def count(name):
        return int(row[f"{method}_{name}"])

    ref, gen = int(row["reference_words"]), count("words")
    unigrams, bigrams, lcs = count("unigrams"), count("bigrams"), count("lcs")
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return (
        ratio(unigrams, gen), ratio(unigrams, ref), ratio(2 * unigrams, ref + gen),
        ratio(2 * bigrams, max(ref - 1, 0) + max(gen - 1, 0)), ratio(2 * lcs, ref + gen),
        *(float(row[f"{method}_{metric}"]) for metric in ("token_set_f1", "bge_cosine", "bertscore_f1")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=0, metavar="B",
                        help="Draw B paired bootstrap resamples for a 95%% interval on the ROUGE-1 F1 gain.")
    parser.add_argument("--seed", type=int, default=123, help="Bootstrap random seed")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    with (HERE / "paired_evaluation.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 100 or len({r["source_index"] for r in rows}) != 100:
        raise ValueError("Expected 100 different input indices")
    direct = list(zip(*(per_input(row, "direct") for row in rows)))
    full = list(zip(*(per_input(row, "full") for row in rows)))
    print(f"{len(rows)} paired English news inputs (CoNLL-2003 / Reuters), 256 model tokens each\n")
    print(f"{'Metric':<15} {'Direct':>9} {'Full':>9} {'Delta':>9}")
    for label, left, right in zip(LABELS, direct, full):
        a, b = fmean(left), fmean(right)
        print(f"{label:<15} {a:9.4f} {b:9.4f} {b - a:+9.4f}")
    if args.bootstrap:
        gains = [b - a for a, b in zip(direct[2], full[2])]
        rng = Random(args.seed)
        draws = sorted(fmean(rng.choices(gains, k=len(gains))) for _ in range(args.bootstrap))
        def percentile(p):
            h = (len(draws) - 1) * p
            i = int(h)
            return draws[i] + (h - i) * (draws[min(i + 1, len(draws) - 1)] - draws[i])
        print(f"Paired bootstrap 95% interval for the mean ROUGE-1 F1 gain: "
              f"[{percentile(.025):.4f}, {percentile(.975):.4f}] (B={args.bootstrap}, seed={args.seed})")


if __name__ == "__main__":
    main()
