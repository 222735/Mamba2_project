"""Run the public backbone on supplied text or inspect the archived Figure 4 case."""
import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
BACKBONE = "AntonV/mamba2-130m-av"
REVISION = "9f0dd145df8556bbf2de843811ccc6b49d27024d"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def forward(text, local_dir, case=None):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    location = local_dir or BACKBONE
    options = {"local_files_only": True} if local_dir else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(location, **options)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if case is not None:
        # Decoding the original token stream to text loses some token boundaries.
        for start, end, tokens in reversed(case["input_token_edits"]):
            ids[start:end] = tokens
        if digest(",".join(map(str, ids))) != case["input_ids_sha256"]:
            raise ValueError("The original case token stream could not be verified.")
    if not 1 <= len(ids) <= 256:
        raise ValueError(f"Expected 1 to 256 model tokens. Received {len(ids)}.")
    # This published checkpoint has one SSM group (1792 = 1536 + 2 * 128).
    config = AutoConfig.from_pretrained(location, n_groups=1, **options)
    weights_file = (local_dir / "model.safetensors" if local_dir else
                    hf_hub_download(BACKBONE, "model.safetensors", revision=REVISION))
    weights = load_file(str(weights_file))
    # The mirror stores its shared input/output embedding once, under lm_head.
    weights["backbone.embeddings.weight"] = weights["lm_head.weight"]
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval()
    with torch.inference_mode():
        output = model(torch.tensor([ids]), use_cache=True)
    if output.cache_params is None:
        raise RuntimeError("The backbone did not return a cache.")
    print(f"Live forward on CPU: {len(ids)} tokens, {model.config.num_hidden_layers} layers, cache produced.")


def replay(case, reference):
    scorer = None
    if reference is not None:
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    print(f"Archived Figure 4 trace, news passage {case['source_index']} ({case['input_tokens']} original model tokens).")
    for stage in case["stages"]:
        r1, rl = stage["rouge1_f1"], stage["rougeL_f1"]
        if scorer is not None:
            # Copy spans from the known reference and insert the archived literal edits.
            text = "".join(part if isinstance(part, str) else reference[part[0]:part[1]]
                           for part in stage["output_parts"])
            if digest(text) != stage["output_sha256"]:
                raise ValueError(f"Archived output hash mismatch: {stage['label']}")
            scores = scorer.score(reference, text)
            fresh = (scores["rouge1"].fmeasure, scores["rougeL"].fmeasure)
            if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8) for a, b in zip(fresh, (r1, rl))):
                raise ValueError(f"Archived and recomputed scores differ: {stage['label']}")
            r1, rl = fresh
            print(f"\n{stage['label']} (archived output):\n{text}")
        origin = "recomputed" if scorer is not None else "archived"
        print(f"{stage['label']}: {stage['output_tokens']} tokens, {origin} ROUGE-1 F1 {r1:.4f}, ROUGE-L F1 {rl:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="UTF-8 input for the live backbone pass")
    source.add_argument("--text-file", type=Path, help="Read the input from a UTF-8 file")
    parser.add_argument("--model-dir", type=Path, help="Local copy of the published backbone for offline use")
    parser.add_argument("--replay-only", action="store_true", help="Inspect the fixed historical case without running the backbone")
    args = parser.parse_args()
    case = json.loads((HERE / "case68.json").read_text(encoding="utf-8"))
    text = args.text_file.read_text(encoding="utf-8-sig") if args.text_file else args.text
    if text is None:
        if not args.replay_only:
            parser.error("Supply --text, --text-file, or --replay-only.")
        replay(case, None)
        return
    if not text.strip():
        parser.error("Please supply a nonempty passage.")
    # Editors often insert a final newline when saving a passage to disk.
    reference = next((t for t in (text, text.rstrip("\r\n")) if digest(t) == case["reference_sha256"]), None)
    if args.replay_only and reference is None:
        parser.error("The supplied text differs from the archived passage.")
    if not args.replay_only:
        if reference is None:
            forward(text, args.model_dir)
        else:
            forward(reference, args.model_dir, case)
    if reference is None:
        print("This input has no archived reconstruction. The live pass ends at the backbone.")
    else:
        replay(case, reference)


if __name__ == "__main__":
    main()
