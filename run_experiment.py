"""Regenerate direct and optimized reconstructions from the released plaintext inputs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from inference import EngineArgs, TorchBackend, run_from_cache


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_model(path, device):
    import torch
    from transformers import AutoConfig
    from ouroboros.models.modeling_mamba2_decoder_v2 import Mamba2ForCausalLM
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config.n_groups = 1
    config.tie_word_embeddings = True
    model, info = Mamba2ForCausalLM.from_pretrained(
        path, config=config, torch_dtype=torch.float32, local_files_only=True,
        output_loading_info=True)
    if any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Model weights do not completely match the released architecture")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/samples.jsonl")
    parser.add_argument("--decoder", type=Path, default=HERE / "checkpoints/decoder")
    parser.add_argument("--backbone", default="AntonV/mamba2-130m-av")
    parser.add_argument("--output", type=Path, default=HERE / "results/run.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    if not cli.decoder.is_dir():
        parser.error("Download the decoder first with python download_decoder.py")
    data = [json.loads(line) for line in cli.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= cli.limit <= len(data) or len({r["sample_id"] for r in data}) != len(data):
        parser.error("The limit must fit the dataset and sample IDs must be unique")
    if any(len(r["input_ids"]) != 256 or not isinstance(r["reference"], str) for r in data):
        parser.error("Each sample must have 256 input token IDs and reference text")
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from ouroboros.encode_dataset import encode_states
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = (str(Path(cli.backbone).resolve()) if Path(cli.backbone).is_dir() else
                snapshot_download(cli.backbone, revision="9f0dd145df8556bbf2de843811ccc6b49d27024d",
                                  allow_patterns=["*.json", "*.safetensors"]))
    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (0, 0):
        raise ValueError("Unexpected BOS/EOS tokenizer configuration")
    if any(tokenizer.decode(r["input_ids"], skip_special_tokens=True) != r["reference"] for r in data):
        raise ValueError("Sample tokens differ from the supplied reference texts")
    identity = {"data_sha256": digest(cli.data), "decoder_sha256": digest(cli.decoder / "pytorch_model.bin"),
                "decoder_config_sha256": digest(cli.decoder / "config.json"),
                "backbone_sha256": digest(Path(backbone) / "model.safetensors"),
                "device": cli.device, "dtype": "float32", "torch": torch.__version__,
                "engine_sha256": digest(HERE / "src/inference.py"),
                "model_code_sha256": digest(HERE / "src/ouroboros/models/modeling_mamba2_decoder_v2.py")}
    result = {"identity": identity, "rows": [], "complete": False}
    if cli.output.exists():
        if not cli.resume:
            parser.error("Output exists. Use --resume or choose another --output.")
        result = json.loads(cli.output.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("The inputs, model, code or runtime changed since the saved run")
        ids = [r["sample_id"] for r in result["rows"]]
        if ids != [r["sample_id"] for r in data[:len(ids)]] or len(ids) > cli.limit:
            raise ValueError("Saved rows do not match this sample order/limit")
    print("Loading the encoder and trained decoder in FP32.", flush=True)
    encoder, decoder = load_model(backbone, cli.device), load_model(str(cli.decoder), cli.device)
    args = EngineArgs(device=cli.device)
    for sample in data[len(result["rows"]):cli.limit]:
        started = time.perf_counter()
        with torch.no_grad():
            cache = encode_states(encoder, torch.tensor([sample["input_ids"]], device=cli.device), cli.device)
            backend = TorchBackend(encoder, decoder, tokenizer, cache, args)
            direct_tokens, direct_logp = backend.rollout()
            direct = backend.text(direct_tokens)
            full = run_from_cache(encoder, decoder, tokenizer, cache, args)
        result["rows"].append({"sample_id": sample["sample_id"], "reference": sample["reference"],
                               "direct": {"text": direct, "tokens": direct_tokens, "logprob": direct_logp},
                               "full": full, "seconds": time.perf_counter() - started})
        result["complete"] = len(result["rows"]) == cli.limit
        result["requested_samples"] = cli.limit
        save(cli.output, result)
        print(f"Generated {len(result['rows'])}/{cli.limit}, sample {sample['sample_id']}, {time.perf_counter()-started:.1f}s", flush=True)
        del cache, backend, full
        gc.collect()
    print(f"Saved new predictions to {cli.output}")


if __name__ == "__main__":
    main()
