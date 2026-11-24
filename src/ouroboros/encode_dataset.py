import json
import random

import torch


def read_dataset(filename):
    texts = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    return texts


def tokenize_dataset(dataset, tokenizer):
    tokenized = []
    for item in dataset:
        tokens = tokenizer(item, return_tensors="pt")
        tokenized.append(tokens)
    return tokenized


def chunk_dataset(tokenized_dataset, block_size):
    ## drops the last few tokens if it doesn't fit in block size.
    block_size = block_size  # Make room for bos_token
    concatenated_input_ids = torch.cat(
        [tokens["input_ids"][0] for tokens in tokenized_dataset], dim=0
    )

    total_length = len(concatenated_input_ids)
    chunks = {
        "input_ids": [
            concatenated_input_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ],
    }
    # Add padding for last chunk if less than block_size
    block_size = block_size + 1
    if len(chunks["input_ids"][-1]) < block_size:
        chunks["input_ids"].pop()
    return chunks


def chunk_dataset_varied(tokenized_dataset, min_block_size=4, max_block_size=64):
    concatenated_input_ids = torch.cat(
        [tokens["input_ids"][0] for tokens in tokenized_dataset], dim=0
    )

    total_length = len(concatenated_input_ids)
    chunks = {"input_ids": []}

    i = 0
    while i < total_length:
        # Random block size between min_block_size and max_block_size
        block_size = random.randint(min_block_size, max_block_size)
        end_index = min(i + block_size, total_length)
        chunks["input_ids"].append(concatenated_input_ids[i:end_index].reshape(1,-1))
        i = end_index

    # Remove any empty last chunk (in case total_length == 0)
    if len(chunks["input_ids"][-1]) == 0:
        chunks["input_ids"].pop()

    return chunks


def batch_chunks(chunked_dataset, batch_size):
    total_chunks = len(chunked_dataset["input_ids"])
    batched_chunks = []
    for i in range(0, total_chunks, batch_size):
        batched_chunks.append(
            {
                "input_ids": torch.stack(
                    chunked_dataset["input_ids"][i : i + batch_size]
                ),
            }
        )
    return batched_chunks


def batch_chunks_varied(chunked_dataset, batch_size):
    total_chunks = len(chunked_dataset["input_ids"])
    batched_chunks = []

    for i in range(0, total_chunks, batch_size):
        batch = chunked_dataset["input_ids"][i : i + batch_size]
        batched_chunks.append(batch

        )
    return batched_chunks

def pad_batch_varied(batch):
    pad_token=0
    batch = [b.squeeze(0) for b in batch]
    max_length = max(len(chunk) for chunk in batch)
    padded_batch = [
        torch.cat([chunk, torch.full((max_length - len(chunk),), pad_token).to(chunk.device)])
        if len(chunk) < max_length else chunk
        for chunk in batch
    ]
    batched_chunks ={
            "input_ids": torch.stack(padded_batch),
        }

    return batched_chunks


def move_cache(cache_params, device):
    for key in cache_params.ssm_states:
        cache_params.ssm_states[key] = cache_params.ssm_states[key].to(device)
    for key in cache_params.conv_states:
        cache_params.conv_states[key] = cache_params.conv_states[key].to(device)
    return cache_params


def get_cache_params(batch, model):
    # STATE FIX 2026-08-11: naive chunked scan returns ZERO ssm_states when
    # len % 256 == 0 (ssd_chunk) - prefill len-1 then single-step update.
    L = batch.shape[1]
    if L % 256 == 0:
        outputs = model(batch[:, : L - 1], output_hidden_states=True, use_cache=True, return_dict=True)
        cache = outputs.cache_params
        outputs = model(batch[:, L - 1:], cache_params=cache,
                        cache_position=torch.tensor([L - 1], device=batch.device),
                        use_cache=True, return_dict=True)
        return outputs.cache_params
    outputs = model(batch, output_hidden_states=True, use_cache=True, return_dict=True)
    return outputs.cache_params

def encode_states(model, ids, device):
    # Correct encoder cache when seqlen is a multiple of ssd_chunk (256):
    # the naive chunked-scan prefill returns ZERO ssm_states there. Workaround:
    # prefill len-1 (safe) then apply the last token via the single-step
    # recurrent update. Verified bit-identical to the scan path at 255/128.
    L = ids.shape[1]
    with torch.no_grad():
        if L > 0 and L % 256 == 0:
            out = model(input_ids=ids[:, : L - 1], use_cache=True, return_dict=True)
            cache = out.cache_params
            out = model(input_ids=ids[:, L - 1:], cache_params=cache,
                        cache_position=torch.tensor([L - 1], device=device),
                        use_cache=True, return_dict=True)
            return out.cache_params
        return model(input_ids=ids, use_cache=True, return_dict=True).cache_params
