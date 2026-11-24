"""Inference policy and cache backend from the verified research implementation."""
from __future__ import annotations

import argparse

import copy

from dataclasses import asdict, dataclass

import hashlib

import json

import math

import os

from pathlib import Path

import sys

import time

@dataclass(frozen=True)
class EngineArgs:
    mode: str = 'full'
    device: str = 'cuda'
    max_len: int = 256
    target_layer: int = 19
    baseline_branches: int = 32
    local_first_branches: int = 16
    repair_parents: int = 4
    positions_per_parent: int = 4
    alternatives_per_position: int = 2
    risk_window: int = 64
    min_position_gap: int = 4
    gate_threshold: float = 0.027218597339261106
    eos_alternatives: int = 2
    eos_cap: int = 64

    def validate(self):
        if self.mode not in ('full', 'logp', 'b64'):
            raise ValueError('Unknown inference mode')
        frozen = EngineArgs(mode=self.mode, device=self.device)
        if asdict(self) != asdict(frozen):
            raise ValueError('This experiment uses frozen rules; configuration changed')
        if self.device not in ('cuda', 'cpu'):
            raise ValueError('Device must be cuda or cpu')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def unique_candidates(candidates):
    """Historical token deduplication retains the first proposal metadata."""
    seen, answer = set(), []
    for row in candidates:
        key = tuple(row['tokens'])
        if key not in seen:
            seen.add(key)
            answer.append(row)
    return answer

def ranking_value(row, mode):
    value = row['logprob'] if mode == 'logp' else row['layer_mse']
    require(isinstance(value, (float, int)) and math.isfinite(value),
            'Non-finite candidate selection score')
    return -value if mode == 'logp' else value

def choose_with_baseline(baseline, repairs, args):
    """Stable baseline fallback; logp has no hidden state-dependent gate."""
    if not repairs:
        return baseline
    best = min(repairs, key=lambda row: ranking_value(row, args.mode))
    if args.mode == 'logp':
        return best if best['logprob'] > baseline['logprob'] else baseline
    relative = ((baseline['layer_mse'] - best['layer_mse']) /
                max(abs(baseline['layer_mse']), 1e-30))
    return best if relative >= args.gate_threshold else baseline

def capped(unrestricted, parent, cap):
    if (unrestricted['stage'] == 'local_repair_eos' and
            unrestricted['extension_tokens'] > cap):
        return parent
    return unrestricted

def resolve_top32(top32, top16, scores):
    """Load the frozen, conservative same-set exact-tie compatibility helper."""
    try:
        from topk_prefix_compat_20261014 import resolve_prefix
    except ModuleNotFoundError as error:
        if error.name != 'topk_prefix_compat_20261014':
            raise
        from recovery_topk_20261014.topk_prefix_compat_20261014 import resolve_prefix
    return resolve_prefix(top32, top16, scores)

def run_policy(backend, args):
    """Reference-free orchestration; backend is mockable without torch/GPU."""
    args.validate()
    branch_count = 64 if args.mode == 'b64' else args.baseline_branches
    first_tokens, topk_audit = backend.first_tokens(branch_count)
    require(len(first_tokens) == branch_count and len(set(first_tokens)) == branch_count,
            'Wrong first-token proposal count')
    first = []
    for rank, token in enumerate(first_tokens, start=1):
        tokens, logprob = backend.rollout(first_token=token)
        require(tokens and tokens[0] == token, 'Forced first token not reproduced')
        first.append(dict(tokens=tokens, logprob=logprob, first_rank=rank,
            branch_position=-1, second_rank=1, stage='first_token', repair_round=0,
            repaired_positions=[], length=len(tokens), extension_tokens=0))
    # Historical full encodes ALL first32 separately (including any duplicates).
    for row in first:
        row['layer_mse'] = None if args.mode == 'logp' else backend.distance(row['tokens'])
        row['text'] = backend.text(row['tokens'])
    baseline = min(first, key=lambda row: ranking_value(row, args.mode))
    if args.mode == 'b64':
        return dict(tokens=list(baseline['tokens']), reconstruction=baseline['text'],
            trace=dict(baseline_pool=first, candidate_pool=first, chosen=baseline,
                pre_eos_parent=baseline, unrestricted=baseline, topk=topk_audit,
                eos_attempts=[], selection='layer19_ssm_mse_simple_top64'))

    candidates = list(first[:args.local_first_branches])
    known_distances = {tuple(row['tokens']): row['layer_mse'] for row in candidates}
    parents = sorted(candidates, key=lambda row: ranking_value(row, args.mode))[:args.repair_parents]
    proposals = []
    parent_records = []
    for parent in parents:
        repairs = backend.repair_positions(parent['tokens'])
        parent_records.append(dict(first_rank=parent['first_rank'],
            tokens=list(parent['tokens']), score=ranking_value(parent,args.mode),
            proposed_positions=[r[1] for r in repairs]))
        for margin, position, alternatives in repairs:
            for rank, alternative in enumerate(alternatives, start=2):
                tokens, logprob = backend.rollout(prefix=parent['tokens'],
                    replace_position=position, replacement=alternative)
                proposals.append(dict(tokens=tokens, logprob=logprob,
                    first_rank=parent['first_rank'], branch_position=position,
                    second_rank=rank, risk_margin=margin, stage='local_repair_r1',
                    repair_round=1, repaired_positions=[position], length=len(tokens),
                    extension_tokens=0))
    # Generate first, then encode in original order, exactly as frozen pipeline.
    for row in proposals:
        key = tuple(row['tokens'])
        if key not in known_distances:
            known_distances[key] = None if args.mode == 'logp' else backend.distance(row['tokens'])
        row['layer_mse'] = known_distances[key]
        row['text'] = backend.text(row['tokens'])
    candidates = unique_candidates(candidates + proposals)
    repair_rows = [row for row in candidates if row['stage'].startswith('local_repair')]
    parent = choose_with_baseline(baseline, repair_rows, args)
    backend.mark_pre_eos()
    eos_attempts = []
    if len(parent['tokens']) < args.max_len:
        alternatives = backend.eos_alternatives(parent['tokens'])
        for rank, alternative in enumerate(alternatives, start=2):
            tokens, logprob = backend.rollout(prefix=parent['tokens'],
                replace_position=len(parent['tokens']), replacement=alternative)
            text = backend.text(tokens)
            duplicate = any(row['text'] == text for row in candidates)
            eos_attempts.append(dict(rank=rank, token=alternative, duplicate=duplicate,
                                     output_length=len(tokens)))
            # Historical EOS uses text-level dedup, not token-level dedup.
            if duplicate:
                continue
            mse = None if args.mode == 'logp' else backend.distance(tokens)
            candidates.append(dict(tokens=tokens, text=text, logprob=logprob,
                layer_mse=mse, first_rank=parent['first_rank'],
                branch_position=len(parent['tokens']), second_rank=rank,
                stage='local_repair_eos', repair_round=1, repaired_positions=[],
                length=len(tokens), extension_tokens=len(tokens)-len(parent['tokens'])))
    repair_rows = [row for row in candidates if row['stage'].startswith('local_repair')]
    unrestricted = choose_with_baseline(baseline, repair_rows, args)
    final = capped(unrestricted, parent, args.eos_cap)
    return dict(tokens=list(final['tokens']), reconstruction=final['text'],
        trace=dict(baseline_pool=first, candidate_pool=candidates,
            repair_parent_selection=parent_records, pre_eos_parent=parent,
            unrestricted=unrestricted, chosen=final, topk=topk_audit,
            eos_attempts=eos_attempts,
            selection=('decoder_avg_logp_everywhere_strict_improvement_then_cap64'
                       if args.mode == 'logp' else 'frozen_mse_gate_then_cap64')))

class TorchBackend:
    def __init__(self, encoder, decoder, tokenizer, cache, args):
        import torch
        from ouroboros.encode_dataset import encode_states
        self.t, self.encode = torch, encode_states
        self.encoder, self.decoder, self.tok = encoder, decoder, tokenizer
        self.cache, self.args = cache, args
        require(cache.seqlen_offset == args.max_len, 'Source offset must be known 256')
        require(tokenizer.bos_token_id == tokenizer.eos_token_id == 0,
                'Frozen tokenizer BOS/EOS mismatch')
        self.costs = dict(decoder_forward_calls=0, decoder_token_steps=0,
            decoder_rollout_steps=0, decoder_replay_steps=0, decoder_probe_steps=0,
            generated_non_eos_tokens_including_forced=0, encoder_candidate_calls=0,
            encoder_candidate_tokens=0, encoder_model_forward_calls=0,
            encoder_actual_forward_tokens=0,
            cache_clones=0, cache_clone_cpu_dispatch_seconds=0.0,
            target_encoding_calls=0, model_training_steps=0)
        self.initial_versions = self.versions()
        self.pre_eos_seconds = None

    def versions(self):
        return (self.cache.seqlen_offset,
            tuple((name, layer, tensor.data_ptr(), tensor._version)
                for name in ('ssm_states','conv_states')
                for layer,tensor in sorted(getattr(self.cache,name).items())))

    def sync(self):
        if self.args.device == 'cuda':
            self.t.cuda.synchronize()

    def clone(self):
        start = time.perf_counter()
        cloned = copy.deepcopy(self.cache)
        for name in ('ssm_states','conv_states'):
            for layer,tensor in getattr(self.cache,name).items():
                other = getattr(cloned,name)[layer]
                require(tensor.data_ptr() != other.data_ptr(), 'Mutable cache alias')
        self.costs['cache_clones'] += 1
        self.costs['cache_clone_cpu_dispatch_seconds'] += time.perf_counter()-start
        return cloned

    def initial(self):
        cache = self.clone()
        current = self.t.tensor([[self.tok.bos_token_id]],device=self.args.device)
        position = self.t.tensor([cache.seqlen_offset],device=self.args.device)
        return cache, current, position

    def step(self,current,cache,position,phase):
        self.costs['decoder_forward_calls'] += 1
        self.costs['decoder_token_steps'] += int(current.numel())
        self.costs['decoder_'+phase+'_steps'] += int(current.numel())
        out = self.decoder(input_ids=current,cache_params=cache,
            cache_position=position,use_cache=True,return_dict=True)
        require(bool(self.t.isfinite(out.logits[0,-1]).all()),'Non-finite decoder logits')
        return out

    def first_tokens(self,k):
        cache,current,position = self.initial()
        out = self.step(current,cache,position,'probe')
        logits = out.logits[0,-1].clone()
        logits[self.tok.eos_token_id] = -float('inf')
        top32 = self.t.topk(logits,32).indices.tolist()
        if k == 64:
            top64 = self.t.topk(logits,64).indices.tolist()
            return top64,dict(policy='independent_top64_not_assumed_nested',
                independent_top32=top32, top64=top64,
                first32_equals_independent_top32=top64[:32]==top32,
                independent_top32_subset=set(top32).issubset(top64))
        top16 = self.t.topk(logits,16).indices.tolist()
        ids = list(dict.fromkeys(top32+top16))
        resolved,audit = resolve_top32(top32,top16,dict(zip(ids,logits[ids].tolist())))
        return resolved,dict(policy='frozen_top32_top16_exact_tie_compatibility',
                              independent_top32=top32,anchor_top16=top16,
                              resolved_top32=resolved,compatibility=audit)

    def rollout(self,first_token=None,prefix=None,replace_position=None,replacement=None):
        t,a = self.t,self.args
        cache,current,position = self.initial()
        tokens,total = [],0.0
        for step in range(a.max_len):
            phase = 'replay' if prefix is not None and step < replace_position else 'rollout'
            out = self.step(current,cache,position,phase)
            logits = out.logits[0,-1]
            probabilities = t.softmax(logits,dim=-1)
            if prefix is not None and step < replace_position:
                token = int(prefix[step])
            elif replace_position is not None and step == replace_position:
                token = int(replacement)
            elif step == 0 and first_token is not None:
                token = int(first_token)
            else:
                token = int(logits.argmax())
            total += float(t.log(probabilities[token].clamp_min(1e-12)))
            cache,position = out.cache_params,position+1
            if token == self.tok.eos_token_id:
                break
            tokens.append(token)
            self.costs['generated_non_eos_tokens_including_forced'] += 1
            current = t.tensor([[token]],device=a.device)
        return tokens,total/max(len(tokens),1)

    def repair_positions(self,tokens):
        t,a = self.t,self.args
        cache,current,position = self.initial()
        risks=[]
        for step in range(min(len(tokens),a.risk_window+1)):
            out = self.step(current,cache,position,'replay')
            logits = out.logits[0,-1]
            actual = int(tokens[step])
            if step >= 1:
                masked = logits.clone()
                masked[self.tok.eos_token_id] = -float('inf')
                masked[actual] = -float('inf')
                alternatives = t.topk(masked,a.alternatives_per_position).indices.tolist()
                probabilities = t.softmax(logits,dim=-1)
                margin = float(probabilities[actual]-probabilities[int(alternatives[0])])
                require(math.isfinite(margin),'Non-finite repair margin')
                risks.append((margin,step,alternatives))
            cache,position = out.cache_params,position+1
            current = t.tensor([[actual]],device=a.device)
        chosen=[]
        for margin,step,alternatives in sorted(risks):
            if step >= 1 and all(abs(step-prior[1]) >= a.min_position_gap for prior in chosen):
                chosen.append((margin,step,alternatives))
                if len(chosen) == a.positions_per_parent:
                    break
        return chosen

    def eos_alternatives(self,tokens):
        t,a = self.t,self.args
        cache,current,position = self.initial()
        for step in range(len(tokens)+1):
            out = self.step(current,cache,position,'replay')
            logits = out.logits[0,-1]
            cache,position = out.cache_params,position+1
            if step == len(tokens):
                require(int(logits.argmax()) == self.tok.eos_token_id,
                        'Parent EOS not exactly reproduced')
                masked = logits.clone()
                masked[self.tok.eos_token_id] = -float('inf')
                return masked.topk(a.eos_alternatives).indices.tolist()
            current = t.tensor([[tokens[step]]],device=a.device)
        raise RuntimeError('Unreachable EOS replay state')

    def distance(self,tokens):
        require(self.args.mode != 'logp','B1 may not consult candidate state MSE')
        t,a = self.t,self.args
        require(len(tokens)<=a.max_len,'Candidate exceeds generation length')
        padded = tokens+[self.tok.eos_token_id]*(a.max_len-len(tokens))
        self.costs['encoder_candidate_calls'] += 1
        self.costs['encoder_candidate_tokens'] += len(padded)
        # Count actual model calls, not inferred counts from candidate budgets.
        def count_forward(module, positional, keyword):
            ids=keyword.get('input_ids')
            if ids is None and positional:
                ids=positional[0]
            require(ids is not None,'Encoder forward without countable candidate input')
            self.costs['encoder_model_forward_calls'] += 1
            self.costs['encoder_actual_forward_tokens'] += int(ids.numel())
        hook=self.encoder.register_forward_pre_hook(count_forward,with_kwargs=True)
        try:
            candidate = self.encode(self.encoder,
                t.tensor([padded],device=a.device,dtype=t.long),a.device)
        finally:
            hook.remove()
        changed = candidate.ssm_states[a.target_layer].float()
        stolen = self.cache.ssm_states[a.target_layer].float()
        result = float((changed-stolen).pow(2).mean())
        require(math.isfinite(result) and result>=0,'Invalid candidate state distance')
        return result

    def text(self,tokens):
        return self.tok.decode(tokens,skip_special_tokens=True).split('<|endoftext|>')[0]

    def mark_pre_eos(self):
        self.sync()
        self.pre_eos_seconds=time.perf_counter()-self.start

def run_from_cache(encoder,decoder,tokenizer,cache,args):
    """Return JSONable tokens/reconstruction/trace/costs; never inspect labels."""
    if not isinstance(args,EngineArgs):
        args = EngineArgs(**{key:getattr(args,key) for key in asdict(EngineArgs())})
    args.validate()
    backend=TorchBackend(encoder,decoder,tokenizer,cache,args)
    t=backend.t
    backend.sync()
    if args.device == 'cuda':
        t.cuda.reset_peak_memory_stats()
    backend.start=time.perf_counter()
    with t.no_grad():
        answer=run_policy(backend,args)
    backend.sync()
    require(backend.versions()==backend.initial_versions,'Immutable source cache mutated')
    backend.costs.update(generation_seconds=time.perf_counter()-backend.start,
        pre_eos_seconds=backend.pre_eos_seconds,
        peak_allocated_MiB=t.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None,
        peak_reserved_MiB=t.cuda.max_memory_reserved()/2**20 if args.device=='cuda' else None,
        scope='fresh_cache_only_generation_including_all_decoder_replays_and_candidate_encodes',
        excluded='model_load; cache_package_creation_and_loading; final_quality_scoring',
        proposal_count=len(answer['trace']['baseline_pool'])+
            sum(r['stage']!='first_token' for r in answer['trace']['candidate_pool']),
        source_cache_immutable=True)
    answer['costs']=backend.costs
    return answer
