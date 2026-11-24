"""Conservative compatibility for independent Top-32 and Top-16 calls.

Only a permutation of the *same* first-16 tokens at exactly tied scores is
repairable. The complete 32-token set and the original tail are unchanged.
This helper does not define a new boundary-tie candidate-selection policy.
"""

import math
from numbers import Real


def resolve_prefix(top32, top16, scores):
    """Return ``(compatible_tokens, audit_or_none)`` without mutating inputs.

    ``scores`` maps token IDs to the scalar logits used for both top-k calls.
    The caller remains responsible for obtaining both calls from the same
    logits tensor. Global vocabulary ranking cannot be checked from this map.
    Every unsupported discrepancy raises ``ValueError``; no tolerance-based
    or boundary-set substitution is performed.
    """
    if not isinstance(top32, list) or not isinstance(top16, list):
        raise ValueError("top32 and top16 must be Python lists")
    if len(top32) != 32 or len(top16) != 16:
        raise ValueError("Expected exactly 32 and 16 token IDs")
    for name, tokens in (("top32", top32), ("top16", top16)):
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(name + " contains an invalid token ID")
        if len(set(tokens)) != len(tokens):
            raise ValueError(name + " contains duplicate token IDs")

    values = {}
    for token in set(top32) | set(top16):
        if token not in scores:
            raise ValueError("Missing score for token " + str(token))
        value = scores[token]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Score must be a real scalar for token " + str(token))
        if not math.isfinite(value):
            raise ValueError("Non-finite score for token " + str(token))
        values[token] = value

    scores32 = [values[token] for token in top32]
    scores16 = [values[token] for token in top16]
    for name, ranked_scores in (("top32", scores32), ("top16", scores16)):
        if any(left < right for left, right in zip(ranked_scores, ranked_scores[1:])):
            raise ValueError(name + " scores are not non-increasing")

    if top32[:16] == top16:
        return top32, None

    # Reject both non-nested sets and equal-score substitutions across rank 16.
    # Such substitutions could change the historical local-repair parent pool.
    if set(top32[:16]) != set(top16):
        raise ValueError("Top-16 prefix membership differs; boundary-tie policy is not authorized")
    if scores32[:16] != scores16:
        raise ValueError("Prefix discrepancy is not an exact-score tie permutation")

    resolved = list(top16) + top32[16:]
    if len(set(resolved)) != 32 or set(resolved) != set(top32):
        raise ValueError("Internal error: complete Top-32 membership changed")
    if [values[token] for token in resolved] != scores32:
        raise ValueError("Internal error: per-rank scores changed")

    changes = [
        {
            "rank": rank + 1,
            "original_token": before,
            "resolved_token": after,
            "score": float(values[before]),
        }
        for rank, (before, after) in enumerate(zip(top32, resolved))
        if before != after
    ]
    audit = {
        "policy": "same_prefix_set_exact_tie_permutation_v1",
        "original_top32": list(top32),
        "anchor_top16": list(top16),
        "resolved_top32": list(resolved),
        "changed_ranks": changes,
        "complete_top32_membership_unchanged": True,
        "tail16_order_unchanged": True,
        "rankwise_scores_exactly_equal": True,
    }
    return resolved, audit
