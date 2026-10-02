"""Nearest-anchor selection with a Joint fallback."""

import math
from collections.abc import Mapping, Sequence
from typing import Any


def decide(
    policy: Mapping[str, Any],
    features: Sequence[float],
) -> tuple[str, str | None, str]:
    """Select a measured path within the anchor's payload and shape bounds.

    Features are log2 payload, normalized rail skew, and normalized receive
    concentration. Eligible anchors are ordered by squared distance, then ID.
    """
    eligible = []
    size_bound = math.log2(policy["max_size_ratio"]) + 1e-12
    for anchor in policy["anchors"]:
        reference = anchor["features"]
        if abs(features[0] - reference[0]) > size_bound:
            continue
        if max(abs(features[i] - reference[i]) for i in (1, 2)) > policy["shape_tolerance"]:
            continue
        distance = sum((value - expected) ** 2 for value, expected in zip(features, reference))
        eligible.append((distance, anchor["case_id"], anchor))
    if not eligible:
        return "joint", None, "no stable in-range alternative; identical Joint fallback"
    anchor = min(eligible, key=lambda item: (item[0], item[1]))[2]
    return anchor["selected"], anchor["case_id"], "stable observed benefit near development anchor"
