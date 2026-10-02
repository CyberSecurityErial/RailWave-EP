"""Lightweight CPU policy selection for RailWave network fast paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence


class NetworkPolicy(str, Enum):
    OFF = 'off'
    RAIL = 'rail'
    FANIN = 'fanin'


@dataclass(frozen=True)
class TrafficFeatures:
    total_tokens: int
    rail_max_over_mean: float
    max_node_fan_in: int
    pair_max_over_mean: float
    num_qps: int
    num_nodes: int


@dataclass(frozen=True)
class PolicyProfilePoint:
    """A measured profile and its traffic features."""

    features: TrafficFeatures
    latency_us: Mapping[NetworkPolicy, float]

    def best_policy(self, min_gain: float = 0.0) -> NetworkPolicy:
        off_us = float(self.latency_us[NetworkPolicy.OFF])
        policy, best_us = min(
            self.latency_us.items(), key=lambda item: float(item[1]))
        if off_us / float(best_us) < 1.0 + min_gain:
            return NetworkPolicy.OFF
        return NetworkPolicy(policy)


@dataclass(frozen=True)
class PolicyDecision:
    """Selected path and reference measurement."""

    policy: NetworkPolicy
    reference_features: TrafficFeatures
    reference_distance: float
    predicted_latency_us: float
    predicted_gain_over_off: float
    reason: str


def extract_traffic_features(
        demand: Sequence[Sequence[float]],
        source_rail_counts: Sequence[Sequence[float]], *,
        num_qps: int) -> TrafficFeatures:
    """Reduce an asynchronous count snapshot to scheduler features."""
    num_nodes = len(demand)
    if num_nodes < 2 or any(len(row) != num_nodes for row in demand):
        raise ValueError('demand must be a square matrix with at least 2 nodes')
    if len(source_rail_counts) != num_nodes or not source_rail_counts:
        raise ValueError('source_rail_counts must have one row per node')
    num_rails = len(source_rail_counts[0])
    if num_rails < 1 or any(len(row) != num_rails
                            for row in source_rail_counts):
        raise ValueError('source_rail_counts must be a non-empty rectangle')
    if num_qps < 1:
        raise ValueError('num_qps must be positive')

    remote = []
    fan_in = 0
    for destination in range(num_nodes):
        sources = 0
        for source in range(num_nodes):
            value = float(demand[source][destination])
            if not math.isfinite(value) or value < 0:
                raise ValueError('demand must contain finite non-negative values')
            if source != destination and value > 0:
                remote.append(value)
                sources += 1
        fan_in = max(fan_in, sources)

    rail_skew = 1.0
    for row in source_rail_counts:
        values = [float(value) for value in row]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(
                'source_rail_counts must contain finite non-negative values')
        total = sum(values)
        if total > 0:
            rail_skew = max(rail_skew, max(values) / (total / num_rails))

    total_tokens = int(round(sum(remote)))
    pair_skew = (
        max(remote) / (sum(remote) / len(remote)) if remote else 1.0)
    return TrafficFeatures(
        total_tokens=total_tokens,
        rail_max_over_mean=rail_skew,
        max_node_fan_in=fan_in,
        pair_max_over_mean=pair_skew,
        num_qps=num_qps,
        num_nodes=num_nodes,
    )


class NetworkPolicySelector:
    """Select the closest measured profile with gain and switching margins."""

    def __init__(self, profile: Sequence[PolicyProfilePoint], *,
                 min_gain: float = 0.02, switch_margin: float = 0.02):
        if not profile:
            raise ValueError('profile must contain at least one point')
        if min_gain < 0 or switch_margin < 0:
            raise ValueError('gain thresholds must be non-negative')
        self.profile = tuple(profile)
        self.min_gain = float(min_gain)
        self.switch_margin = float(switch_margin)

    @staticmethod
    def _distance(lhs: TrafficFeatures, rhs: TrafficFeatures) -> float:
        def log_distance(a: int, b: int) -> float:
            return abs(math.log2(max(a, 1) / max(b, 1)))

        return (
            log_distance(lhs.total_tokens, rhs.total_tokens) +
            abs(lhs.rail_max_over_mean - rhs.rail_max_over_mean) +
            abs(lhs.max_node_fan_in - rhs.max_node_fan_in) +
            abs(lhs.pair_max_over_mean - rhs.pair_max_over_mean) +
            0.5 * log_distance(lhs.num_qps, rhs.num_qps) +
            abs(lhs.num_nodes - rhs.num_nodes)
        )

    def decide(self, features: TrafficFeatures, *,
               current: NetworkPolicy | str | None = None) -> PolicyDecision:
        point = min(
            self.profile,
            key=lambda item: self._distance(features, item.features))
        candidate = point.best_policy(self.min_gain)
        policy = candidate
        held_by_hysteresis = False
        if current is None:
            current_policy = None
        else:
            current_policy = NetworkPolicy(current)
            if current_policy in point.latency_us:
                candidate_us = float(point.latency_us[candidate])
                current_us = float(point.latency_us[current_policy])
                if current_us <= candidate_us * (1.0 + self.switch_margin):
                    policy = current_policy
                    held_by_hysteresis = current_policy != candidate

        off_us = float(point.latency_us[NetworkPolicy.OFF])
        policy_us = float(point.latency_us[policy])
        gain = off_us / policy_us - 1.0
        if held_by_hysteresis:
            reason = (
                'hysteresis: current path is within the switch margin of '
                'the calibrated winner')
        elif policy == NetworkPolicy.RAIL:
            reason = (
                'Rail projection bottleneck: Rail-only clears the calibrated '
                'minimum-gain gate')
        elif policy == NetworkPolicy.FANIN:
            reason = (
                'Node projection bottleneck: FanIn-only clears the calibrated '
                'minimum-gain gate')
        else:
            reason = (
                'benefit gate: no specialized path has enough calibrated '
                'gain over native')
        return PolicyDecision(
            policy=policy,
            reference_features=point.features,
            reference_distance=self._distance(features, point.features),
            predicted_latency_us=policy_us,
            predicted_gain_over_off=gain,
            reason=reason,
        )

    def select(self, features: TrafficFeatures, *,
               current: NetworkPolicy | str | None = None) -> NetworkPolicy:
        return self.decide(features, current=current).policy
