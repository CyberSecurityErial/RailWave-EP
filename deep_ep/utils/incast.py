"""Control-plane planning for RailWave cross-node incast avoidance."""

from __future__ import annotations

import math

import torch


def _as_signed_int32(value: int) -> int:
    """Encode a 32-bit Rail bitset in a signed torch.int32 element."""
    value &= 0xFFFFFFFF
    return value if value < (1 << 31) else value - (1 << 32)


def _rotated_order(length: int, offset: int) -> list[int]:
    if length == 0:
        return []
    offset %= length
    return [(offset + index) % length for index in range(length)]


def plan_pairwise_incast_waves(
        num_nodes: int, *, max_peers_per_wave: int = 1,
        epoch: int = 0) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Factorize node All-to-All into bounded-fan-in communication waves.

    Each pair ``(a, b)`` is bidirectional: both ``a -> b`` and ``b -> a`` may
    run in that wave.  With ``max_peers_per_wave=1``, every real node talks to
    at most one remote node per wave, so node-level fan-in is one.  Every
    unordered real-node pair appears exactly once across the returned plan.

    The construction is the round-robin (circle-method) one-factorization of
    the complete graph.  Even ``N`` takes ``N - 1`` waves.  Odd ``N`` adds one
    dummy node and takes ``N`` waves, with one real node idle in each wave.
    Combining adjacent one-factors provides a tunable launch-overhead versus
    fan-in tradeoff while retaining exact pair coverage.
    """
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
        raise TypeError('num_nodes must be an integer')
    if num_nodes < 2:
        raise ValueError('num_nodes must be at least 2')
    if (not isinstance(max_peers_per_wave, int) or
            isinstance(max_peers_per_wave, bool)):
        raise TypeError('max_peers_per_wave must be an integer')
    if max_peers_per_wave < 1:
        raise ValueError('max_peers_per_wave must be at least 1')
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError('epoch must be an integer')

    effective_nodes = num_nodes if num_nodes % 2 == 0 else num_nodes + 1
    ring = list(range(effective_nodes))
    one_factors: list[tuple[tuple[int, int], ...]] = []
    for _ in range(effective_nodes - 1):
        pairs = []
        for index in range(effective_nodes // 2):
            lhs = ring[index]
            rhs = ring[effective_nodes - 1 - index]
            if lhs < num_nodes and rhs < num_nodes:
                pairs.append((min(lhs, rhs), max(lhs, rhs)))
        one_factors.append(tuple(sorted(pairs)))
        # Hold node zero fixed and rotate every other participant clockwise.
        ring = [ring[0], ring[-1], *ring[1:-1]]

    # This orientation gives the intuitive four-node order
    # (01,23), (02,13), (03,12).
    one_factors.reverse()
    round_offset = epoch % len(one_factors)
    one_factors = (
        one_factors[round_offset:] + one_factors[:round_offset])

    budget = min(max_peers_per_wave, num_nodes - 1)
    waves = []
    for begin in range(0, len(one_factors), budget):
        waves.append(tuple(
            pair
            for factor in one_factors[begin:begin + budget]
            for pair in factor
        ))
    return tuple(waves)


def _allocate_counts(
        active_sources: list[int], demand: list[list[float]], destination: int,
        total_slots: int, num_rails: int,
        source_order: list[int]) -> dict[int, int]:
    """Allocate Rail memberships while minimizing the hottest flow/Rail."""
    counts = {source: 1 for source in active_sources}
    priority = {source: index for index, source in enumerate(source_order)}
    for _ in range(total_slots - len(active_sources)):
        # Splitting the currently hottest per-Rail flow is the greedy optimum for
        # the next Rail. Stable rotated ties avoid pinning the same node forever.
        source = max(
            (item for item in active_sources if counts[item] < num_rails),
            key=lambda item: (
                demand[item][destination] / counts[item],
                demand[item][destination],
                -priority[item],
            ),
        )
        counts[source] += 1
    return counts


def plan_incast_rail_masks(
        demand: torch.Tensor, num_rails: int, *, epoch: int = 0,
        rail_overlap: float = 1.0) -> torch.Tensor:
    """Plan per-node-pair Rail subsets from a predicted traffic matrix.

    Args:
        demand: Square ``[num_nodes, num_nodes]`` tensor. ``demand[s, d]`` is
            the predicted cross-node byte or token volume from source node
            ``s`` to destination node ``d`` in the next control window.
        num_rails: Number of corresponding egress/ingress Rails per node.
        epoch: Optional deterministic rotation. Advancing it changes equally
            good Rail identities without changing the volume allocation.
        rail_overlap: Rail-membership budget relative to a disjoint partition.
            ``1`` gives strict disjoint subsets. ``1.5`` keeps more source-side
            Rail parallelism while limiting equal three-source traffic to at
            most two sources on a destination Rail.

    Returns:
        A CPU int32 tensor of shape ``[num_nodes, num_nodes]``. Bit ``r`` in
        element ``[s, d]`` allows flow ``s -> d`` to use Rail ``r``. Diagonal
        and zero-demand entries are zero.

    With the default ``rail_overlap=1``, flows receive disjoint subsets and
    collectively use every Rail when there are enough Rails. Larger budgets
    allow bounded subset overlap to preserve source-side parallelism. Subset
    sizes minimize the largest predicted flow volume per Rail. When sources
    outnumber Rails, every flow still receives one Rail and a largest-first
    list scheduler minimizes unavoidable destination contention.
    """
    if not isinstance(demand, torch.Tensor):
        raise TypeError('demand must be a torch.Tensor')
    if demand.ndim != 2 or demand.shape[0] != demand.shape[1]:
        raise ValueError('demand must have square shape [num_nodes, num_nodes]')
    if not isinstance(num_rails, int) or isinstance(num_rails, bool):
        raise TypeError('num_rails must be an integer')
    if not 1 <= num_rails <= 32:
        raise ValueError('num_rails must be in [1, 32]')
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError('epoch must be an integer')
    if (not isinstance(rail_overlap, (int, float)) or
            isinstance(rail_overlap, bool)):
        raise TypeError('rail_overlap must be a number')
    rail_overlap = float(rail_overlap)
    if not math.isfinite(rail_overlap) or rail_overlap < 1.0:
        raise ValueError('rail_overlap must be finite and at least 1')

    demand_cpu = demand.detach().to(device='cpu', dtype=torch.float64)
    if not bool(torch.isfinite(demand_cpu).all().item()):
        raise ValueError('demand must contain only finite values')
    if bool((demand_cpu < 0).any().item()):
        raise ValueError('demand must be non-negative')

    num_nodes = demand_cpu.shape[0]
    demand_values = demand_cpu.tolist()
    masks = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Predicted source-NIC load is carried across destinations so masks also
    # avoid creating a new egress Rail hotspot while fixing destination incast.
    source_rail_load = [
        [0.0 for _ in range(num_rails)] for _ in range(num_nodes)
    ]

    destination_order = _rotated_order(num_nodes, epoch)
    for destination in destination_order:
        active_sources = [
            source for source in range(num_nodes)
            if source != destination and demand_values[source][destination] > 0
        ]
        if not active_sources:
            continue

        source_rotation = (destination + epoch) % num_nodes
        rotated_sources = _rotated_order(num_nodes, source_rotation)
        source_priority = {source: index for index, source in enumerate(rotated_sources)}
        source_order = sorted(
            active_sources,
            key=lambda source: (
                -demand_values[source][destination],
                source_priority[source],
            ),
        )
        rail_rotation = (destination * 3 + epoch) % num_rails
        rail_priority_order = _rotated_order(num_rails, rail_rotation)
        rail_priority = {rail: index for index, rail in enumerate(rail_priority_order)}

        if len(active_sources) <= num_rails:
            total_slots = min(
                len(active_sources) * num_rails,
                max(len(active_sources), math.ceil(num_rails * rail_overlap)),
            )
            counts = _allocate_counts(
                active_sources, demand_values, destination, total_slots,
                num_rails, rotated_sources)
            strict_partition = total_slots == num_rails
            available_rails = set(range(num_rails))
            destination_rail_load = [0.0 for _ in range(num_rails)]
            destination_memberships = [0 for _ in range(num_rails)]

            # Place the largest per-Rail flows first. For each source, select
            # Rails with the least destination and source-side pressure. The
            # strict mode consumes each Rail once; overlap mode may reuse it.
            placement_order = sorted(
                active_sources,
                key=lambda source: (
                    -(demand_values[source][destination] / counts[source]),
                    source_priority[source],
                ),
            )
            for source in placement_order:
                count = counts[source]
                candidates = (
                    available_rails if strict_partition else range(num_rails))
                selected = sorted(
                    candidates,
                    key=lambda rail: (
                        destination_rail_load[rail],
                        destination_memberships[rail],
                        source_rail_load[source][rail],
                        rail_priority[rail],
                    ),
                )[:count]
                per_rail = demand_values[source][destination] / count
                mask = 0
                for rail in selected:
                    mask |= 1 << rail
                    source_rail_load[source][rail] += per_rail
                    destination_rail_load[rail] += per_rail
                    destination_memberships[rail] += 1
                    if strict_partition:
                        available_rails.remove(rail)
                masks[source][destination] = mask
            if strict_partition:
                assert not available_rails
        else:
            # Disjointness is impossible. LPT placement minimizes the
            # destination's maximum Rail load; source load breaks equal ties.
            destination_rail_load = [0.0 for _ in range(num_rails)]
            for source in source_order:
                volume = demand_values[source][destination]
                rail = min(
                    range(num_rails),
                    key=lambda item: (
                        destination_rail_load[item],
                        source_rail_load[source][item],
                        rail_priority[item],
                    ),
                )
                masks[source][destination] = 1 << rail
                destination_rail_load[rail] += volume
                source_rail_load[source][rail] += volume

    encoded = [
        [_as_signed_int32(mask) for mask in row]
        for row in masks
    ]
    return torch.tensor(encoded, dtype=torch.int32, device='cpu')
