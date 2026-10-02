"""Directed permutation schedules for node-to-node traffic.

A transfer is (source, destination, offset, count, edge_total). Cyclic schedules
use fixed topology matchings. The weighted and equal-chunk planners consume
the complete demand matrix and serve as experimental alternatives.
"""

from __future__ import annotations

from collections.abc import Sequence


Transfer = tuple[int, int, int, int, int]
Wave = tuple[Transfer, ...]


def _validated_matrix(demand: Sequence[Sequence[int]]) -> list[list[int]]:
    size = len(demand)
    if size < 2 or any(len(row) != size for row in demand):
        raise ValueError("demand must be a square matrix with at least 2 nodes")
    result = []
    for row in demand:
        values = []
        for value in row:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("demand entries must be integers")
            if value < 0:
                raise ValueError("demand entries must be non-negative")
            values.append(value)
        result.append(values)
    return result


def _complete_balanced(matrix: list[list[int]]) -> tuple[list[list[int]], int]:
    """Pad a non-negative matrix to one common row/column sum."""
    size = len(matrix)
    result = [row[:] for row in matrix]
    row_sums = [sum(row) for row in result]
    col_sums = [sum(result[row][col] for row in range(size)) for col in range(size)]
    target = max([0, *row_sums, *col_sums])
    row_deficit = [target - value for value in row_sums]
    col_deficit = [target - value for value in col_sums]
    if sum(row_deficit) != sum(col_deficit):
        raise AssertionError("row and column padding deficits must match")

    row = 0
    col = 0
    added = 0
    while row < size and col < size:
        while row < size and row_deficit[row] == 0:
            row += 1
        while col < size and col_deficit[col] == 0:
            col += 1
        if row == size or col == size:
            break
        amount = min(row_deficit[row], col_deficit[col])
        result[row][col] += amount
        row_deficit[row] -= amount
        col_deficit[col] -= amount
        added += amount

    if any(row_deficit) or any(col_deficit):
        raise AssertionError("failed to complete balanced matrix")
    assert all(sum(row) == target for row in result)
    assert all(sum(result[row][col] for row in range(size)) == target for col in range(size))
    return result, added


def _positive_perfect_matching(matrix: list[list[int]]) -> tuple[int, ...]:
    """Find one deterministic perfect matching over positive matrix entries."""
    size = len(matrix)
    destination_source = [-1] * size

    def augment(source: int, visited: list[bool]) -> bool:
        destinations = sorted(
            (destination for destination, value in enumerate(matrix[source]) if value > 0),
            key=lambda destination: (-matrix[source][destination], destination),
        )
        for destination in destinations:
            if visited[destination]:
                continue
            visited[destination] = True
            previous = destination_source[destination]
            if previous == -1 or augment(previous, visited):
                destination_source[destination] = source
                return True
        return False

    source_order = sorted(
        range(size),
        key=lambda source: (sum(value > 0 for value in matrix[source]), source),
    )
    for source in source_order:
        if not augment(source, [False] * size):
            raise ValueError("positive support has no perfect matching")

    source_destination = [-1] * size
    for destination, source in enumerate(destination_source):
        if source < 0:
            raise AssertionError("incomplete matching")
        source_destination[source] = destination
    return tuple(source_destination)


def plan_fast_style_waves(demand: Sequence[Sequence[int]]) -> tuple[Wave, ...]:
    """Return weighted directed permutation waves from full global demand."""
    original = _validated_matrix(demand)
    residual, _ = _complete_balanced(original)
    components: list[tuple[int, tuple[int, ...]]] = []

    while any(any(value > 0 for value in row) for row in residual):
        matching = _positive_perfect_matching(residual)
        weight = min(residual[source][destination] for source, destination in enumerate(matching))
        if weight <= 0:
            raise AssertionError("matching weight must be positive")
        components.append((weight, matching))
        for source, destination in enumerate(matching):
            residual[source][destination] -= weight

    # FAST schedules larger permutation components first so long stages can be
    # overlapped with the local redistribution pipeline.
    components.sort(key=lambda item: (-item[0], item[1]))

    remaining = [row[:] for row in original]
    consumed = [[0] * len(original) for _ in original]
    waves: list[Wave] = []
    for weight, matching in components:
        transfers = []
        for source, destination in enumerate(matching):
            count = min(weight, remaining[source][destination])
            if count <= 0 or source == destination:
                continue
            total = original[source][destination]
            offset = consumed[source][destination]
            transfers.append((source, destination, offset, count, total))
            consumed[source][destination] += count
            remaining[source][destination] -= count
        if transfers:
            waves.append(tuple(transfers))

    if any(any(value != 0 for value in row) for row in remaining):
        raise AssertionError("weighted decomposition did not cover demand")
    return tuple(waves)


def plan_cyclic_source_local_waves(demand: Sequence[Sequence[int]]) -> tuple[Wave, ...]:
    """Return topology-fixed directed matchings using only source-local rows.

    Stage ``shift`` maps every source ``s`` to ``(s + shift) % N``.  The
    destination uniqueness is guaranteed by the cyclic construction, so no
    destination demand, global matrix exchange, or matching solver is needed.
    A distributed implementation only reads its own row; this full matrix
    argument exists so the benchmark can construct and audit all ranks' work.
    """
    matrix = _validated_matrix(demand)
    size = len(matrix)
    waves = []
    for shift in range(1, size):
        transfers = []
        for source in range(size):
            destination = (source + shift) % size
            count = matrix[source][destination]
            if count > 0:
                transfers.append((source, destination, 0, count, count))
        if transfers:
            waves.append(tuple(transfers))
    if reconstruct_demand(size, waves) != matrix:
        raise ValueError(
            "cyclic source-local waves require zero self demand and cover "
            "every positive off-diagonal edge"
        )
    return tuple(waves)


def plan_equal_chunk_interleaved_waves(
    demand: Sequence[Sequence[int]], chunk_tokens: int
) -> tuple[Wave, ...]:
    """Interleave equal-size edge chunks, then drain the final tails.

    Every original edge is split into ``chunk_tokens``-sized pieces.  At each
    chunk depth, the active pieces form a bipartite multigraph which is
    decomposed into directed matchings.  Thus every non-tail transfer in a
    wave has exactly the same size and no source or destination appears more
    than once.  Residual pieces smaller than one chunk are scheduled last
    with the weighted planner.

    This experimental planner consumes the complete node-demand matrix.  It
    is deliberately separate from the source-local cyclic fast path so its
    scheduling benefit and global-control cost can be measured honestly.
    """
    original = _validated_matrix(demand)
    if not isinstance(chunk_tokens, int) or isinstance(chunk_tokens, bool):
        raise TypeError("chunk_tokens must be an integer")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    size = len(original)
    full_chunks = [[value // chunk_tokens for value in row] for row in original]
    tails = [[value % chunk_tokens for value in row] for row in original]
    consumed = [[0] * size for _ in range(size)]
    waves: list[Wave] = []
    max_depth = max(value for row in full_chunks for value in row)

    # Breadth-first chunk order is the interleaving: every active edge sends
    # its first chunk before any edge sends its second chunk, and so on.
    for depth in range(max_depth):
        layer = [
            [int(full_chunks[source][destination] > depth) for destination in range(size)]
            for source in range(size)
        ]
        residual, _ = _complete_balanced(layer)
        while any(any(value > 0 for value in row) for row in residual):
            matching = _positive_perfect_matching(residual)
            transfers = []
            for source, destination in enumerate(matching):
                residual[source][destination] -= 1
                if layer[source][destination] == 0:
                    continue
                total = original[source][destination]
                offset = consumed[source][destination]
                transfers.append((source, destination, offset, chunk_tokens, total))
                consumed[source][destination] += chunk_tokens
                layer[source][destination] = 0
            if transfers:
                waves.append(tuple(transfers))

    # Tail transfers are smaller than the common chunk and are intentionally
    # kept after all full chunks.  Offsets are translated back to the original
    # unsplit edge so the executor can use contiguous views.
    for tail_wave in plan_fast_style_waves(tails):
        transfers = []
        for source, destination, offset, count, _ in tail_wave:
            transfers.append(
                (
                    source,
                    destination,
                    consumed[source][destination] + offset,
                    count,
                    original[source][destination],
                )
            )
        if transfers:
            waves.append(tuple(transfers))

    if reconstruct_demand(size, waves) != original:
        raise AssertionError("equal-chunk waves did not cover demand")
    return tuple(waves)


def reconstruct_demand(num_nodes: int, waves: Sequence[Wave]) -> list[list[int]]:
    """Reconstruct original directed demand; useful for audits and tests."""
    result = [[0] * num_nodes for _ in range(num_nodes)]
    expected_offsets: dict[tuple[int, int], int] = {}
    totals: dict[tuple[int, int], int] = {}
    for wave in waves:
        sources = set()
        destinations = set()
        for source, destination, offset, count, total in wave:
            if source in sources or destination in destinations:
                raise ValueError("wave is not a directed permutation")
            sources.add(source)
            destinations.add(destination)
            key = (source, destination)
            if offset != expected_offsets.get(key, 0):
                raise ValueError("edge chunks are not contiguous")
            if total <= 0 or count <= 0 or offset + count > total:
                raise ValueError("invalid transfer range")
            expected_offsets[key] = offset + count
            totals[key] = total
            result[source][destination] += count
    if any(expected_offsets[key] != total for key, total in totals.items()):
        raise ValueError("edge chunks do not cover their declared total")
    return result
