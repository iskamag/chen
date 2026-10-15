"""OFFLINE auditor only. Inference never imports this matching implementation."""


def maximum_matching(columns, resources):
    owners = [-1] * resources

    def augment(target, visited):
        for resource in range(resources):
            if not (columns[target] >> resource) & 1 or resource in visited:
                continue
            visited.add(resource)
            if owners[resource] == -1 or augment(owners[resource], visited):
                owners[resource] = target
                return True
        return False

    return sum(augment(i, set()) for i in range(len(columns)))


def feasible_edge_domains(columns, resources):
    """Exact marginal support across feasible matchings, used after inference.

    Forcing/deleting edges here is an offline audit query, never a latent state
    update or a deployed decision procedure. No answer is fed into solver.py.
    """
    assert maximum_matching(columns, resources) == len(columns)
    supports, forced = [], []
    for target, edges in enumerate(columns):
        supported = required = 0
        for resource in range(resources):
            bit = 1 << resource
            if not edges & bit:
                continue
            others = [mask & ~bit for i, mask in enumerate(columns) if i != target]
            if maximum_matching(others, resources) == len(columns) - 1:
                supported |= bit
            removed = list(columns)
            removed[target] &= ~bit
            if maximum_matching(removed, resources) < len(columns):
                required |= bit
        supports.append(supported)
        forced.append(required)
    return tuple(supports), tuple(forced)


def deduction_errors(graph, state, supports, forced):
    removed_supported = 0
    wrongly_forced = 0
    for i, edges in enumerate(graph.columns):
        removed_supported += (supports[i] & ~state.one[i]).bit_count()
        wrongly_forced += (edges & ~state.zero[i] & ~forced[i]).bit_count()
    return removed_supported, wrongly_forced
