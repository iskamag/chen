"""Fixed-graph Boolean support messages. No matching oracle is imported here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Graph:
    resources: int
    columns: tuple[int, ...]


@dataclass(frozen=True)
class State:
    # Bits mark edge variables that still permit value zero / one.
    zero: tuple[int, ...]
    one: tuple[int, ...]
    contradiction: str | None = None


def initial(graph):
    return State(graph.columns, graph.columns)


def step(graph, old, capacity=True):
    """All factors read old; their messages are intersected only in the output.

    Exactly-one and at-most-one supports are evaluated by counts, never by
    assignments. In particular, no updated row is read by another factor in
    this round. One call traverses one factor dependency, not an unbounded wave.
    """
    if old.contradiction:
        return old
    zero, one = list(old.zero), list(old.one)
    for target, edges in enumerate(graph.columns):
        if edges & ~(old.zero[target] | old.one[target]):
            return State(old.zero, old.one, f"empty edge domain at target {target}")
        forced = old.one[target] & ~old.zero[target]
        if old.one[target] == 0:
            return State(old.zero, old.one, f"target {target} has no candidate")
        if forced.bit_count() > 1:
            return State(old.zero, old.one, f"target {target} has multiple forced choices")
        if forced:
            one[target] &= forced
            zero[target] &= ~forced
        elif old.one[target].bit_count() == 1:
            zero[target] &= ~old.one[target]
    if capacity:
        for resource in range(graph.resources):
            bit = 1 << resource
            forced_targets = [i for i in range(len(graph.columns))
                              if old.one[i] & ~old.zero[i] & bit]
            if len(forced_targets) > 1:
                return State(old.zero, old.one, f"resource {resource} has multiple forced claimants")
            if forced_targets:
                owner = forced_targets[0]
                for target in range(len(graph.columns)):
                    if target != owner:
                        one[target] &= ~bit
    updated = State(tuple(zero), tuple(one))
    for target, edges in enumerate(graph.columns):
        if edges & ~(updated.zero[target] | updated.one[target]):
            return State(updated.zero, updated.one, f"intersected messages empty an edge at target {target}")
    return updated


def status(graph, state):
    if state.contradiction:
        return "INFEASIBLE"
    witness = [one & ~zero for zero, one in zip(state.zero, state.one)]
    if any(value.bit_count() != 1 for value in witness):
        return "UNKNOWN"
    occupied = 0
    for target, value in enumerate(witness):
        if occupied & value or not (value & graph.columns[target]):
            return "UNKNOWN"
        if graph.columns[target] & ~(state.zero[target] | value):
            return "UNKNOWN"
        occupied |= value
    return "FEASIBLE"


def stages(graph, rounds, mode="coupled_synchronous"):
    original = initial(graph)
    result = [original]
    first_pass = None
    for _ in range(rounds):
        if mode == "input_only_onepass":
            if first_pass is None:
                first_pass = step(graph, original)
            new = first_pass
        elif mode == "repeat_original_no_feedback":
            new = step(graph, original)
        else:
            new = step(graph, result[-1], capacity=mode != "target_only_no_capacity_messages")
        result.append(new)
    return result


def explain_round(graph, old):
    """Selected-example diagnostics, excluded from measured inference costs."""
    target_messages = []
    capacity_messages = []
    if old.contradiction:
        return {"target_messages": [], "capacity_messages": []}
    for target in range(len(graph.columns)):
        forced = old.one[target] & ~old.zero[target]
        if forced.bit_count() == 1:
            target_messages.append({"target": target, "require_one": forced,
                                    "require_zero": old.one[target] & ~forced})
        elif old.one[target].bit_count() == 1:
            target_messages.append({"target": target, "require_one": old.one[target], "require_zero": 0})
    for resource in range(graph.resources):
        bit = 1 << resource
        owners = [i for i in range(len(graph.columns)) if old.one[i] & ~old.zero[i] & bit]
        if len(owners) == 1:
            recipients = [i for i, mask in enumerate(old.one) if i != owners[0] and mask & bit]
            if recipients:
                capacity_messages.append({"resource": resource, "owner": owners[0],
                                          "recipients_require_zero": recipients})
    return {"target_messages": target_messages, "capacity_messages": capacity_messages}
