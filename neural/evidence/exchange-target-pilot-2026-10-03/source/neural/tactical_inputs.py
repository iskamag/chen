"""Exact CPU replay/reference ingress for the sparse tactical operator.

No Torch dependency until tensor emission. No teacher, learned score, child
target or future PV is accepted. This is not a device-resident producer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import chess

SCHEMA = 'chen_tactical_exact_v1'
VALUES = (0, 100, 320, 330, 500, 900, 0)


def canonical_square(square, mover):
    return square ^ (0 if mover else 56)


def canonical_mask(mask, mover):
    return int(mask) if mover else int.from_bytes(int(mask).to_bytes(8, 'little'), 'big')


def action_id(move, mover):
    promotion = 0 if move.promotion is None else ' nbrq'.index(chess.piece_symbol(move.promotion))
    return (canonical_square(move.from_square, mover) * 64 +
            canonical_square(move.to_square, mover)) * 5 + promotion


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _hash_key(offset):
    mask = (1 << 64) - 1
    value = (0x4348454E00000000 + offset + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def position_hash(board):
    """Chen standard-chess Zobrist layout; effective EP requires a LEGAL capture."""
    rights = sum(1 << i for i, (color, kingside) in enumerate(
        ((True, True), (True, False), (False, True), (False, False)))
        if (board.has_kingside_castling_rights(color) if kingside else
            board.has_queenside_castling_rights(color)))
    value = _hash_key(768 + rights) ^ (_hash_key(792) if not board.turn else 0)
    if board.has_legal_en_passant():
        value ^= _hash_key(784 + chess.square_file(board.ep_square))
    for square, piece in board.piece_map().items():
        value ^= _hash_key((piece.piece_type - 1 + (0 if piece.color else 6)) * 64 + square)
    return value


def _code(piece, mover):
    return 0 if piece is None else piece.piece_type + (0 if piece.color == mover else 6)


def _material(board, mover):
    return sum(VALUES[p.piece_type] * (1 if p.color == mover else -1)
               for p in board.piece_map().values())


def _terminal(board):
    if not any(board.legal_moves):
        return 'checkmate' if board.is_check() else 'stalemate'
    if board.halfmove_clock >= 100:
        return 'fifty_move'
    if board.is_repetition(3):
        return 'threefold'
    if board.is_insufficient_material():
        return 'insufficient_material'
    return None


@dataclass(frozen=True)
class ReplaySpec:
    initial_fen: str = chess.STARTING_FEN
    moves: tuple[str, ...] = ()
    explicit_fresh_seed: bool = False


@dataclass(frozen=True)
class ActionEffect:
    id: int
    uci: str
    source: int
    target: int
    promotion: int                 # canonical action suffix 0/n1/b2/r3/q4
    capture: bool
    en_passant: bool
    castling: bool
    gives_check: bool
    victim_square: int
    edits: tuple[tuple[int, int, int], ...]  # canonical square, old/new parent-color code
    material_parent_cp: int


@dataclass(frozen=True)
class CompactPosition:
    replay: ReplaySpec
    fen: str
    mover_white: bool
    board: tuple[int, ...]          # 0 empty; own1..6; opponent7..12
    source_attacks: tuple[int, ...] # canonical source -> uint64 canonical targets
    pinned: tuple[int, int]
    pinner_by_square: tuple[int, ...]
    checkers: int                   # only checkers of CURRENT mover king
    castling: tuple[bool, ...]      # ownK/ownQ/opponentK/opponentQ
    ep_square: int
    history_counters: tuple[int, int, int, int]
    previous_hashes: tuple[int, ...]
    position_hash: int
    node_id: int
    root_epoch: int
    context_sha256: str
    terminal: str | None
    material_mover_cp: int
    actions: tuple[ActionEffect, ...]


def _pinner(board, square):
    """Find the first slider beyond an absolutely pinned piece on its king ray."""
    piece = board.piece_at(square)
    if piece is None or not board.is_pinned(piece.color, square):
        return -1
    king = board.king(piece.color)
    dx, dy = chess.square_file(square) - chess.square_file(king), chess.square_rank(square) - chess.square_rank(king)
    if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
        raise ValueError('oracle pin has no king ray')
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    x, y = chess.square_file(square) + sx, chess.square_rank(square) + sy
    while 0 <= x < 8 and 0 <= y < 8:
        target = chess.square(x, y)
        blocker = board.piece_at(target)
        if blocker:
            allowed = (chess.BISHOP, chess.QUEEN) if sx and sy else (chess.ROOK, chess.QUEEN)
            if blocker.color != piece.color and blocker.piece_type in allowed:
                return target
            break
        x, y = x + sx, y + sy
    raise ValueError('oracle pin has no pinner')


def _replay(spec):
    if (not isinstance(spec, ReplaySpec) or type(spec.explicit_fresh_seed) is not bool or
            not isinstance(spec.moves, tuple) or not isinstance(spec.initial_fen, str)):
        raise ValueError('immutable ReplaySpec required')
    board = chess.Board(spec.initial_fen)
    if not board.is_valid() or board.chess960:
        raise ValueError('valid standard-chess seed required')
    standard = board.fen(en_passant='fen') == chess.STARTING_FEN
    if not standard and (not spec.explicit_fresh_seed or board.halfmove_clock != 0):
        raise ValueError('nonstandard seed must explicitly start fresh history at halfmove zero')
    previous = []
    for uci in spec.moves:
        if _terminal(board) is not None:
            raise ValueError('replay crosses an engine-adjudicated terminal state')
        if not isinstance(uci, str):
            raise ValueError('UCI replay strings required')
        move = chess.Move.from_uci(uci)
        if not move or move not in board.legal_moves:
            raise ValueError('illegal/null replay move')
        previous.append(position_hash(board))
        board.push(move)
    return board, previous


def compact_from_replay(spec=ReplaySpec(), *, node_id=0, root_epoch=0, ply_from_search_root=0):
    """Replay truth, including current-history draws. No unknown FEN prehistory."""
    for value in (node_id, root_epoch, ply_from_search_root):
        if type(value) is not int or not 0 <= value < 1 << 64:
            raise ValueError('nonnegative uint64 identity/counter required')
    if ply_from_search_root > len(spec.moves):
        raise ValueError('search ply cannot exceed admitted replay length')
    board, previous = _replay(spec)
    mover = board.turn
    codes, attacks, pinners = [0] * 64, [0] * 64, [-1] * 64
    pins = [0, 0]
    for square, piece in board.piece_map().items():
        target = canonical_square(square, mover)
        codes[target] = _code(piece, mover)
        attacks[target] = canonical_mask(board.attacks_mask(square), mover)
        pinner = _pinner(board, square)
        if pinner >= 0:
            pinners[target] = canonical_square(pinner, mover)
            pins[int(piece.color != mover)] |= 1 << target
    actions = []
    for move in sorted(board.legal_moves, key=lambda m: action_id(m, mover)):
        capture, ep, castle = board.is_capture(move), board.is_en_passant(move), board.is_castling(move)
        victim = move.to_square + (-8 if mover else 8) if ep else move.to_square if capture else -1
        old = board.piece_map()
        check = board.gives_check(move)
        board.push(move)
        new = board.piece_map()
        edits = tuple(sorted((canonical_square(s, mover), _code(old.get(s), mover), _code(new.get(s), mover))
                             for s in set(old) | set(new) if old.get(s) != new.get(s)))
        material = _material(board, mover)
        board.pop()
        index = action_id(move, mover)
        actions.append(ActionEffect(index, move.uci(), canonical_square(move.from_square, mover),
                                    canonical_square(move.to_square, mover), index % 5,
                                    capture, ep, castle, check,
                                    canonical_square(victim, mover) if victim >= 0 else -1, edits, material))
    current = position_hash(board)
    retained = tuple(previous[-board.halfmove_clock:]) if board.halfmove_clock else ()
    history = (*retained, current)
    rights = tuple(board.has_kingside_castling_rights(c) if k else board.has_queenside_castling_rights(c)
                   for c in (mover, not mover) for k in (True, False))
    counters = (board.halfmove_clock, history.count(current), len(history), ply_from_search_root)
    context = _digest([asdict(spec), current, history, node_id, root_epoch, counters])
    return CompactPosition(spec, board.fen(en_passant='fen'), mover, tuple(codes), tuple(attacks),
                           tuple(pins), tuple(pinners), canonical_mask(board.checkers_mask(), mover),
                           rights, canonical_square(board.ep_square, mover) if board.ep_square is not None else -1,
                           counters, retained, current, node_id, root_epoch, context, _terminal(board),
                           _material(board, mover), tuple(actions))


def admit_compact(state):
    """Cold authoritative replay comparison, not certification by an identity string."""
    if not isinstance(state, CompactPosition):
        raise ValueError('expected CompactPosition')
    expected = compact_from_replay(state.replay, node_id=state.node_id, root_epoch=state.root_epoch,
                                   ply_from_search_root=state.history_counters[3])
    if state != expected:
        raise ValueError('compact packet disagrees with exact replay')


def tactical_config(**kwargs):
    from .tactical_operator import TacticalConfig
    return TacticalConfig(feature_schema=SCHEMA, **kwargs)


def topology_packet(states):
    """CPU ABI1 oracle words; uint64 represented by signed int64 storage bits.

    Native input also needs board/action/history ownership from its authoritative
    state exporter. These words alone cannot construct a TacticalBatch.
    """
    import torch
    states = tuple(states)
    if not states:
        raise ValueError('nonempty states required')
    for state in states:
        admit_compact(state)
    signed = lambda value: value if value < 1 << 63 else value - (1 << 64)
    return {name: torch.tensor(values, dtype=torch.int64) for name, values in dict(
        source_attacks=[[signed(x) for x in s.source_attacks] for s in states],
        pinned=[[signed(x) for x in s.pinned] for s in states],
        checkers=[signed(s.checkers) for s in states],
        pinner_by_square=[list(s.pinner_by_square) for s in states],
        position_hash=[signed(s.position_hash) for s in states],
        node_id=[signed(s.node_id) for s in states],
        root_epoch=[signed(s.root_epoch) for s in states]).items()}


def admit_native_topology(packet, states):
    """Cold CPU content oracle for copied native ABI1 output, including identity."""
    import torch
    expected = topology_packet(states)
    if not isinstance(packet, dict) or set(packet) != set(expected):
        raise ValueError('exact ABI1 topology field set required')
    for name, value in packet.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                value.dtype != torch.int64 or value.shape != expected[name].shape or
                not torch.equal(value, expected[name])):
            raise ValueError(f'native topology disagrees with replay: {name}')


def to_tactical_batch(states, *, max_edges=None):
    """CPU exact sparse emitter. Overflow rejects before allocation; never truncates."""
    import torch
    from .tactical_operator import Relation as R, TacticalBatch, admit_contents

    states = tuple(states)
    if not states or (max_edges is not None and (type(max_edges) is not int or max_edges < 1)):
        raise ValueError('nonempty states and positive optional edge capacity required')
    for state in states:
        admit_compact(state)
    p, k = 32, max(len(state.actions) for state in states)
    rows = []
    for state in states:
        locations = [s for s, code in enumerate(state.board) if code]
        objects = {square: slot for slot, square in enumerate(locations)}
        if len(locations) > p:
            raise ValueError('more than 32 piece objects')
        node_square = locations + [-1] * (p - len(locations)) + list(range(64)) + [a.target for a in state.actions]
        incoming = [[0] * 64 for _ in range(2)]
        for source, mask in enumerate(state.source_attacks):
            for target in chess.scan_forward(mask):
                incoming[int(state.board[source] > 6)][target] += 1
        phase = min(24, sum((0, 0, 1, 1, 2, 4, 0)[(c - 1) % 6 + 1] for c in state.board if c)) / 24
        history = [min(1., value / scale) for value, scale in zip(state.history_counters, (100, 3, 256, 256))]
        pieces = []
        for s in locations:
            pieces.append([float(state.board[s] == c) for c in range(1, 13)] +
                          [s % 8 / 7, s // 8 / 7, float(state.pinner_by_square[s] >= 0), float(state.checkers >> s & 1)])
        squares = [[min(8, incoming[c][s]) / 8 for c in range(2)] +
                   [min(8, state.source_attacks[s].bit_count()) / 8 if (state.board[s] > 6) == bool(c) and state.board[s] else 0.
                    for c in range(2)] + [float(mask >> s & 1) for mask in state.pinned] +
                   [float(state.checkers >> s & 1), phase] + history for s in range(64)]
        actions = []
        for a in state.actions:
            victim = state.board[a.victim_square] if a.victim_square >= 0 else 0
            actions.append([a.source % 8 / 7, a.source // 8 / 7, a.target % 8 / 7, a.target // 8 / 7,
                            float(a.capture), float(a.en_passant), float(a.castling), float(a.gives_check),
                            a.promotion / 4, VALUES[(victim - 1) % 6 + 1] / 900 if victim else 0.,
                            VALUES[(state.board[a.source] - 1) % 6 + 1] / 900, a.material_parent_cp / 32000])
        edges = []
        def edge(source, receiver, kind, edit=None):
            ss, rs = node_square[source], node_square[receiver]
            old, new = (state.board[rs], state.board[rs]) if edit is None else edit
            edges.append((source, receiver, int(kind), [(rs % 8 - ss % 8) / 7,
                         (rs // 8 - ss // 8) / 7, old / 12, new / 12]))
        for s, slot in objects.items():
            edge(slot, p + s, R.OCCUPIES)
            edge(p + s, slot, R.OCCUPIED_BY)
            own = state.board[s] <= 6
            for target in chess.scan_forward(state.source_attacks[s]):
                edge(slot, p + target, R.PSEUDOATTACK)
                if target in objects:
                    receiver_own = state.board[target] <= 6
                    kind = (R.OWN_SUPPORT_OWN if receiver_own else R.OWN_ATTACK_ENEMY) if own else (
                        R.ENEMY_ATTACK_OWN if receiver_own else R.ENEMY_SUPPORT_ENEMY)
                    edge(slot, objects[target], kind)
            pinner = state.pinner_by_square[s]
            if pinner >= 0:
                king = state.board.index(6 if own else 12)
                edge(slot, objects[king], R.PINNED_TO_KING)
                edge(objects[pinner], slot, R.PINNER_TO_PINNED)
            if state.checkers >> s & 1:
                edge(slot, objects[state.board.index(6)], R.CHECKER_TO_KING)
        for slot, a in enumerate(state.actions):
            action = p + 64 + slot
            edge(objects[a.source], action, R.ACTION_SOURCE)
            edge(p + a.target, action, R.ACTION_DESTINATION)
            if a.victim_square >= 0:
                edge(objects[a.victim_square], action, R.ACTION_VICTIM)
            for square, old, new in a.edits:
                edge(action, p + square, R.ACTION_EFFECT, (old, new))
        if max_edges is not None and len(edges) > max_edges:
            raise ValueError('topology overflow: reject packet, never truncate')
        rows.append((locations, pieces, squares, actions, edges))
    b, e = len(states), max(len(row[4]) for row in rows)
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
    batch = TacticalBatch(zeros(b, p, 16), zeros(b, 64, 12), zeros(b, k, 12),
                          torch.zeros(b, p, dtype=torch.bool), torch.full((b, p), -1, dtype=torch.int64),
                          torch.zeros(b, k, dtype=torch.bool), torch.full((b, k), -1, dtype=torch.int64),
                          torch.zeros(b, e, 2, dtype=torch.int64), torch.zeros(b, e, dtype=torch.int64),
                          torch.zeros(b, e, dtype=torch.bool), zeros(b, e, 4), torch.zeros(b, dtype=torch.bool),
                          zeros(b), zeros(b, k), feature_schema=SCHEMA)
    for row, (state, values) in enumerate(zip(states, rows)):
        locations, pieces, squares, actions, edges = values
        n, a, m = len(locations), len(actions), len(edges)
        batch.pieces[row, :n] = torch.tensor(pieces, dtype=torch.float32)
        batch.piece_mask[row, :n] = True
        batch.piece_squares[row, :n] = torch.tensor(locations)
        batch.squares[row] = torch.tensor(squares)
        if a:
            batch.actions[row, :a] = torch.tensor(actions)
            batch.action_mask[row, :a] = True
            batch.action_ids[row, :a] = torch.tensor([action.id for action in state.actions])
            batch.action_prior_cp[row, :a] = torch.tensor([action.material_parent_cp for action in state.actions])
        batch.value_prior_cp[row] = state.material_mover_cp
        if m:
            batch.edge_index[row, :m] = torch.tensor([[edge[0], edge[1]] for edge in edges])
            batch.edge_kind[row, :m] = torch.tensor([edge[2] for edge in edges])
            batch.edge_mask[row, :m] = True
            batch.edge_features[row, :m] = torch.tensor([edge[3] for edge in edges])
    admit_contents(batch, tactical_config(), expected_legal_actions=[[a.id for a in s.actions] for s in states])
    return batch
