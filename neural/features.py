"""Chen schema v1: side-to-move canonical squares, explicit rule-state planes.

No python-chess dependency is needed here. These are encoders, never legality
oracles. Squares use a1=0 and black canonicalization flips ranks (xor 56).
"""
from __future__ import annotations

import torch

SCHEMA_VERSION = 1
FEATURE_PLANES = 18
ACTION_COUNT = 64 * 64 * 5
PIECES = "PNBRQK"
PROMOTIONS = " nbrq"


def square_index(square: str) -> int:
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"invalid square: {square!r}")
    return ord(square[0]) - ord("a") + 8 * (ord(square[1]) - ord("1"))


def square_name(square: int) -> str:
    if not 0 <= square < 64:
        raise ValueError("square must be in [0, 64)")
    return chr(ord("a") + square % 8) + str(square // 8 + 1)


def move_to_index(uci: str, side_to_move: str = "w") -> int:
    if side_to_move not in ("w", "b") or len(uci) not in (4, 5):
        raise ValueError("expected UCI move and side 'w' or 'b'")
    flip = 56 if side_to_move == "b" else 0
    source, target = square_index(uci[:2]) ^ flip, square_index(uci[2:4]) ^ flip
    promotion = 0
    if len(uci) == 5:
        if uci[4] not in "nbrq":
            raise ValueError("invalid promotion")
        promotion = PROMOTIONS.index(uci[4])
    return (source * 64 + target) * 5 + promotion


def index_to_move(index: int, side_to_move: str = "w") -> str:
    if not 0 <= index < ACTION_COUNT or side_to_move not in ("w", "b"):
        raise ValueError("invalid action index or side")
    pair, promotion = divmod(index, 5)
    source, target = divmod(pair, 64)
    flip = 56 if side_to_move == "b" else 0
    return square_name(source ^ flip) + square_name(target ^ flip) + (PROMOTIONS[promotion] if promotion else "")


def encode_fen(fen: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Encode one FEN as [18,8,8]; validates syntax, not legal chess state."""
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError("expected six FEN fields")
    board, side, rights, ep, clock, fullmove = fields
    if rights != "-" and (any(c not in "KQkq" for c in rights) or len(set(rights)) != len(rights)):
        raise ValueError("invalid castling rights")
    halfmove, fullmove_number = int(clock), int(fullmove)
    if halfmove < 0 or fullmove_number < 1:
        raise ValueError("invalid FEN clocks")
    output = torch.zeros(FEATURE_PLANES, 8, 8, dtype=torch.float32)
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError("expected eight FEN ranks")
    for fen_rank, rank in enumerate(ranks):
        file = 0
        for character in rank:
            if character in "12345678":
                file += int(character)
            elif character.upper() in PIECES:
                if file >= 8:
                    raise ValueError("too many squares in rank")
                raw_rank = 7 - fen_rank
                canonical_rank = raw_rank if side == "w" else 7 - raw_rank
                own = character.isupper() == (side == "w")
                plane = PIECES.index(character.upper()) + (0 if own else 6)
                output[plane, canonical_rank, file] = 1
                file += 1
            else:
                raise ValueError("invalid FEN piece")
        if file != 8:
            raise ValueError("rank must contain eight squares")
    for plane, right in enumerate("KQkq" if side == "w" else "kqKQ", start=12):
        output[plane].fill_(float(right in rights))
    if ep != "-":
        square = square_index(ep) ^ (56 if side == "b" else 0)
        if ep[1] not in "36":
            raise ValueError("invalid en passant rank")
        output[16, square // 8, square % 8] = 1
    output[17].fill_(min(halfmove / 100, 1))
    return output.to(device)


def legal_probabilities(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Normalize legal moves; terminal rows return zero, with no NaNs."""
    if logits.shape != legal_mask.shape or legal_mask.dtype != torch.bool:
        raise ValueError("boolean legal mask must match logits")
    masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    probabilities = masked.softmax(-1) * legal_mask
    return probabilities / probabilities.sum(-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
