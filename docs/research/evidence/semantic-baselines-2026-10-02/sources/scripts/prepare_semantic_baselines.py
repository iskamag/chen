#!/usr/bin/env python3
"""Prepare label-free native static baseline and exact terminal witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import chess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def invoke(binary, text):
    result = subprocess.run([str(binary)], input=text, text=True, capture_output=True, timeout=120)
    require(result.returncode == 0, result.stderr)
    require(len(result.stdout) <= 16 << 20, 'oracle output exceeds bound')
    return [json.loads(line) for line in result.stdout.splitlines()]


def build(args):
    require(sha(args.publication) == args.publication_sha256, 'publication pin differs')
    require(sha(args.baseline_binary) == args.baseline_binary_sha256, 'baseline binary pin differs')
    require(sha(args.terminal_oracle) == args.terminal_oracle_sha256, 'terminal oracle pin differs')
    require(sha(args.oracle_build) == args.oracle_build_sha256, 'terminal source binding differs')
    publication = json.loads(args.publication.read_bytes())
    require(set(publication) == {'train', 'validation', 'test'}, 'split inventory')
    splits = {}
    for split, entry in publication.items():
        folder = Path(entry['path'])
        require(sha(folder / 'manifest.json') == entry['manifest_sha256'], 'pack manifest differs')
        manifest = json.loads((folder / 'manifest.json').read_bytes())
        require(manifest['payload_bytes'] <= 64 << 20, 'payload cap')
        blob = (folder / 'payload.bin').read_bytes()
        require(len(blob) == manifest['payload_bytes'] and hashlib.sha256(blob).hexdigest() == manifest['payload_sha256'], 'payload binding')

        def values(name, fmt):
            spec = manifest['tables'][name]
            raw = blob[spec['offset']:spec['offset'] + spec['nbytes']]
            require(len(raw) == spec['nbytes'], 'truncated table')
            return [row[0] for row in struct.iter_unpack('<' + fmt, raw)]

        boards, fields = values('states.boards', 'q'), values('states.fields', 'i')
        hashes = [h & ((1 << 64) - 1) for h in values('states.hash', 'q')]
        terminals = values('states.terminal', 'B')
        offsets, history = values('history.offsets', 'q'), values('history.keys', 'q')
        count = manifest['counts']['states']
        require(len(boards) == count * 12 and len(fields) == count * 5 and len(hashes) == len(terminals) == count, 'state geometry')
        fens = []
        for index in range(count):
            board = chess.Board(None)
            for piece in range(12):
                occupied = boards[index * 12 + piece] & ((1 << 64) - 1)
                for square in chess.scan_forward(occupied):
                    board.set_piece_at(square, chess.Piece(piece % 6 + 1, piece < 6))
            side, rights, ep, halfmove, fullmove = fields[index * 5:index * 5 + 5]
            board.turn = side == 0
            board.castling_rights = sum(chess.BB_SQUARES[square] for flag, square in
                                        [(1, chess.H1), (2, chess.A1), (4, chess.H8), (8, chess.A8)] if rights & flag)
            board.ep_square = None if ep == -1 else ep
            board.halfmove_clock = halfmove
            board.fullmove_number = fullmove & ((1 << 32) - 1)
            require(board.is_valid(), 'invalid reconstructed state')
            fens.append(board.fen(en_passant='fen'))
        rows = invoke(args.baseline_binary, ''.join(fen + '\n' for fen in fens))
        require(len(rows) == count and all(row['row'] == i and row['hash'] == hashes[i] and
                row['kind'] == 'raw_native_static_mover_CP_no_terminal_override' for i, row in enumerate(rows)), 'static baseline exact state/hash join differs')
        terminal_ids = [i for i, terminal in enumerate(terminals) if terminal]
        terminal_rows = invoke(args.terminal_oracle, ''.join(fens[i] + '\t' + ','.join(
            f'{h & ((1 << 64) - 1):x}' for h in history[offsets[i]:offsets[i + 1]]) + '\n' for i in terminal_ids))
        require(len(terminal_rows) == len(terminal_ids), 'terminal witness coverage')
        witnesses = []
        for local, (state_id, row) in enumerate(zip(terminal_ids, terminal_rows)):
            state = row['state']
            require(row['row'] == local and state['hash'] == hashes[state_id] and state['terminal'] is True, 'terminal state binding differs')
            require(state['legal'] or not state['in_check'], 'mate terminal requires a separately declared policy')
            witnesses.append(dict(state_id=state_id, kind='draw', in_check=state['in_check'], legal_count=len(state['legal'])))
        splits[split] = dict(manifest_sha256=entry['manifest_sha256'],
                             raw_tapered_state_cp=[r['native_tapered_cp'] for r in rows],
                             native_material_state_cp=[r['native_material_cp'] for r in rows],
                             native_hashes=hashes, observation_state_ids=values('observations.state', 'i'),
                             terminal_states=witnesses)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / 'examples/corpus_baselines.rs', root / 'src/core.rs', root / 'src/eval.rs', Path(chess.__file__).resolve()]
    return dict(kind='semantic_exact_baseline_witness_v1', splits=splits,
                provenance=dict(sources_sha256={str(p): sha(p) for p in sources},
                                cpu_binary_path=str(args.baseline_binary.resolve()), cpu_binary_sha256=args.baseline_binary_sha256,
                                terminal_oracle_path=str(args.terminal_oracle.resolve()), terminal_oracle_sha256=args.terminal_oracle_sha256,
                                terminal_oracle_build_path=str(args.oracle_build.resolve()), terminal_oracle_build_sha256=args.oracle_build_sha256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('publication', 'baseline-binary', 'terminal-oracle', 'oracle-build'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({split: dict(states=len(row['native_hashes']), terminal_draws=len(row['terminal_states']))
                      for split, row in result['splits'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
