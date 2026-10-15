#!/usr/bin/env python3
"""Render a real, legally replayed match log as an annotated MP4.

This is a replay renderer, not a screen capture or a playing engine. Display
analysis is post-game Stockfish analysis and never feeds either player's moves.
"""
import argparse
from array import array
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import wave

import cairosvg
import chess
import chess.engine
import chess.pgn
import chess.svg
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 800
LEFT, TOP, CELL = 60, 112, 77
BG = '#101720'
TEXT, MUTED = '#ecede8', '#8b9aaa'
JPEG, FISH = '#ffbd8b', '#80d7dd'
FONT = '/usr/share/fonts/TTF/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'
MONO = '/usr/share/fonts/TTF/DejaVuSansMono.ttf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Renderer:
    def __init__(self, names, candidate_color, row, analysis, pacing, architecture, budget_seconds=1.0, book_plies=0,
                 selection_note=None, captions=()):
        self.names, self.candidate_color = names, candidate_color
        self.row, self.analysis, self.pacing = row, analysis, pacing
        self.architecture = architecture
        self.budget_ms = round(budget_seconds * 1000)
        self.book_plies = book_plies
        self.selection_note = selection_note
        self.captions = captions
        self.fonts = {}
        self.pieces = {}
        for symbol in 'PNBRQKpnbrqk':
            svg = chess.svg.piece(chess.Piece.from_symbol(symbol), size=CELL - 8)
            self.pieces[symbol] = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert('RGBA')

    def font(self, size, bold=False, mono=False):
        key = size, bold, mono
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(MONO if mono else BOLD if bold else FONT, size)
        return self.fonts[key]

    def text(self, draw, xy, value, size=18, fill=TEXT, bold=False, mono=False):
        draw.text(xy, value, fill=fill, font=self.font(size, bold, mono))

    @staticmethod
    def xy(square):
        return LEFT + chess.square_file(square) * CELL, TOP + (7 - chess.square_rank(square)) * CELL

    def frame(self, board, sans, index, last=None, animation=None, title=False, final=False):
        image = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        self.text(draw, (60, 29), 'JPEG / CHESS', 26, bold=True)
        title_text = ('THE HALF-SECOND TEST' if self.budget_ms == 500 else
                      'THE ONE-SECOND TEST' if self.budget_ms == 1000 else f'{self.budget_ms} MS PER MOVE')
        self.text(draw, (766, 33), title_text, 23, fill=JPEG, bold=True)
        book_label = f'shared book (up to {self.book_plies} ply)' if self.book_plies else 'books off'
        self.text(draw, (61, 70), 'CPU only / 1 thread per bot / 16 MiB hash / ' + book_label, 15, fill=MUTED)
        hidden = set()
        if animation:
            hidden = {target for _, target, _, _ in animation}
        for square in chess.SQUARES:
            x, y = self.xy(square)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = '#e3decf' if light else '#738c82'
            if last and square in [last.from_square, last.to_square]:
                color = '#edce89' if light else '#b69e63'
            draw.rectangle((x, y, x + CELL, y + CELL), fill=color)
            if board.is_check() and square == board.king(board.turn):
                draw.rounded_rectangle((x + 3, y + 3, x + CELL - 3, y + CELL - 3), radius=13, fill='#c06d64')
            piece = board.piece_at(square)
            if piece and square not in hidden:
                image.paste(self.pieces[piece.symbol()], (x + 4, y + 4), self.pieces[piece.symbol()])
        if animation:
            for origin, target, symbol, progress in animation:
                x0, y0 = self.xy(origin)
                x1, y1 = self.xy(target)
                t = progress * progress * (3 - 2 * progress)
                xy = round(x0 + (x1 - x0) * t) + 4, round(y0 + (y1 - y0) * t) + 4
                piece = self.pieces[symbol]
                image.paste(piece, xy, piece)
        for i in range(8):
            self.text(draw, (LEFT - 25, TOP + i * CELL + 27), str(8 - i), 14, fill=MUTED, mono=True)
            self.text(draw, (LEFT + i * CELL + 32, TOP + CELL * 8 + 9), chr(97 + i), 14, fill=MUTED, mono=True)

        # Post-game analysis, explicitly labelled; scores are always White POV.
        entry = self.analysis[index]
        fraction = 1 / (1 + math.exp(-max(-2000, min(2000, entry['cp'])) / 200))
        bar_x, bar_y, bar_h = 709, TOP, CELL * 8
        split = bar_y + round((1 - fraction) * bar_h)
        draw.rounded_rectangle((bar_x, bar_y, bar_x + 13, bar_y + bar_h), radius=5, fill='#33404c')
        draw.rectangle((bar_x, split, bar_x + 13, bar_y + bar_h), fill='#e5e3d9')

        for color, y in [(chess.BLACK, 113), (chess.WHITE, 205)]:
            accent = JPEG if color == self.candidate_color else FISH
            active = board.turn == color and not final
            draw.rounded_rectangle((760, y, 1223, y + 77), radius=10, fill='#19242f', outline=accent if active else '#293643', width=2)
            self.text(draw, (777, y + 10), 'WHITE' if color else 'BLACK', 12, fill=accent, bold=True)
            self.text(draw, (777, y + 33), self.names[color], 21, bold=True)
            if active:
                draw.ellipse((1197, y + 15, 1205, y + 23), fill=accent)

        self.text(draw, (763, 292), self.architecture, 12, fill=JPEG)
        self.text(draw, (762, 312), 'POSITION', 12, fill=MUTED, bold=True)
        self.text(draw, (760, 335), 'Starting position' if index == 0 else sans[-1][0], 35, bold=True)
        if index:
            trace = self.row['move_trace'][index - 1]
            wall = trace['wall_ns'] / 1e6
            depth = trace['reported'].get('depth', '—')
            timing_label = f'Opening book · {wall:.0f} ms' if trace.get('book') else f'Actual move: {wall:.0f} ms   /   depth {depth}'
            self.text(draw, (763, 386), timing_label, 16, fill=MUTED)
        self.text(draw, (763, 420), 'POST-GAME STOCKFISH ANALYSIS', 11, fill=MUTED, bold=True)
        self.text(draw, (763, 441), entry['label'] + '  ·  White perspective', 17)
        # Visible move history does not run off the bottom of the sidebar.
        caption = next((c for c in self.captions if c['start_ply'] <= index <= c['end_ply']), None)
        shown = sans[max(0, index - (6 if caption else 10)):index]
        for j, (label, _) in enumerate(shown):
            self.text(draw, (764 + (j % 2) * 228, 492 + (j // 2) * 29), label, 16, fill=TEXT if j == len(shown) - 1 else MUTED, mono=True)
        if caption:
            for j, line in enumerate(caption['lines']):
                self.text(draw, (763, 594 + j * 22), line, 16, fill=JPEG)

        total = len(self.row['move_trace'])
        draw.rounded_rectangle((762, 667, 1220, 671), radius=2, fill='#2a3845')
        if total and index:
            draw.rounded_rectangle((762, 667, 762 + int(458 * index / total), 671), radius=2, fill=JPEG)
        self.text(draw, (762, 688), f'Ply {index:03d} / {total:03d}', 13, fill=MUTED, mono=True)
        if self.selection_note:
            self.text(draw, (61, 749), self.selection_note, 13, fill=JPEG)
        self.text(draw, (61, 770), f'Rendered replay · {self.pacing:g} s/ply · requested budget {self.budget_ms} ms/move', 13, fill=MUTED)
        if title or final:
            overlay = Image.new('RGBA', image.size)
            d = ImageDraw.Draw(overlay)
            d.rounded_rectangle((151, 316, 590, 499), radius=16, fill=(16, 23, 32, 240))
            heading = self.row['result'] if final else ('HALF A SECOND.' if self.budget_ms == 500 else
                                                       'ONE SECOND.' if self.budget_ms == 1000 else f'{self.budget_ms} MS.')
            sub = self.row['termination'].replace('_', ' ').title() if final else 'Every move. Both bots.'
            self.text(d, (181, 344), heading, 42, bold=True, fill=JPEG)
            self.text(d, (182, 411), sub, 20)
            self.text(d, (182, 451), 'Selected game · saved PGN' if self.selection_note else 'Unscripted game · saved PGN', 14, fill=MUTED)
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        return image


def analyse(boards, binary, path, nodes):
    if path.exists():
        rows = json.loads(path.read_text())
        assert len(rows) == len(boards)
        for row, board in zip(rows, boards):
            assert row['fen'] == board.fen(en_passant='fen')
        return rows
    rows = []
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:
        engine.configure({'Threads': 1, 'Hash': 16, 'NumaPolicy': 'none', 'UCI_LimitStrength': False, 'SyzygyProbeLimit': 0})
        for board in boards:
            if board.is_checkmate():
                score = chess.engine.Mate(-1 if board.turn == chess.WHITE else 1)
                label = 'Black wins' if board.turn == chess.WHITE else 'White wins'
            else:
                info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
                score = info['score'].white()
                label = f'Mate {score.mate():+d}' if score.is_mate() else f'{score.score() / 100:+.2f}'
            rows.append({'fen': board.fen(en_passant='fen'), 'cp': score.score(mate_score=100000), 'label': label})
    path.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def audio(path, frames, fps, move_frames):
    rate = 24000
    samples = array('h', [0]) * round(frames / fps * rate)
    for position in move_frames:
        offset = round(position / fps * rate)
        for n in range(round(rate * .075)):
            if offset + n >= len(samples):
                break
            t = n / rate
            value = 1700 * math.exp(-t * 65) * (math.sin(2 * math.pi * 640 * t) + .35 * math.sin(2 * math.pi * 1100 * t))
            samples[offset + n] = int(value)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--game', type=int, default=0)
    parser.add_argument('--candidate-label', required=True)
    parser.add_argument('--base-label')
    parser.add_argument('--stockfish', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--seconds-per-ply', type=float, default=1.1)
    parser.add_argument('--analysis-nodes', type=int, default=20000)
    parser.add_argument('--architecture', default='Relational proofs + conventional Chen PVS')
    parser.add_argument('--selection-note', help='Visible disclosure for a curated game, also saved in the manifest')
    parser.add_argument('--captions', type=Path, help='Post-game captions: start_ply, end_ply, lines (up to three)')
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    assert args.fps > 0 and args.seconds_per_ply > 0
    metadata = json.loads((args.run / 'metadata.json').read_text())
    budget_seconds = metadata['budgets']['movetime']
    assert budget_seconds is not None and budget_seconds > 0, 'A fixed positive move-time budget is required.'
    for engine in metadata['config'].values():
        assert engine['options'].get('Hash') == 16
        assert engine['options'].get('Threads', 1) == 1
        assert not engine['options'].get('OwnBook', False)
        assert not engine['options'].get('Ponder', False)
    base_options = metadata['config']['base']['options']
    if args.base_label is None:
        args.base_label = (f'Stockfish 19 · setting {base_options["UCI_Elo"]}'
                           if base_options.get('UCI_LimitStrength') else 'Stockfish 19 · full')
    row = list(map(json.loads, (args.run / 'games.jsonl').read_text().splitlines()))[args.game]
    assert row['failure'] is None and row['result'] != '*', 'Only completed, valid games can be presented as complete.'
    board = chess.Board(row['move_trace'][0]['fen'])
    boards, sans, moves = [board.copy()], [], []
    for index, trace in enumerate(row['move_trace']):
        assert trace['ply'] == index and trace['fen'] == board.fen(en_passant='fen')
        assert board.outcome(claim_draw=True) is None
        move = chess.Move.from_uci(trace['move'])
        assert move in board.legal_moves
        label = f'{board.fullmove_number}{"." if board.turn else "..."} {board.san(move)}'
        sans.append((label, board.turn))
        moves.append(move)
        board.push(move)
        boards.append(board.copy())
    assert board.fen(en_passant='fen') == row['final_fen']
    assert board.outcome(claim_draw=True).result() == row['result']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    analysis_path = args.out.with_suffix('.analysis.json')
    analysis = analyse(boards, args.stockfish, analysis_path, args.analysis_nodes)
    candidate_color = row['candidate_color'] == 'white'
    names = {candidate_color: args.candidate_label, not candidate_color: args.base_label}
    book_metadata = metadata.get('opening_book')
    captions = json.loads(args.captions.read_text()) if args.captions else []
    for caption in captions:
        assert 0 <= caption['start_ply'] <= caption['end_ply'] <= len(moves)
        assert 1 <= len(caption['lines']) <= 3
        assert all(isinstance(line, str) and len(line) <= 48 for line in caption['lines'])
    renderer = Renderer(names, candidate_color, row, analysis, args.seconds_per_ply, args.architecture, budget_seconds,
                        book_metadata['plies'] if book_metadata else 0, args.selection_note, captions)
    preview = renderer.frame(boards[min(12, len(moves))], sans[:min(12, len(moves))], min(12, len(moves)), last=moves[min(11, len(moves) - 1)])
    preview.save(args.preview or args.out.with_suffix('.png'))
    if args.preview_only:
        print('preview saved', flush=True)
        return
    intro, ending = 3 * args.fps, 6 * args.fps
    per_move = round(args.fps * args.seconds_per_ply)
    anim_frames = min(round(args.fps * .22), per_move)
    total_frames = intro + per_move * len(moves) + ending
    wav = args.out.with_suffix('.wav')
    audio(wav, total_frames, args.fps, [intro + (i + 1) * per_move - 1 for i in range(len(moves))])
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(args.fps), '-i', 'pipe:0', '-i', str(wav),
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p', '-threads', '2',
               '-c:a', 'aac', '-b:a', '96k', '-shortest', '-movflags', '+faststart', str(args.out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        def write(image, repeats=1):
            content = image.tobytes()
            for _ in range(repeats):
                process.stdin.write(content)
        write(renderer.frame(boards[0], [], 0, title=True), intro)
        for i, move in enumerate(moves):
            for frame in range(anim_frames):
                progress = (frame + 1) / anim_frames
                source = boards[i].piece_at(move.from_square).symbol()
                if move.promotion and progress > .8:
                    source = boards[i + 1].piece_at(move.to_square).symbol()
                animation = [(move.from_square, move.to_square, source, progress)]
                if boards[i].is_castling(move):
                    rank = chess.square_rank(move.from_square)
                    kingside = move.to_square > move.from_square
                    animation.append((chess.square(7 if kingside else 0, rank), chess.square(5 if kingside else 3, rank), 'R' if boards[i].turn else 'r', progress))
                write(renderer.frame(boards[i + 1], sans[:i + 1], i + 1, last=move, animation=animation))
            write(renderer.frame(boards[i + 1], sans[:i + 1], i + 1, last=move), per_move - anim_frames)
            if i % 20 == 0:
                print(f'rendered {i + 1}/{len(moves)} plies', flush=True)
        write(renderer.frame(boards[-1], sans, len(moves), last=moves[-1], final=True), ending)
    finally:
        process.stdin.close()
    assert process.wait() == 0, 'video encoder failed'
    manifest = {'kind': 'rendered_replay_of_actual_bot_game', 'run': str(args.run), 'game': args.game,
                'result': row['result'], 'termination': row['termination'], 'plies': len(moves),
                'candidate_label': args.candidate_label, 'base_label': args.base_label,
                'architecture': args.architecture, 'playing_config': metadata['config'],
                'requested_seconds_per_move': budget_seconds,
                'selection_note': args.selection_note,
                'captions': captions,
                'opening_book': book_metadata,
                'resolution': [WIDTH, HEIGHT], 'fps': args.fps, 'frames': total_frames,
                'seconds': total_frames / args.fps, 'seconds_per_ply_requested': args.seconds_per_ply,
                'analysis': 'Independent post-game full-strength Stockfish, not used by either playing bot',
                'analysis_nodes_per_position': args.analysis_nodes,
                'sha256': {str(p): sha(p) for p in [args.out, args.run / 'games.jsonl', args.run / 'games.pgn', args.run / 'metadata.json', args.stockfish, Path(__file__)]}}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
