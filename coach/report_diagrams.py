"""Verified chess-board SVGs and report image provenance."""
from __future__ import annotations

import re
from pathlib import Path

import chess
import chess.svg

from analysis.cache import identity


class BoardDiagrams:
    def __init__(self, analysis, board, output_dir):
        self.analysis, self.board, self.directory = analysis, board, Path(output_dir)
        self.paths = set()

    def render(self, history, arrows=(), *, played=False):
        board = self.board(history)
        played = played if arrows else False
        orientation = self.analysis['selected_player']['side'] == 'white'
        relative = f'positions/{identity([self.analysis["start_fen"], history, arrows, orientation, played])[:16]}.svg'
        colors = ('#e69138cc', '#189b55cc') if played else ('#189b55cc', '#e69138cc')
        converted = [chess.svg.Arrow(chess.Move.from_uci(uci).from_square, chess.Move.from_uci(uci).to_square,
                     color=colors[index % len(colors)]) for index, uci in enumerate(arrows)]
        fill = {chess.Move.from_uci(uci).to_square: colors[index % len(colors)][:7]+'55' for index, uci in enumerate(arrows)}
        svg = chess.svg.board(board, orientation=orientation, arrows=converted,
                              fill=fill, check=board.king(board.turn) if board.is_check() else None,
                              lastmove=board.peek() if board.move_stack else None, size=420)
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding='utf-8')
        self.paths.add(relative)
        return relative

    def validate(self, answer):
        links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', answer)
        if not links or any(link not in self.paths for link in links):
            raise ValueError('Include valid board diagram paths returned by the chess tools, using Markdown image syntax.')
