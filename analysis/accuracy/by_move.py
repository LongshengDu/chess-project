"""Actual move accuracy and Maia expectations from saved game evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import chess

from analysis.accuracy.evidence import RATINGS, accuracy_moments
from analysis.accuracy.plot_style import SIDE_COLORS, SVG_STYLE

DEFAULT_MAIA_ELOS = (1600, 1800, 2000)
DEFAULT_MAIA_ELO = DEFAULT_MAIA_ELOS[0]


def record_move_accuracy(analysis, evidence):
    """Store prepared accuracy measurements inside each Elo's Maia record."""
    by_side = {side: iter(rows) for side, rows in evidence.items()}
    for move in analysis['moves']:
        row = next(by_side[move['side']])
        for rating, moments in accuracy_moments(row).items():
            move['maia'][rating].update(moments)


def expected_accuracy_by_move(analysis, maia_elo=DEFAULT_MAIA_ELO):
    """Select actual accuracy and expectations at one Elo from saved evidence.

    Each point pairs a position's expectation with the played move's accuracy,
    rather than a running game average.
    Forced positions are omitted without renumbering later moves. Only saved
    native anchors are accepted; this tool never interpolates or runs engines.
    """
    if type(maia_elo) is not int or maia_elo not in RATINGS:
        raise ValueError('Maia Elo must be a saved anchor from 600 to 2600 in steps of 100.')
    players = {'white': [], 'black': []}
    excluded = {'white': 0, 'black': 0}
    board = chess.Board(analysis['start_fen'])
    for ply, move in enumerate(analysis['moves'], 1):
        side = 'white' if board.turn else 'black'
        played = move['played']['move']
        if move['side'] != side or chess.Move.from_uci(played) not in board.legal_moves:
            raise ValueError('Saved move side or legality does not match its position.')
        maia = move.get('maia')
        values = maia.get(str(maia_elo)) if isinstance(maia, dict) else None
        expected = values.get('expected_accuracy') if isinstance(values, dict) else None
        if type(expected) not in (int, float) or not math.isfinite(expected) or not 0 <= expected <= 100:
            raise ValueError(f'Missing or invalid Maia expected_accuracy at ply {ply}; refresh the saved game analysis first.')
        if board.legal_moves.count() == 1:
            excluded[side] += 1
        else:
            accuracy = move.get('accuracy')
            if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
                raise ValueError(f'Missing or invalid played accuracy at ply {ply}; refresh the saved game analysis first.')
            players[side].append({'ply': ply, 'move_number': board.fullmove_number,
                                  'played': played, 'accuracy': accuracy, 'expected_accuracy': expected})
        board.push_uci(played)
    return {'maia_elo': maia_elo, 'players': players, 'forced_moves_excluded': excluded}


def _plot_series(axis, points, field, color, label, *, alpha, width, marker_size, marker):
    """Break solid lines at omitted forced moves; connect only endpoints with dots."""
    x, y = [], []
    for index, point in enumerate(points):
        if index and point['move_number'] > points[index-1]['move_number']+1:
            previous = points[index-1]
            x.append(math.nan)
            y.append(math.nan)
            axis.plot([previous['move_number'], point['move_number']],
                      [previous[field], point[field]], color=color, lw=width, ls=':', alpha=alpha)
        x.append(point['move_number'])
        y.append(point[field])
    axis.plot(x, y, color=color, lw=width, alpha=alpha, marker=marker, ms=marker_size,
              markerfacecolor='white' if field == 'accuracy' else color, label=label)


def export_move_curve(analysis, output_dir, maia_elo=DEFAULT_MAIA_ELO):
    """Compare both Maia sides and each player's actual accuracy in three panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    data = expected_accuracy_by_move(analysis, maia_elo)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'accuracy-by-move-{maia_elo}.svg'
    with plt.rc_context(SVG_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, sharey=True, layout='constrained')
        try:
            start = chess.Board(analysis['start_fen'])
            count = len(analysis['moves'])
            last = start.fullmove_number + ((count-1 + int(not start.turn))//2 if count else 0)
            panels = (
                ('White Maia vs Black Maia', (('white', 'expected_accuracy'), ('black', 'expected_accuracy'))),
                ('White Maia vs White player', (('white', 'expected_accuracy'), ('white', 'accuracy'))),
                ('Black Maia vs Black player', (('black', 'expected_accuracy'), ('black', 'accuracy'))),
            )
            for axis, (title, series) in zip(axes, panels, strict=True):
                for side, field in series:
                    points = data['players'][side]
                    expected = field == 'expected_accuracy'
                    label = (f'{side.title()} Maia {maia_elo}: {len(points)} pos' if expected
                             else f'{side.title()} actual')
                    _plot_series(axis, points, field, SIDE_COLORS[side], label,
                                 alpha=1. if expected else .78, width=1.8 if expected else 1.5,
                                 marker_size=3.5 if expected else 4., marker='o' if expected else 's')
                axis.set(ylabel='Accuracy (%)', ylim=(-2, 102),
                         xlim=(start.fullmove_number-.5, last+.5), title=title)
                axis.set_yticks(range(0, 101, 20))
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.tick_params(axis='x', labelbottom=True)
                axis.grid(alpha=.18)
                handles, labels = axis.get_legend_handles_labels()
                if any(b['move_number'] > a['move_number']+1 for side, _ in series
                       for a, b in zip(data['players'][side], data['players'][side][1:])):
                    handles.append(Line2D([], [], color='#777777', lw=1.2, ls=':'))
                    labels.append('Skipped forced moves')
                axis.legend(handles, labels, loc='best', ncols=len(handles), fontsize=9, framealpha=.95)
            axes[-1].set_xlabel('PGN move number')
            game = analysis['game']
            fig.suptitle(f'{game["white"]["name"] or "White"} — {game["black"]["name"] or "Black"}\n'
                         f'Accuracy by move · Maia {maia_elo} (Lichess Blitz)', fontsize=13)
            with NamedTemporaryFile(dir=directory, suffix='.svg', delete=False) as temp:
                temporary = Path(temp.name)
            try:
                fig.savefig(temporary, format='svg', metadata={'Date': None})
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            plt.close(fig)
    return {'curve': str(target)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path, help='Saved analysis.json containing per-move expected accuracies.')
    parser.add_argument('--maia-elo', type=int, choices=RATINGS, default=DEFAULT_MAIA_ELO,
                        help='Native Lichess Blitz Maia anchor; both rating inputs use this value (default: 1600).')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory; defaults to the saved analysis directory.')
    args = parser.parse_args(argv)
    try:
        analysis = json.loads(args.analysis.read_text(encoding='utf-8'))
        paths = export_move_curve(analysis, args.output_dir or args.analysis.parent, args.maia_elo)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Figure: {paths['curve']}")
    return paths


if __name__ == '__main__':
    main()
