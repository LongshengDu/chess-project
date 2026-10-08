"""Refresh anonymous accuracy-curve paper figures from saved game measurements."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.accuracy.figures import export_saved_figures


def render_example(analysis_path, figure_path, move_figure_path):
    """Render both publication views without altering source identity or evidence."""
    analysis = json.loads(Path(analysis_path).read_text(encoding='utf-8'))
    anonymous = deepcopy(analysis)
    anonymous['headers'] = {'White': 'White', 'Black': 'Black'}
    for side in ('white', 'black'):
        anonymous['game'][side]['name'] = side.title()
    targets = {'curve': Path(figure_path), 'by_move': Path(move_figure_path)}
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.paper-figure-', dir=targets['curve'].parent) as staging:
        figures = export_saved_figures(anonymous, staging)
        for key, target in targets.items():
            Path(figures[key]).replace(target)
    return targets


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=root/'games/output/game10-full/analysis.json')
    parser.add_argument('--figure', type=Path, default=root/'docs/figures/accuracy_curve_game10.svg')
    parser.add_argument('--move-figure', type=Path,
                        default=root/'docs/figures/accuracy_by_move_game10_1600.svg')
    args = parser.parse_args()
    for path in render_example(args.analysis, args.figure, args.move_figure).values():
        print(path)


if __name__ == '__main__':
    main()
