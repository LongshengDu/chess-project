"""Render the hierarchical-affine paper from cached evidence, without references."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import chess.pgn
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from analysis.cache import write_json
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.scale import display_fit, normalize_evidence, rating_context
from analysis.settings import CONFIG
from tests.analysis.compare_blitz_rating_methods import current_pgn_context, intersection
from tests.analysis.compare_rating_methods import load_cases


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT/'docs/figures/hierarchical-affine'
OUTPUT = ROOT/'tests/analysis/output/hierarchical-affine-paper'
FILENAMES = ('model-flow.svg', 'curve-blending.svg', 'translated-prior.svg',
             'affine-decision.svg', 'game10-worked-example.svg')


def figure(size):
    result = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(result)
    return result


def save(fig, path):
    staged = path.with_suffix('.svg.tmp')
    try:
        with rc_context({'svg.fonttype': 'none'}):
            fig.savefig(staged, format='svg', facecolor='white')
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
        fig.clear()


def flow():
    fig = figure((12., 6.2))
    axis = fig.subplots()
    axis.set(xlim=(0., 1.), ylim=(0., 1.))
    axis.axis('off')
    nodes = [(.02, .72, .45, .22, 'GAME EVIDENCE\nMaia probabilities + Stockfish move quality\nArithmetic curve C(r), observed A, variance v', '#e8f1f3'),
             (.55, .72, .43, .22, 'FROZEN CONTEXT POPULATION\nMean curve P(r), between-context spread τ²\nMatching target context excluded', '#f0ebf5'),
             (.02, .37, .50, .24, 'SHRINK THE ACCURACY CURVE\nw = τ² / (τ² + v)\nM(r) = w C(r) + (1 − w) P(r)\ns² = v + τ²v / (τ² + v)', '#e5f1e8'),
             (.61, .37, .37, .24, 'COMMON ACCOUNT PRIOR\nConvert actual ratings to native Elo\nAverage available ratings\nTranslate, truncate, normalize π(r)', '#edf0f5'),
             (.14, .025, .72, .235, 'ONE AFFINE DECISION FOR BOTH PLAYERS\nβ = Covπ(R, M(R)) / (Varπ(M(R)) + s²)\nEstimate = Eπ[R] + β (A − Eπ[M(R)])\nClip on native support, then convert to the displayed scale', '#edf3ea')]
    for left, bottom, width, height, text, color in nodes:
        axis.add_patch(FancyBboxPatch((left, bottom), width, height, boxstyle='round,pad=.009',
                                     facecolor=color, edgecolor='#6f8190', linewidth=1.))
        axis.text(left+width/2, bottom+height/2, text, ha='center', va='center', fontsize=11, linespacing=1.55)
    for start, end in (((.24, .71), (.24, .62)), ((.7, .71), (.44, .62)),
                       ((.25, .36), (.35, .27)), ((.77, .36), (.67, .27))):
        axis.annotate('', end, xytext=start, arrowprops={'arrowstyle': '-|>', 'lw': 1.5, 'color': '#526575'})
    return fig


def numeric_example(case, native, displayed):
    diagnostics = native['diagnostics']
    hierarchy, affine = diagnostics['hierarchy'], diagnostics['affine']
    actual_source = displayed['rating_scale']['actual_ratings']
    return {'game': case['name'], 'moves': case['game'].accept(chess.pgn.StringExporter(headers=False, variations=False, comments=False)),
            'source_context': {key: case['game'].headers[key] for key in ('Site', 'TimeControl', 'WhiteElo', 'BlackElo')},
            'rating_scale': displayed['rating_scale']['name'], 'actual_source': actual_source,
            'actual_native': displayed['rating_scale']['native_actual_ratings'],
            'parameters': native['parameters'], 'calibration': diagnostics['calibration'],
            'hierarchy': {key: hierarchy[key] for key in ('measurement_variance', 'between_context_variance', 'local_weight', 'residual_variance')},
            'affine': affine, 'model_accuracy_variance': affine['accuracy_variance']-hierarchy['residual_variance'],
            'players': {side: {'accuracy': values['average_accuracy'], 'native_estimate': values['unrounded_estimate'],
                         'display_estimate': displayed['players'][side]['unrounded_estimate'],
                         'accuracy_adjustment': diagnostics['components'][side]['accuracy_adjustment'],
                         'raw_intersection': intersection(diagnostics['curve']['monotone_expected_accuracy'], values['average_accuracy'])}
                        for side, values in native['players'].items()},
            'evidence_key': case['key']}


def render(*, figures=FIGURES, output=OUTPUT):
    """Read current PGN actuals and frozen chess evidence; never reference headers."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    # Kept as an unreachable historical record; these implementations were removed.
    from analysis.player_rating.hierarchical_affine import Rating
    from analysis.player_rating.figures import (COLORS, DISPLAY_RATING_RANGE, _hierarchical_curves,
        _hierarchical_mapping, _hierarchical_prior, _hierarchical_variance, export_figures)
    figures, output = Path(figures), Path(output)
    cases = load_cases(ROOT/'games', (10, 15), Path(CONFIG['ANALYSIS']['CACHE_DIR']))
    prepared = {}
    for case in cases:
        source_evidence = current_pgn_context(case)
        context = rating_context(case['game'].headers)
        native_evidence = validate_evidence(normalize_evidence(source_evidence, context))
        native = Rating().fit(native_evidence)
        actual = {side: source_evidence[side]['actual_rating'] for side in ('White', 'Black')}
        displayed = display_fit(native, context, actual)
        prepared[case['name']] = (case, native, displayed)
    # Validate every input before publishing any paper asset.
    for case in cases:
        for path, digest in case['inputs'].items():
            if sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Example input changed: {path}')
    figures.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    examples = {name: numeric_example(*values) for name, values in prepared.items()}
    case, native, displayed = prepared['game10']
    save(flow(), figures/FILENAMES[0])

    fig = figure((14., 5.1))
    variance, curves = fig.subplots(1, 2)
    _hierarchical_variance(variance, native)
    _hierarchical_curves(curves, native, 'lb', noise_band=True)
    for side, player in native['players'].items():
        curves.axhline(player['average_accuracy'], ls=':', color=COLORS[side], lw=1,
                       label=f'{side} observed: {player["average_accuracy"]:.2f}%')
    curves.set(xlim=DISPLAY_RATING_RANGE, ylim=(50., 100.), xlabel='Rating (native Lichess Blitz)', ylabel='Arithmetic accuracy (%)')
    curves.set_title('Worked game: retain local difficulty with uncertainty', fontsize=11)
    curves.grid(alpha=.2)
    curves.spines[['top', 'right']].set_visible(False)
    curves.legend(fontsize=8, loc='lower left')
    save(fig, figures/FILENAMES[1])

    fig = figure((8., 5.4))
    prior = fig.subplots()
    _hierarchical_prior(prior, native, 'lb')
    prior.set_xlabel('Rating (native Lichess Blitz)')
    save(fig, figures/FILENAMES[2])

    fig = figure((14., 5.4))
    for axis, (name, (_, fit, _)) in zip(fig.subplots(1, 2), prepared.items(), strict=True):
        _hierarchical_mapping(axis, fit, 'lb')
        axis.set_ylabel('Estimated rating (native Lichess Blitz)')
        label = 'Worked game' if name == 'game10' else 'Second game: observations beyond the native curve'
        axis.set_title(label+'\n'+axis.get_title(), fontsize=10)
    save(fig, figures/FILENAMES[3])

    exported = export_figures(displayed, output/'game10', title='Worked game · hierarchical affine estimation')
    staged = (figures/FILENAMES[4]).with_suffix('.svg.tmp')
    shutil.copyfile(exported['analysis_svg'], staged)
    staged.replace(figures/FILENAMES[4])
    export_figures(prepared['game15'][2], output/'game15', title='Second game · hierarchical affine estimation')
    write_json(figures/'examples.json', {'method': 'hierarchical_affine', 'reference_labels_used': False,
                                       'examples': examples, 'figures': list(FILENAMES)})
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = render(figures=args.figures, output=args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
