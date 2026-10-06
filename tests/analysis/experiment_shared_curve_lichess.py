"""Compare arithmetic and probability-weighted Lichess shared curves offline.

Read saved game analysis and its existing Maia evidence without changing either.
All fits, comparisons and SVG figures stay in tests/analysis/output; PGN reference
ratings enter only after inference. No engines or coaching APIs are called.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import statistics
from time import perf_counter

import chess
import numpy as np
from scipy.integrate import trapezoid

from analysis.cache import write_json
from analysis.game.study import load_game
from analysis.lichess_accuracy import INITIAL_CP, game_accuracy, win_percent
from analysis.player_rating.bayesian_shared_curve import summarize as arithmetic_summary
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.figures import (
    COLORS, DISPLAY_RATING_RANGE, PANEL_SIZE, _accuracy_intersection, _accuracy_settings,
)
from analysis.player_rating.service import evidence_cache_path
from analysis.position_evaluation import centipawns
from analysis.settings import CONFIG
from tests.analysis.compare_shared_curve_games import audit_evidence, metrics, read_json
from tests.analysis.shared_curve_lichess import ARGS, summarize

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / 'tests/analysis/output'
DEFAULT_OUTPUT = OUTPUT_ROOT / 'shared-curve-lichess-top99-current-prior'


def _digest(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def _source_paths():
    paths = list(ROOT.glob('*.py')) + [ROOT / 'config.yaml']
    for component in ('analysis', 'backend', 'coach', 'engine', 'web'):
        paths.extend((ROOT / component).rglob('*.py'))
    return paths


def _output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()) or path == OUTPUT_ROOT.resolve():
        raise ValueError('Experiment outputs must be a subdirectory of tests/analysis/output.')
    return path


def _observed_accuracy(game, analysis, evidence):
    """Independently reconstruct Lichess accuracy and verify stored weights."""
    board = game.board()
    start_white = board.turn
    rows = analysis['moves']
    initial = INITIAL_CP if board.board_fen() == chess.Board().board_fen() else centipawns(rows[0]['position_eval'])
    scores = []
    for index, row in enumerate(rows):
        score = centipawns(rows[index + 1]['position_eval'] if index + 1 < len(rows) else row['played']['eval'])
        board.push_uci(row['played']['move'])
        if board.is_checkmate():
            score = '#-0' if board.turn else '#0'
        elif board.is_game_over(claim_draw=False):
            score = 0
        scores.append(score)
    wins = [win_percent(score) for score in [initial, *scores]]
    size = max(2, min(8, len(scores) // 10))
    windows = [wins[:size]] * (min(size, len(wins)) - 2)
    windows += [wins[index:index + size] for index in range(max(1, len(wins) - size + 1))]
    offsets = {'White': 0, 'Black': 0}
    for index, window in enumerate(windows):
        side = 'White' if (index % 2 == 0) == start_white else 'Black'
        expected = max(.5, min(12., statistics.pstdev(window)))
        actual = evidence[side]['observations'][offsets[side]]['weight']
        offsets[side] += 1
        if abs(expected - actual) > 1e-9:
            raise ValueError(f'Saved volatility weight differs at ply {index + 1}: {actual} vs {expected}.')
    return game_accuracy(scores, start_white, initial)


def export_figures(fit, output_dir, title):
    """Plot the experimental metric explicitly; never label it arithmetic."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'fit.json', fit)
    curve = fit['diagnostics']['curve']
    x = np.asarray(curve['fine_ratings'])
    y = np.asarray(curve['shared_accuracy'])
    measured = np.asarray(curve['rating_grid'])
    display_low, display_high = DISPLAY_RATING_RANGE

    def figure(size):
        fig = Figure(figsize=size, layout='constrained')
        FigureCanvasAgg(fig)
        return fig

    def decorate(axis, ylabel):
        axis.set(xlim=DISPLAY_RATING_RANGE, xlabel='Rating', ylabel=ylabel)
        axis.set_xticks(np.arange(display_low, display_high + 1, 200))
        axis.tick_params(axis='x', labelsize=8)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)

    def shade(axis):
        axis.axvspan(display_low, measured[0], color='#dce1e6', alpha=.55)
        axis.axvspan(measured[-1], display_high, color='#dce1e6', alpha=.55)

    fig = figure((2 * PANEL_SIZE[0], PANEL_SIZE[1]))
    fig.suptitle(title, fontsize=15, fontweight='bold')
    axis, posterior = fig.subplots(1, 2)
    shade(axis)
    axis.plot(x, y, color='#344154', lw=2, label='Shared expected Lichess accuracy')
    axis.scatter(measured, curve['monotone_expected_accuracy'], color='#344154', s=13,
                 label='Monotone Maia accuracy knots')
    visible_x = np.r_[display_low, x[(x > display_low) & (x < display_high)], display_high]
    visible_accuracy = np.interp(visible_x, x, y)
    for index, side in enumerate(('White', 'Black')):
        player = fit['players'][side]
        observed = player['average_accuracy']
        if observed is None:
            continue
        color = COLORS[side]
        axis.axhline(observed, color=color, ls='--', lw=1.4,
                     label=f'{side} played: {observed:.2f}%')
        crossing = _accuracy_intersection(visible_x, visible_accuracy, observed)
        label_y = .53 if index == 0 else .30
        if crossing is None:
            axis.text(.50, label_y, f'{side}: no intersection in {display_low:g}–{display_high:g}',
                      color=color, fontsize=9, ha='center', transform=axis.transAxes)
            continue
        start, end = crossing
        rating = (start + end) / 2
        value = f'{rating:,.0f}' if end - start < 1e-8 else f'{start:,.0f}–{end:,.0f}'
        label = f'{side} intersection: {value}\nAccuracy: {observed:.2f}%'
        if start < measured[0] or end > measured[-1]:
            label += '\nExtrapolated'
        axis.scatter([rating], [observed], color=color, s=38, zorder=5, edgecolor='white')
        axis.vlines(rating, 50, observed, color=color, ls=':', lw=1, alpha=.65)
        axis.annotate(label, (rating, observed),
                      xytext=(float(np.clip((rating - display_low) / (display_high - display_low), .20, .76)), label_y),
                      textcoords='axes fraction', ha='center', fontsize=9, color=color,
                      bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white', 'edgecolor': color, 'alpha': .95},
                      arrowprops={'arrowstyle': '-', 'color': color, 'lw': .9})
    axis.set_title('Probability-weighted Lichess accuracy · gray tails extrapolated\n' +
                   _accuracy_settings(fit, curve), fontsize=10)
    decorate(axis, 'Lichess game accuracy (%)')
    axis.set_ylim(50, 100)
    axis.legend(fontsize=8, loc='lower left', framealpha=.95)

    shade(posterior)
    for side in ('White', 'Black'):
        player = fit['players'][side]
        raw = curve['posterior_densities'][side]
        if raw is None:
            continue
        density = np.asarray(raw)
        low, high = player['interval']
        inner = np.r_[low, x[(x > low) & (x < high)], high]
        posterior.fill_between(inner, np.interp(inner, x, density), color=COLORS[side], alpha=.22)
        posterior.axvline(player['estimate'], color=COLORS[side], ls='--', lw=1.2)
        posterior.plot(x, density, color=COLORS[side], lw=2,
                       label=f'{side}: {player["estimate"]:,} [{low:,}–{high:,}] · {player["moves_used"]} moves')
    posterior.set_title(f'White and Black estimates · shaded central {fit["central_interval"]:.0%}', fontsize=11)
    decorate(posterior, 'Posterior density (per Elo)')
    posterior.set_ylim(0, max(posterior.get_ylim()[1], 1e-12) * 1.18)
    posterior.legend(fontsize=9, loc='best')
    fig.savefig(output_dir / 'analysis.svg', facecolor='white')
    fig.clear()

    fig = figure(PANEL_SIZE)
    axis = fig.subplots()
    prior = np.asarray(curve['prior_weights'])
    axis.fill_between(x, prior, color='#647b96', alpha=.16)
    axis.plot(x, prior, color='#647b96', lw=2)
    axis.set_title('Rating prior before normalization', fontsize=11)
    decorate(axis, 'Prior weight')
    axis.set_ylim(0, 1)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.yaxis.set_major_formatter('{x:.1f}')
    fig.savefig(output_dir / 'prior.svg', facecolor='white')
    fig.clear()
    for name in ('analysis.png', 'prior.png'):
        (output_dir / name).unlink(missing_ok=True)


def run(games_dir, output_dir=DEFAULT_OUTPUT, args=None):
    started = perf_counter()
    args = args or ARGS
    output_dir = _output_path(output_dir)
    games_dir = Path(games_dir)
    paths = sorted(games_dir.glob('game*.pgn'), key=lambda path: int(path.stem[4:]))
    if not paths:
        raise ValueError(f'No game PGNs in {games_dir}.')
    sources, saved = _source_paths(), []
    for pgn in paths:
        folder = pgn.parent / 'output' / f'{pgn.stem}-full'
        analysis_path = folder / 'analysis.json'
        analysis = read_json(analysis_path)
        cache = evidence_cache_path(CONFIG['ANALYSIS']['CACHE_DIR'], analysis['rating_fit']['evidence_key'])
        if not cache.is_file():
            raise FileNotFoundError(f'Existing rating evidence required: {cache}')
        sources.extend([pgn, analysis_path, cache, *list((folder / 'player-rating').glob('*'))])
        saved.append((pgn, analysis, cache))
    before = _digest(path for path in sources if path.is_file())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, games = [], []
    for pgn, analysis, cache in saved:
        game = load_game(pgn)
        evidence = validate_evidence(read_json(cache))
        audit = audit_evidence(game, analysis, evidence)
        baseline = arithmetic_summary(evidence, args=args)
        experimental = summarize(evidence, args=args)
        curve = experimental['diagnostics']['curve']
        actual = _observed_accuracy(game, analysis, evidence)
        if actual is None:
            raise ValueError(f'{pgn.name} has insufficient valid Lichess observations.')
        differences = {side: abs(experimental['players'][side]['average_accuracy'] - actual[side.lower()])
                       for side in ('White', 'Black')}
        if max(differences.values()) > 1e-9:
            raise ValueError(f'{pgn.name}: observed Lichess accuracy mismatch: {differences}')
        np.testing.assert_array_less(-1e-9, np.diff(curve['shared_accuracy']))
        if not 0 <= min(curve['shared_accuracy']) <= max(curve['shared_accuracy']) <= 100:
            raise ValueError(f'{pgn.name}: invalid bounded accuracy curve.')
        for density in curve['posterior_densities'].values():
            if density is not None:
                np.testing.assert_allclose(trapezoid(density, curve['fine_ratings']), 1., atol=1e-12)
        folder = output_dir / pgn.stem
        export_figures(experimental, folder, f'{pgn.stem} — experimental Lichess shared curve')
        write_json(folder / 'arithmetic-fit.json', baseline)
        game_rows = []
        # Commercial labels are evaluation only and are not passed to either fit.
        for side in ('White', 'Black'):
            player = experimental['players'][side]
            reference = game.headers.get(f'{side}EloEstimate')
            row = {'game': pgn.stem, 'side': side,
                   'reference': int(reference) if reference and reference.isdigit() else None,
                   'arithmetic_top99': baseline['players'][side]['estimate'],
                   'lichess_top99': player['estimate'],
                   'interval_low': player['interval'][0], 'interval_high': player['interval'][1],
                   'arithmetic_accuracy': baseline['players'][side]['average_accuracy'],
                   'lichess_accuracy': player['average_accuracy'], 'moves_used': player['moves_used'],
                   'accuracy_sigma': curve['likelihood']['accuracy_sigma']}
            rows.append(row)
            game_rows.append(row)
        games.append({'game': pgn.stem, 'evidence_path': str(cache.resolve()),
                      'evidence_sha256': before[str(cache.resolve())], 'evidence_audit': audit,
                      'saved_volatility_weights_match': True,
                      'observed_lichess_accuracy_maximum_difference': max(differences.values()),
                      'raw_curve_reversal': curve['largest_raw_curve_reversal'],
                      'isotonic_adjustment': curve['largest_isotonic_adjustment']})
        print(f'{pgn.stem}: arithmetic {game_rows[0]["arithmetic_top99"]} / {game_rows[1]["arithmetic_top99"]}; '
              f'Lichess {game_rows[0]["lichess_top99"]} / {game_rows[1]["lichess_top99"]}', flush=True)
    summary = {key: metrics(rows, key) for key in ('arithmetic_top99', 'lichess_top99')}
    for key in summary:
        pairs = [(rows[index], rows[index + 1]) for index in range(0, len(rows), 2)]
        comparable = [(w, b) for w, b in pairs if all(value is not None for value in
                      (w['reference'], b['reference'], w[key], b[key]))]
        summary[key]['ordering_matches'] = int(sum(np.sign(w[key] - b[key]) == np.sign(w['reference'] - b['reference'])
                                                   for w, b in comparable))
        summary[key]['comparable_games'] = len(comparable)
    after = _digest(Path(path) for path in before)
    if before != after:
        changed = [path for path in before if before[path] != after[path]]
        raise RuntimeError(f'Protected source files changed during this experiment: {changed}')
    result = {'parameters': experimental['parameters'], 'summary': summary, 'players': rows, 'games': games,
              'reference_usage': 'Evaluation only; no parameter fitting or selection using reference ratings.',
              'baseline': 'Arithmetic top-99% shared curve with the same current production prior and sigma scale.',
              'experiment': 'Observed Lichess game accuracy; normalized top-99% expected move accuracies and inverse accuracies; fixed actual-game volatility weights; delta-method variance.',
              'approximation': 'The harmonic component uses n / sum E[1/max(1,Q)], not E[n / sum 1/max(1,Q)]. Alternative full-game trajectories are not simulated.',
              'checks': {'source_files_unchanged': True, 'source_file_count': len(before),
                         'all_observed_accuracies_match_local_lichess': True},
              'elapsed_seconds': round(perf_counter() - started, 3)}
    write_json(output_dir / 'source-hashes.json', {'before': before, 'after': after, 'unchanged': True})
    write_json(output_dir / 'comparison.json', result)
    with (output_dir / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(games)} games to {output_dir}. Source hashes unchanged.', flush=True)
    print(summary, flush=True)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--games-dir', type=Path, default=ROOT / 'games')
    cli.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT,
                     help='Must be beneath tests/analysis/output; production outputs are protected.')
    supplied = cli.parse_args()
    run(supplied.games_dir, supplied.output_dir)


if __name__ == '__main__':
    main()
