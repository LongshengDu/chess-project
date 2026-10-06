"""Rank a fixed arithmetic shared-curve grid on games with measured intersections.

Both players must intersect the measured 600–2600 curve. The primary ranking
uses the same games for every variant; a secondary ranking uses each variant's
own qualifying games. Production code, caches and game outputs are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import trapezoid

from analysis.cache import write_json
from analysis.game.study import load_game
from analysis.player_rating.bayesian_shared_curve import Args, summarize
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.figures import export_figures
from analysis.player_rating.service import evidence_cache_path
from analysis.settings import CONFIG
from tests.analysis.compare_shared_curve_games import audit_evidence, read_json
from tests.analysis.experiment_shared_curve_lichess import _digest, _output_path, _source_paths

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/shared-curve-arithmetic-sweep-current-prior'
TOP_PROBABILITIES = (.95, .96, .97, .98, .99, 1.)
SIGMA_SCALES = (.5, 1.)


def build_variants(top_probabilities=TOP_PROBABILITIES, sigma_scales=SIGMA_SCALES):
    """Validate an explicit grid while retaining the original experiment defaults."""
    probabilities, scales = tuple(top_probabilities), tuple(sigma_scales)
    if not probabilities or not scales:
        raise ValueError('Both parameter grids must be nonempty.')
    if len(set(probabilities)) != len(probabilities) or len(set(scales)) != len(scales):
        raise ValueError('Parameter grids must not contain duplicates.')
    variants = {f'top-{probability*100:03g}-sigma-{scale:g}':
                replace(Args(), top_probability=probability, accuracy_sigma_scale=scale)
                for probability in probabilities for scale in scales}
    if len(variants) != len(probabilities)*len(scales):
        raise ValueError('Parameter values must have distinct display names.')
    return variants


def eligibility(fit):
    """Keep a whole game only if both estimates have an in-range intersection."""
    curve = fit['diagnostics']['curve']
    low, high = map(float, (curve['monotone_expected_accuracy'][0],
                           curve['monotone_expected_accuracy'][-1]))
    sides = {}
    for side in ('White', 'Black'):
        player = fit['players'][side]
        accuracy = player['average_accuracy']
        if accuracy is None or player['estimate'] is None or not np.isfinite(accuracy):
            position = 'unavailable'
        elif accuracy < low-1e-10:
            position = 'below'
        elif accuracy > high+1e-10:
            position = 'above'
        else:
            position = 'inside'
        sides[side] = {'position': position, 'average_accuracy': accuracy,
                       'curve_low': low, 'curve_high': high}
    return {'included': all(side['position'] == 'inside' for side in sides.values()),
            'sides': sides}


def _game_sort(name):
    return int(name.removeprefix('game'))


def rank_variants(rows, game_sets):
    """Evaluate all eligible players; the common cohort is label-independent."""
    common = set.intersection(*(set(games) for games in game_sets.values())) if game_sets else set()

    def ranking(common_cohort):
        results = []
        for variant, eligible_games in game_sets.items():
            candidates = [row for row in rows if row['variant'] == variant]
            games = common if common_cohort else set(eligible_games)
            selected = [row for row in candidates if row['game'] in games]
            if any(row['reference'] is None or row['estimate'] is None for row in selected):
                raise ValueError('Comparison requires reference and estimated ratings for every included player.')
            errors = np.asarray([round(row['estimate'])-row['reference'] for row in selected], dtype=float)
            pairs = {game: {row['side']: row for row in selected if row['game'] == game} for game in games}
            if any(set(pair) != {'White', 'Black'} for pair in pairs.values()):
                raise ValueError('A qualifying game must include both players.')
            matches = sum(np.sign(round(pair['White']['estimate'])-round(pair['Black']['estimate'])) ==
                          np.sign(pair['White']['reference']-pair['Black']['reference'])
                          for pair in pairs.values())
            first = candidates[0]
            results.append({
                'variant': variant, 'top_probability': first['top_probability'],
                'sigma_scale': first['sigma_scale'], 'games': len(games), 'players': len(selected),
                'mean_absolute_error': float(np.mean(abs(errors))) if len(errors) else None,
                'maximum_absolute_error': float(np.max(abs(errors))) if len(errors) else None,
                'root_mean_square_error': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                'mean_signed_error': float(np.mean(errors)) if len(errors) else None,
                'ordering_matches': int(matches), 'included_games': sorted(games, key=_game_sort),
            })
        metrics = ('mean_absolute_error', 'root_mean_square_error', 'maximum_absolute_error')
        results.sort(key=lambda item: tuple(item[key] if item[key] is not None else float('inf')
                                            for key in metrics)+(item['variant'],))
        return [dict(rank=index, **item) for index, item in enumerate(results, 1)]

    return {'common_games': sorted(common, key=_game_sort),
            'common_ranking': ranking(True), 'individual_ranking': ranking(False)}


def _write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_fit(job):
    fit, folder, title = job
    export_figures(fit, folder, title=title)
    return str(folder)


def _ranking_figure(result, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    probabilities = sorted({row['top_probability'] for row in result['common_ranking']})
    scales = sorted({row['sigma_scale'] for row in result['common_ranking']})
    values = [row['mean_absolute_error'] for key in ('common_ranking', 'individual_ranking')
              for row in result[key] if row['mean_absolute_error'] is not None]
    limits = {'vmin': min(values), 'vmax': max(values)} if values else {}
    fig = Figure(figsize=(12, max(4.5, len(scales)*.7+1.8)), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for axis, key, title in zip(axes, ('common_ranking', 'individual_ranking'),
                                ('Same qualifying games for every setting', 'Each setting’s qualifying games'), strict=True):
        rows = result[key]
        matrix = np.asarray([[next(item['mean_absolute_error'] for item in rows
                                  if item['top_probability'] == probability and item['sigma_scale'] == scale)
                              for probability in probabilities] for scale in scales], dtype=float)
        chart = axis.imshow(matrix, cmap='YlGnBu', aspect='auto', **limits)
        for y, scale in enumerate(scales):
            for x, probability in enumerate(probabilities):
                item = next(row for row in rows if row['top_probability'] == probability and row['sigma_scale'] == scale)
                text = f'{matrix[y,x]:.1f} Elo\n{item["games"]} games'
                color = 'white' if np.isfinite(matrix[y,x]) and chart.norm(matrix[y,x]) > .6 else '#182333'
                axis.text(x, y, text, ha='center', va='center', color=color, fontsize=9)
        axis.set_xticks(range(len(probabilities)), [f'{probability*100:g}%' for probability in probabilities])
        axis.set_yticks(range(len(scales)), [f'{scale:g}' for scale in scales])
        axis.set(xlabel='Retained Maia probability', ylabel='Sigma scale', title=title)
        fig.colorbar(chart, ax=axis, label='Mean absolute error (Elo)', shrink=.85)
    fig.suptitle('Arithmetic shared-curve parameter comparison · lower error is better', fontsize=14)
    fig.savefig(output/'ranking.svg', facecolor='white')
    (output/'ranking.png').unlink(missing_ok=True)
    fig.clear()


def run(games_dir, output=OUTPUT, *, top_probabilities=TOP_PROBABILITIES, sigma_scales=SIGMA_SCALES):
    started = perf_counter()
    output = _output_path(output)
    variants = build_variants(top_probabilities, sigma_scales)
    paths = sorted(Path(games_dir).glob('game*.pgn'), key=lambda path: _game_sort(path.stem))
    if not paths:
        raise ValueError('No saved game PGNs found.')
    sources, saved = _source_paths(), []
    for pgn in paths:
        folder = pgn.parent/'output'/f'{pgn.stem}-full'
        analysis_path = folder/'analysis.json'
        analysis = read_json(analysis_path)
        cache = evidence_cache_path(CONFIG['ANALYSIS']['CACHE_DIR'], analysis['rating_fit']['evidence_key'])
        sources.extend([pgn, analysis_path, cache, *(folder/'player-rating').glob('*')])
        saved.append((pgn, analysis, cache))
    before = _digest(path for path in sources if path.is_file())
    rows, included, fits, audits, exclusions = [], {key: set() for key in variants}, [], {}, []
    for pgn, analysis, cache in saved:
        game = load_game(pgn)
        evidence = validate_evidence(read_json(cache))
        audits[pgn.stem] = audit_evidence(game, analysis, evidence)
        observed, by_probability = None, {}
        for variant, args in variants.items():
            fit = summarize(evidence, args=args)
            qualifies = eligibility(fit)
            if args.top_probability in by_probability:
                if qualifies != by_probability[args.top_probability]:
                    raise AssertionError('Sigma multiplier must not change intersection eligibility.')
            by_probability[args.top_probability] = qualifies
            accuracies = tuple(fit['players'][side]['average_accuracy'] for side in ('White', 'Black'))
            if observed is not None and accuracies != observed:
                raise AssertionError('Candidate cutoffs must not change actual played accuracy.')
            observed = accuracies
            curve = fit['diagnostics']['curve']
            if np.any(np.diff(curve['shared_accuracy']) < -1e-10):
                raise AssertionError('Shared curve must remain monotone.')
            for density in curve['posterior_densities'].values():
                if density is not None:
                    np.testing.assert_allclose(trapezoid(density, curve['fine_ratings']), 1., atol=1e-12)
            if qualifies['included']:
                included[variant].add(pgn.stem)
            exclusions.append({'variant': variant, 'game': pgn.stem, 'included': qualifies['included'],
                               'White': qualifies['sides']['White'], 'Black': qualifies['sides']['Black']})
            # Labels enter only after inference and the intersection decision.
            for side in ('White', 'Black'):
                reference = game.headers.get(f'{side}EloEstimate')
                player = fit['players'][side]
                rows.append({'variant': variant, 'game': pgn.stem, 'side': side,
                             'top_probability': args.top_probability, 'sigma_scale': args.accuracy_sigma_scale,
                             'reference': int(reference) if reference and reference.isdigit() else None,
                             'estimate': player['estimate'], 'included': qualifies['included'],
                             'average_accuracy': player['average_accuracy'],
                             'curve_low': qualifies['sides'][side]['curve_low'],
                             'curve_high': qualifies['sides'][side]['curve_high'],
                             'intersection': qualifies['sides'][side]['position'],
                             'accuracy_sigma': curve['likelihood']['accuracy_sigma']})
            fits.append((fit, output/variant/pgn.stem, f'{pgn.stem} — top {args.top_probability:.0%}, sigma scale {args.accuracy_sigma_scale:g}'))
        print(f'{pgn.stem}: {len(variants)} variants fitted; observed accuracy and sigma-independent selection verified.', flush=True)
    result = rank_variants(rows, included)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output/'ranking-common.csv', result['common_ranking'])
    _write_csv(output/'ranking-individual.csv', result['individual_ranking'])
    _write_csv(output/'players.csv', rows)
    _ranking_figure(result, output)
    print(f'Common cohort: {result["common_games"]}; rendering {len(fits)} fit/curve/prior sets.', flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        for count, _ in enumerate(executor.map(_render_fit, fits), 1):
            if count % 24 == 0 or count == len(fits):
                print(f'Figures completed: {count}/{len(fits)}.', flush=True)
    after = _digest(Path(path) for path in before)
    if before != after:
        raise RuntimeError('A protected production file or saved input changed during the experiment.')
    result.update({
        'parameters': {key: asdict(args) for key, args in variants.items()},
        'intersection_rule': 'Both players must intersect the monotone measured 600–2600 shared accuracy curve; inclusive endpoints; otherwise exclude the entire game.',
        'ranking_rule': f'Primary: same games eligible under all {len(variants)} variants. Secondary: each variant own eligible games. MAE of rounded posterior medians; ties by RMSE, maximum error and variant name.',
        'reference_usage': 'Commercial labels evaluate and rank this requested fixed grid only; they do not enter inference or determine exclusions. This is benchmark selection, not independent validation.',
        'players': rows, 'eligibility': exclusions, 'evidence_audits': audits,
        'checks': {'source_files_unchanged': True, 'source_file_count': len(before),
                   'sigma_does_not_change_eligibility': True, 'observed_accuracy_identical_across_variants': True,
                   'posterior_normalization_verified': True, 'fit_and_figure_sets': len(fits)},
        'elapsed_seconds': round(perf_counter()-started, 3),
    })
    write_json(output/'comparison.json', result)
    write_json(output/'source-hashes.json', {'before': before, 'after': after, 'unchanged': True})
    print(f'Finished in {result["elapsed_seconds"]:.1f}s. Best common-cohort setting: {result["common_ranking"][0]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-dir', type=Path, default=ROOT/'games')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--top-probabilities', type=float, nargs='+', default=TOP_PROBABILITIES,
                        help='Retained probabilities in (0, 1]; e.g. 0.99 1.0.')
    parser.add_argument('--sigma-scales', type=float, nargs='+', default=SIGMA_SCALES,
                        help='Positive accuracy-sigma multipliers to compare.')
    args = parser.parse_args()
    run(args.games_dir, args.output_dir, top_probabilities=args.top_probabilities, sigma_scales=args.sigma_scales)


if __name__ == '__main__':
    main()
