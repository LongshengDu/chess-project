"""Evaluate a fixed top-probability candidate rule on every saved game.

References are evaluation labels only. Production analysis, fits and settings
remain untouched; experiments and SVG figures are written below tests/analysis.
The runner uses the current production prior; earlier saved Gaussian-prior
experiments retain their original parameters in their output artifacts.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from analysis.cache import write_json
from analysis.game.study import load_game
from analysis.player_rating.bayesian_shared_curve import SharedCurve
from analysis.player_rating.figures import export_figures
from analysis.player_rating.service import evidence_cache_path
from analysis.settings import CONFIG
from tests.analysis.compare_shared_curve_games import audit_evidence, metrics, read_json
from tests.analysis.shared_curve_top_probability import Args, TopProbabilityRating, prior_density, prior_weights

ROOT = Path(__file__).resolve().parents[2]


def check_resolution(fit, args):
    """Independently integrate at one Elo to check the five-Elo output grid."""
    curve = fit['diagnostics']['curve']
    grid = np.arange(args.rating_range[0], args.rating_range[1]+1.)
    accuracy = SharedCurve(curve['monotone_expected_accuracy'])(grid)
    prior = prior_density(grid, args=args)
    interior = prior > 0
    variance = curve['likelihood']['accuracy_variance']
    error = 0.
    for player in fit['players'].values():
        if player['estimate'] is None:
            continue
        logs = -.5*(player['average_accuracy']-accuracy)**2/max(variance, 1e-12)
        logs[interior] += np.log(prior[interior])
        density = np.zeros_like(grid)
        density[interior] = np.exp(logs[interior]-logs[interior].max())
        cdf = cumulative_trapezoid(density, grid, initial=0.)
        cdf /= cdf[-1]
        median = np.interp(.5, cdf, grid)
        error = max(error, abs(float(median)-player['unrounded_estimate']))
    if error >= .1:
        raise ValueError(f'Posterior grid needs refinement: median changed by {error:.4f} Elo.')
    return error


def run(games_dir, output_dir, args):
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(games_dir.glob('game*.pgn'), key=lambda p: int(p.stem[4:]))
    if not paths:
        raise ValueError(f'No saved game PGNs in {games_dir}.')
    variants = {'full_probability': TopProbabilityRating(replace(args, top_probability=1.)),
                'top_probability': TopProbabilityRating(args)}
    rows, games, checks = [], [], {}
    for pgn in paths:
        game = load_game(pgn)
        analysis_path = pgn.parent/'output'/f'{pgn.stem}-full'/'analysis.json'
        original_hash = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
        analysis = read_json(analysis_path)
        cache = evidence_cache_path(CONFIG['ANALYSIS']['CACHE_DIR'], analysis['rating_fit']['evidence_key'])
        evidence = read_json(cache)
        audit = audit_evidence(game, analysis, evidence)
        fits = {key: estimator.fit(evidence) for key, estimator in variants.items()}
        grid_errors = {}
        for key, fit in fits.items():
            curve = fit['diagnostics']['curve']
            assert np.all(np.diff(curve['shared_accuracy']) >= -1e-10)
            assert min(curve['shared_accuracy']) >= 0 and max(curve['shared_accuracy']) <= 100
            for density in curve['posterior_densities'].values():
                if density is not None:
                    np.testing.assert_allclose(trapezoid(density, curve['fine_ratings']), 1., atol=1e-12)
            grid_errors[key] = check_resolution(fit, variants[key].args)
            if key == 'top_probability':
                export_figures(fit, output_dir/pgn.stem,
                               title=f'{pgn.stem} — top {args.top_probability:.0%} Maia probability')
            else:
                write_json(output_dir/pgn.stem/'full-probability-fit.json', fit)
        top, control = fits['top_probability'], fits['full_probability']
        curve = top['diagnostics']['curve']
        for side in ('White', 'Black'):
            player = top['players'][side]
            assert player['average_accuracy'] == control['players'][side]['average_accuracy']
            assert player['moves_used'] == control['players'][side]['moves_used']
            # Reference labels are read only after both inference calls finish.
            reference = game.headers.get(f'{side}EloEstimate')
            rows.append({'game': pgn.stem, 'side': side,
                         'reference': int(reference) if reference and reference.isdigit() else None,
                         'full_probability': control['players'][side]['estimate'],
                         'top_probability': player['estimate'], 'interval_low': player['interval'][0],
                         'interval_high': player['interval'][1], 'played_accuracy': player['average_accuracy'],
                         'moves_used': player['moves_used'],
                         'forced_positions_removed': curve['selection'][side]['forced_positions_removed'],
                         'accuracy_sigma': curve['likelihood']['accuracy_sigma']})
        assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == original_hash
        checks[pgn.stem] = {'saved_analysis_unchanged': True, 'evidence_audit': audit,
                           'maximum_5_vs_1_elo_median_difference': grid_errors}
        games.append({'game': pgn.stem, 'evidence_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                      'raw_curve_reversal': curve['largest_raw_curve_reversal'],
                      'isotonic_adjustment': curve['largest_isotonic_adjustment'],
                      'zero_variance_floor_used': curve['zero_variance_floor_used']})
        print(f"{pgn.stem}: {top['players']['White']['estimate']} / {top['players']['Black']['estimate']}", flush=True)
    summary = {key: metrics(rows, key) for key in variants}
    for key in variants:
        pairs = [(rows[i], rows[i+1]) for i in range(0, len(rows), 2)]
        comparable = [(w, b) for w, b in pairs if all(v is not None for v in (w['reference'], b['reference'], w[key], b[key]))]
        summary[key]['ordering_matches'] = sum(np.sign(w[key]-b[key]) == np.sign(w['reference']-b['reference']) for w, b in comparable)
        summary[key]['ordering_matches'] = int(summary[key]['ordering_matches'])
        summary[key]['comparable_games'] = len(comparable)
    relative_prior = prior_weights(np.array([200., 400., 600., 2600., 2800., 3000.]), args=args)
    result = {'parameters': variants['top_probability'].parameters,
              'experiment': 'Retain probability-ranked moves through cumulative mass strictly above X, including ties; renormalize; derive both expected accuracy and adaptive variance from retained moves.',
              'forced_positions': 'Only one-legal-move positions omitted, from both expected and observed averages.',
              'reference_usage': 'Evaluation only; no parameters fitted or selected using references.',
              'baseline': 'Full probability with the same sigma multiplier, current fourth-power prior and forced-position exclusion.',
              'relative_prior': dict(zip(('200', '400', '600', '2600', '2800', '3000'), relative_prior.tolist())),
              'players': rows, 'games': games, 'checks': checks, 'summary': summary,
              'elapsed_seconds': round(perf_counter()-started, 3)}
    write_json(output_dir/'comparison.json', result)
    with (output_dir/'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'summary': summary, 'elapsed_seconds': result['elapsed_seconds']}, indent=2))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--games-dir', type=Path, default=ROOT/'games')
    cli.add_argument('--output-dir', type=Path,
                     default=ROOT/'tests/analysis/output/shared-curve-top-probability-current-prior')
    cli.add_argument('--top-probability', type=float, default=.68)
    cli.add_argument('--sigma-scale', type=float, default=.5)
    cli.add_argument('--prior-range', type=float, nargs=2, default=(200., 3000.),
                     metavar=('LOW', 'HIGH'), help='Zero-weight endpoints of the fourth-power prior.')
    supplied = cli.parse_args()
    run(supplied.games_dir, supplied.output_dir,
        Args(top_probability=supplied.top_probability, accuracy_sigma_scale=supplied.sigma_scale,
             prior_range=tuple(supplied.prior_range)))


if __name__ == '__main__':
    main()
