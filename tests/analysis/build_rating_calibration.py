"""Build anonymous population curves from saved model evidence, never PGN labels.

Only legal-candidate qualities, Maia policies and before-position engine scores
enter the output. Played indices, actual ratings, observed mean accuracies and
reference estimates are not read by the moment builder or written to the asset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.lichess_accuracy import win_percent
from analysis.position_evaluation import centipawns
from analysis.player_rating.calibration import DEFAULT_PATH, SIDES, evidence_fingerprint, load_calibration
from analysis.player_rating.parameters import RATINGS
from analysis.player_rating.service import evidence_cache_path
from analysis.settings import CONFIG

ROOT = Path(__file__).resolve().parents[2]


def context_moments(evidence, *, competitive=False):
    """Compute shared expected moments without consulting any played move index."""
    sides = []
    for side in SIDES:
        means, variances, weights = [], [], []
        for row in evidence[side]['observations']:
            q = np.asarray(row['qualities']['position'], dtype=float)
            if len(q) == 1:
                continue
            p = np.asarray(row['maia_probabilities'], dtype=float)
            # Reproduce the existing full-policy arithmetic contract exactly.
            # Competitive moments already receive normalized evidence policies.
            if not competitive:
                p = p / p.sum(axis=1, keepdims=True)
            mean = p @ q
            means.append(mean)
            variances.append(np.maximum(0., p @ (q*q)-mean*mean))
            probability = row['position_win_probability']
            weights.append(np.sqrt(4*probability*(1-probability)) if competitive else 1.)
        if not means:
            continue
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            raise ValueError('Calibration requires positive total decision weight.')
        weights /= weights.sum()
        if competitive:
            expected = weights @ means
            variance = float(np.mean(weights**2 @ variances))
        else:
            expected = np.mean(means, axis=0)
            variance = float(np.mean(np.sum(variances, axis=0)/len(means)**2))
        sides.append((expected, variance))
    if not sides:
        raise ValueError('A calibration context must contain at least one non-forced decision.')
    knots = np.clip(isotonic_regression(np.mean([m[0] for m in sides], axis=0)).x, 0., 100.)
    return {'knots': knots.tolist(), 'variance': float(np.mean([m[1] for m in sides]))}


def load_numeric_context(analysis_path, cache_dir):
    """Extract the numeric whitelist; deliberately avoid PGN and fitted ratings."""
    analysis = json.loads(Path(analysis_path).read_text(encoding='utf-8'))
    cache = evidence_cache_path(cache_dir, analysis['rating_fit']['evidence_key'])
    saved = json.loads(cache.read_text(encoding='utf-8'))
    probabilities = {side: [] for side in SIDES}
    for move in analysis['moves']:
        probability = win_percent(centipawns(move['position_eval']))
        if probability is None:
            raise ValueError('A saved before-position evaluation is required.')
        probabilities[move['side'].title()].append(probability/100.)
    result = {}
    for side in SIDES:
        observations = saved[side]['observations']
        if len(observations) != len(probabilities[side]):
            raise ValueError('Saved positions and Maia observations must align.')
        result[side] = {'observations': [
            {'qualities': {'position': row['qualities']['position']},
             'maia_probabilities': row['maia_probabilities'], 'position_win_probability': probability}
            for row, probability in zip(observations, probabilities[side], strict=True)]}
    return result


def build(sources, output=DEFAULT_PATH, cache_dir=None):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    records = []
    for source in sources:
        evidence = load_numeric_context(source, cache_dir or CONFIG['ANALYSIS']['CACHE_DIR'])
        records.append({'fingerprint': evidence_fingerprint(evidence),
                        'arithmetic': context_moments(evidence),
                        'competitive': context_moments(evidence, competitive=True)})
    payload = {
        'schema_version': 1,
        'rating_grid': list(RATINGS),
        'provenance': {
            'purpose': 'Frozen model-implied accuracy moments over an exploratory paired-game position corpus.',
            'calibration_labels': 'none',
            'independent_human_validation': False,
            'context_count': len(records),
            'measurement': 'Full legal Maia policies and engine candidate accuracy; legally forced moves excluded.',
            'weighting': 'Equal side and equal game; competitive position weights sqrt(4*p*(1-p)).',
            'exclusion': 'Exact numeric target-context fingerprint removed before population calculation.',
            'limitations': 'Small selected, potentially dependent context corpus; frozen model moments are not external validation.'},
        'contexts': sorted(records, key=lambda record: record['fingerprint'])}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return load_calibration(output)


def main():
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-dir', type=Path, default=ROOT/'games')
    parser.add_argument('--output', type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    sources = sorted(args.games_dir.glob('output/game*-full/analysis.json'))
    if not sources:
        parser.error('Saved full-game analyses are required.')
    corpus = build(sources, args.output)
    print(json.dumps({'contexts': len(corpus.records), 'sha256': corpus.content_hash, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
