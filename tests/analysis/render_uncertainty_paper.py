"""Render the arithmetic uncertainty paper's fixed game10 worked example."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from analysis.player_rating.context import saved_context
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.figures import export_figures
from analysis.settings import CONFIG


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_KEY = '6219168948ad080b764d0e85bdb28c44f0c1ca983666ddbe6985dce244e35801'
ANALYSIS = ROOT/'games/output/game10-full/analysis.json'
OUTPUT = ROOT/'tests/analysis/output/uncertainty-shared-curve-paper'
FIGURE = ROOT/'docs/figures/uncertainty_ensemble_game10.svg'
ACTUAL_RATINGS = {'WhiteElo': '1405', 'BlackElo': '1390'}


def render():
    """Use fixed numeric evidence; account inputs are declared in the paper."""
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    # Kept as an unreachable historical record; these implementations were removed.
    from analysis.player_rating.uncertainty_ensemble import Rating
    evidence_path = Path(CONFIG['ANALYSIS']['CACHE_DIR'])/'player-rating'/f'rating-{EVIDENCE_KEY}.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    saved = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if saved['rating_fit']['evidence_key'] != EVIDENCE_KEY:
        raise ValueError('The worked example requires its original saved engine-evidence snapshot.')
    # Only before-position scores identify the frozen calibration context.
    # Saved account overrides and reference headers are not inputs.
    context = {'headers': ACTUAL_RATINGS, 'moves': saved['moves']}
    evidence = validate_evidence(saved_context(evidence, context))
    fit = Rating().fit(evidence)
    if fit['diagnostics']['calibration']['contexts_excluded'] != 1:
        raise ValueError('The worked example must match its original numeric calibration context.')
    export_figures(fit, OUTPUT, title='Uncertainty-weighted shared-curve estimation: worked example')
    generated = OUTPUT/'analysis.svg'
    if not generated.is_file():
        raise FileNotFoundError('The worked-example rating figure was not created.')
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    staged = FIGURE.with_suffix('.svg.tmp')
    shutil.copyfile(generated, staged)
    staged.replace(FIGURE)
    return fit


if __name__ == '__main__':
    result = render()
    print(json.dumps({'players': result['players'],
                      'components': result['diagnostics']['components'],
                      'calibration': result['diagnostics']['calibration'],
                      'figure': str(FIGURE)}, indent=2))
