"""Render the paper's fixed numerical example without searching or fitting labels."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from analysis.player_rating.bayesian_shared_curve import Args, Rating
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.figures import export_figures


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/data/bayesian_shared_curve_game10.json'
OUTPUT = ROOT / 'tests/analysis/output/bayesian-shared-curve-paper'
FIGURES = ROOT / 'docs/figures'
PAPER_ARGS = Args(rating_range=(0, 3200), prior_range=(200, 3000), flat_prior_range=(800, 2400),
                  central_interval=.20, accuracy_sigma_scale=1.,
                  top_probability=1.)


def render():
    """Regenerate both paper SVGs, then remove superseded generated figures."""
    evidence = validate_evidence(json.loads(EVIDENCE.read_text(encoding='utf-8')))
    fit = Rating(args=PAPER_ARGS).fit(evidence)
    export_figures(fit, OUTPUT, title='Bayesian shared-curve estimation: worked example')
    FIGURES.mkdir(parents=True, exist_ok=True)
    replacements = {
        'analysis.svg': 'bayesian_shared_curve_game10.svg',
        'prior.svg': 'bayesian_shared_curve_game10_prior.svg',
    }
    for source in replacements:
        if not (OUTPUT / source).is_file():
            raise FileNotFoundError(f'The rating renderer did not create {source}.')
    for source, target in replacements.items():
        destination = FIGURES / target
        staged = destination.with_suffix('.svg.tmp')
        shutil.copyfile(OUTPUT / source, staged)
        staged.replace(destination)
    # Exact generated filenames only: preserve the paper, data and user artifacts.
    for name in ('bayesian_shared_curve_game10.png',
                 'bayesian_shared_curve_game10_posterior.svg',
                 'bayesian_shared_curve_game10_posterior.png',
                 'bayesian_shared_curve_game10_prior.png'):
        (FIGURES / name).unlink(missing_ok=True)
    return fit


if __name__ == '__main__':
    result = render()
    print(json.dumps({
        'players': {side: {key: values[key] for key in ('estimate', 'interval', 'average_accuracy')}
                    for side, values in result['players'].items()},
        'figures': [str(FIGURES / 'bayesian_shared_curve_game10.svg'),
                    str(FIGURES / 'bayesian_shared_curve_game10_prior.svg')],
    }, indent=2))
