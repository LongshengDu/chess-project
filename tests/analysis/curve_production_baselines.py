"""Reuse production fits once, then apply their exact account decisions offline."""
from analysis.player_rating.calibration import load_calibration


def describe():
    return {'arithmetic_coverage': 'Retained arithmetic-coverage production method, version1.',
            'uncertainty_ensemble': 'Unchanged arithmetic-only uncertainty ensemble, version2.'}


def prepare(evidence, actuals=None):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from analysis.player_rating import arithmetic_coverage, uncertainty_ensemble
    calibration = load_calibration()
    coverage = arithmetic_coverage.calculate(evidence, {}, calibration)
    ensemble = uncertainty_ensemble.calculate(evidence, {}, calibration)
    sides = ('White', 'Black')
    return {'coverage': {s: coverage['diagnostics']['components'][s]['coverage_mean'] for s in sides},
            'ensemble': {s: ensemble['diagnostics']['components'][s].get('quality') for s in sides},
            'accuracy': {s: coverage['players'][s]['average_accuracy'] for s in sides}}


def predict_prepared(prepared, actuals):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    from analysis.player_rating import arithmetic_coverage, uncertainty_ensemble
    anchor, _ = arithmetic_coverage._account_anchor(actuals)
    weight = arithmetic_coverage.ARGS.account_weight if anchor is not None else 0.
    coverage = {side: ((1-weight)*value+weight*anchor if anchor is not None else value) if value is not None else None
                for side, value in prepared['coverage'].items()}
    ensemble, _ = uncertainty_ensemble.account_decision(prepared['ensemble'], prepared['accuracy'], actuals,
                                                        uncertainty_ensemble.ARGS.account_weight)
    return {'arithmetic_coverage': coverage, 'uncertainty_ensemble': ensemble}


def predict(evidence, actuals):
    return predict_prepared(prepare(evidence), actuals)
