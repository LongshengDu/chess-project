"""Declared coarse sensitivity of two universal opinion-consensus models.

After specifying the constituent methods, compare coverage weights 1/4, 1/2,
and 3/4. This is an explicitly reference-evaluated minor-assumption check, not
a coefficient optimizer or a game-specific rule. Keep the full ranking; a
selected value requires fresh validation, like other method choices here.
"""
from tests.analysis import universal_native_coverage, universal_competitiveness
from tests.analysis.universal_competitiveness import weighted_fit, project_order

COVERAGE_WEIGHTS = (.25, .5, .75)


def predict(evidence, fit, ratings, calibration_cases):
    ordinary = universal_native_coverage.predict(evidence, fit, ratings, calibration_cases)
    transformed = weighted_fit(evidence, .5)
    calibration = [{'fit': weighted_fit(c['evidence'], .5)} for c in calibration_cases]
    competitive = universal_native_coverage.predict(evidence, transformed, ratings, calibration)
    predictive = universal_competitiveness.predict(evidence, fit, ratings, calibration_cases)
    second = predictive['competitive_sqrt_mean_account']
    results = {}
    for measurement, outputs in (('arithmetic', ordinary), ('competitive', competitive)):
        first = outputs['native_coverage_mean_account_5pct_all']
        for weight in COVERAGE_WEIGHTS:
            pair = {side: weight*first[side]+(1-weight)*second[side] for side in ('White', 'Black')}
            results[f'consensus_{measurement}_coverage{int(100*weight):02d}'] = project_order(pair, fit)
    return results
