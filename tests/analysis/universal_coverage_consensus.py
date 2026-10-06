"""Fixed equal-opinion combinations of coverage and predictive quality evidence.

Native-range compatibility and posterior model evidence express two different
reliability assumptions. Compare equal-point consensus, including competitiveness
as a second measurement, without fitting weights or selecting by game identity.
The original shared-curve color order is an explicit constrained decision.
"""
from tests.analysis import universal_native_coverage, universal_competitiveness, universal_posterior_decision
from tests.analysis.universal_competitiveness import project_order, weighted_fit


def predict(evidence, fit, ratings, calibration_cases):
    inputs = (evidence, fit, ratings, calibration_cases)
    arithmetic = universal_native_coverage.predict(*inputs)
    competitive_fit = weighted_fit(evidence, .5)
    competitive_calibration = [{'fit': weighted_fit(case['evidence'], .5)} for case in calibration_cases]
    competitive = universal_native_coverage.predict(evidence, competitive_fit, ratings, competitive_calibration)
    coverage = {**{'arithmetic_'+key: value for key, value in arithmetic.items()},
                **{'competitive_sqrt_'+key: value for key, value in competitive.items()}}
    competition = universal_competitiveness.predict(*inputs)
    ordinary = universal_posterior_decision.predict(*inputs)
    candidates = {
        'coverage_competitive_consensus': (
            coverage['arithmetic_native_coverage_mean_account_5pct_all'],
            competition['competitive_sqrt_mean_account']),
        'coverage_arithmetic_consensus': (
            coverage['arithmetic_native_coverage_mean_account_5pct_all'],
            ordinary['mixture_mean_account']),
        'competitive_coverage_consensus': (
            coverage['competitive_sqrt_native_coverage_mean_account_5pct_all'],
            competition['competitive_sqrt_mean_account'])}
    result = {name: project_order({side: .5*(first[side]+second[side])
                                  for side in ('White', 'Black')}, fit)
            for name, (first, second) in candidates.items()}
    # Cross the two measurement choices with the two reliability models; treat
    # every combination equally instead of selecting a component per game.
    factorial = (coverage['arithmetic_native_coverage_mean_account_5pct_all'],
                 coverage['competitive_sqrt_native_coverage_mean_account_5pct_all'],
                 competition['competitive_sqrt_mean_account'], ordinary['mixture_mean_account'])
    result['coverage_factorial_consensus'] = project_order(
        {side: sum(model[side] for model in factorial)/4 for side in ('White', 'Black')}, fit)
    return result
