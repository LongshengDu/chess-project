"""Current-game shared accuracy curves and conditional measurement variance."""
from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression

from analysis.player_rating.bayesian_shared_curve import Args as CurveArgs, SharedCurve, summarize


# A local copy keeps this method independent of future changes to the other
# estimator's selected arguments while reusing its mathematical primitives.
ARGS = CurveArgs()


def measure(evidence, competitive_power=0.):
    """Return one shared curve and each side's conditional mean variance.

    Competitive weights use only before-move winning chances. The same weights
    measure played accuracy and every Maia rating's expected accuracy. Legally
    forced positions are excluded. If all competitive weights vanish, use the
    arithmetic measurement explicitly instead of dividing by zero.
    """
    if not np.isfinite(competitive_power) or competitive_power < 0:
        raise ValueError('The competitiveness exponent must be finite and nonnegative.')
    baseline = summarize(evidence, args=ARGS)
    curve = baseline['diagnostics']['curve']
    sides = {}
    for side in ('White', 'Black'):
        rows = [row for row in evidence[side]['observations'] if len(row['qualities']['position']) > 1]
        if not rows:
            sides[side] = None
            continue
        means, variances, observed, weights = [], [], [], []
        for row in rows:
            q = np.asarray(row['qualities']['position'], dtype=float)
            policy = np.asarray(row['maia_probabilities'], dtype=float)
            mean = policy @ q
            means.append(mean)
            variances.append(np.maximum(0., policy @ (q*q)-mean*mean))
            observed.append(q[row['played_index']])
            weight = 1.
            if competitive_power:
                probability = row.get('position_win_probability')
                if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                        or not np.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError('Competitive measurement requires before-position winning probability in [0, 1].')
                weight = (4*probability*(1-probability))**competitive_power
            weights.append(weight)
        weights = np.asarray(weights, dtype=float)
        fallback = not bool(weights.sum())
        if fallback:
            weights[:] = 1.
        weights /= weights.sum()
        sides[side] = {'mean': weights @ means, 'variance': weights**2 @ variances,
                       'accuracy': float(np.clip(weights @ observed, 0., 100.)) if competitive_power else
                                   baseline['players'][side]['average_accuracy'],
                       'moves': len(rows), 'effective_moves': float(1./np.sum(weights**2)),
                       'zero_weight_fallback': fallback}
    available = [moment for moment in sides.values() if moment is not None]
    if competitive_power and available:
        expected = np.mean([moment['mean'] for moment in available], axis=0)
        monotone = np.clip(isotonic_regression(expected).x, 0., 100.)
        variance = float(np.mean([moment['variance'].mean() for moment in available]))
        curve.update(maia_expected_accuracy=expected.tolist(), monotone_expected_accuracy=monotone.tolist(),
                     shared_accuracy=SharedCurve(monotone)(ARGS.grid).tolist(), shared_mean_variance=variance)
        curve['likelihood'].update(accuracy_variance=variance, accuracy_sigma=float(np.sqrt(variance)))
    # Measurement diagnostics do not define the consuming estimator's posterior.
    curve.pop('posterior_densities', None)
    return {'curve': curve, 'sides': sides}
