"""Fixed simple controls for estimating a shared game level and its contrast.

These are experiments, not production changes. A common rating center uses the
arithmetic shared-curve posterior. A smooth contrast multiplier compares the
observed accuracy gap with the model's conditional noise. This is a resolution
assumption, not a calibrated test of equal human strength.
"""
import numpy as np

from analysis.player_rating.bayesian_shared_curve import ARGS, fit_pair


def describe():
    return {
        'pair_pool_half': 'Arithmetic posterior means; fixed half contrast and 10% common account center.',
        'pair_pool_signal': 'Arithmetic posterior means; contrast gain dA^2/(dA^2+2*v), where v is pooled accuracy variance; 10% common account center.',
        'pair_pool_two_sigma': 'Same single-curve model, with contrast gain dA^2/(dA^2+8*v): two-standard-error resolution control.',
        'pair_pool_center_inverse': 'Inverse shared curve at mean observed accuracy as pair center, 10% common account center; posterior-mean contrast with two-standard-error resolution.',
    }


def predict(evidence, actual_ratings):
    fit = fit_pair(evidence['White'], evidence['Black'])
    if not fit['identifiable'] or any(p['estimate'] is None for p in fit['players']):
        return {name: {'White': None, 'Black': None} for name in describe()}
    x = np.asarray(fit['fine_ratings'])
    means = [float(np.trapezoid(x*np.asarray(fit['posterior_densities'][side]), x))
             for side in ('White', 'Black')]
    accuracies = [p['average_accuracy'] for p in fit['players']]
    difference = means[0]-means[1]
    gap = accuracies[0]-accuracies[1]
    variance = fit['shared_mean_variance']
    accounts = [r for r in actual_ratings.values() if r is not None]
    center = float(np.mean(means))
    inverse_center = float(np.interp(np.mean(accuracies), fit['shared_accuracy'], x))
    if accounts:
        center = .9*center+.1*np.mean(accounts)
        inverse_center = .9*inverse_center+.1*np.mean(accounts)
    gains = [.5, gap*gap/max(gap*gap+2*variance, 1e-12),
             gap*gap/max(gap*gap+8*variance, 1e-12)]
    result = {}
    for name, gain, middle in zip(describe(), (*gains, gains[-1]), (center, center, center, inverse_center), strict=True):
        # Do not manufacture a one-Elo separation when the decision shrinks to
        # a displayed tie. Scoring uses the displayed numbers honestly.
        contrast = difference*gain
        low, high = ARGS.rating_range
        middle = float(np.clip(middle, low+abs(contrast)/2, high-abs(contrast)/2))
        result[name] = {'White': middle+contrast/2, 'Black': middle-contrast/2}
    return result
