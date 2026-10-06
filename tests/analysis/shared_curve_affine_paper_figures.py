"""Draw the affine paper's declared synthetic example; never read game evidence.

The analytical inputs are illustrative, not measured Maia calibration data.
Only SVG publication figures are written; no estimator asset is created.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.player_rating.bayesian_shared_curve import GRID, SharedCurve
from analysis.player_rating.shared_curve_affine import CURVE_ARGS, affine_moments, translated_prior


OUTPUT = Path(__file__).resolve().parents[2] / 'docs' / 'figures' / 'shared-curve-affine'
BLUE, ORANGE, GREEN = '#2563a5', '#cc6b28', '#25836e'


def style(axis, xlabel, ylabel):
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.16)


def save(figure, name):
    figure.savefig(OUTPUT / name, format='svg', metadata={'Date': None})
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.titlesize': 12})
    grid = CURVE_ARGS.grid
    knots = 70 + .01 * GRID
    curve = SharedCurve(knots)(grid)
    prior = translated_prior(1600.)
    moments = affine_moments(curve, 4., prior)
    slope = moments['affine_slope']
    accuracy = {'White': 90., 'Black': 84.}
    points = {side: prior['prior_mean'] + slope * (value - moments['accuracy_mean'])
              for side, value in accuracy.items()}

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    ax = axes[0]
    ax.plot(grid, curve, color=GREEN, lw=2, label='Illustrative shared curve')
    ax.scatter(GRID, knots, s=12, color=GREEN, zorder=3, label='Declared knots')
    for lo, hi in ((200, 600), (2600, 3000)):
        ax.axvspan(lo, hi, color='#aab0b9', alpha=.16)
    for side, color, dy in (('White', BLUE, 1), ('Black', ORANGE, -2.2)):
        value = accuracy[side]
        crossing = (value - 70) / .01
        ax.axhline(value, color=color, ls=':', lw=1.5)
        ax.plot(crossing, value, 'o', color=color)
        ax.annotate(f'{side}: {value:.0f}% / crossing {crossing:.0f}',
                    (crossing, value), xytext=(950, value+dy) if side == 'White' else (1680, 80), color=color,
                    arrowprops={'arrowstyle': '-', 'color': color})
    ax.set(xlim=(200, 3000), ylim=(65, 100), title='A. Accuracy evidence')
    ax.set_xticks(np.arange(200, 3001, 400))
    style(ax, 'Native rating', 'Arithmetic move accuracy (%)')
    ax.legend(loc='lower right', frameon=False, fontsize=9)

    ax = axes[1]
    values = np.linspace(72, 100, 400)
    predictions = prior['prior_mean'] + slope * (values-moments['accuracy_mean'])
    ax.plot(values, predictions, color=GREEN, lw=2, label='Shared affine decision')
    ax.plot(knots, GRID, color='#777777', ls='--', label='Direct knot intersection')
    ax.scatter(moments['accuracy_mean'], prior['prior_mean'], color='black', s=28, zorder=4)
    ax.annotate('Prior-moment center', (moments['accuracy_mean'], prior['prior_mean']),
                xytext=(74, 1750), arrowprops={'arrowstyle': '-', 'color': '#333333'}, fontsize=9)
    for side, color in (('White', BLUE), ('Black', ORANGE)):
        value, estimate = accuracy[side], points[side]
        ax.plot(value, estimate, 'o', color=color)
        ax.annotate(f'{side}: {estimate:.1f}', (value, estimate), xytext=(8, -18 if side == 'Black' else 8),
                    textcoords='offset points', color=color)
    ax.set(xlim=(72, 100), ylim=(200, 3000), title='B. One accuracy-to-rating rule')
    style(ax, 'Observed arithmetic accuracy (%)', 'Estimated native rating')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    save(figure, 'shared-curve-and-decision.svg')

    figure, ax = plt.subplots(figsize=(6, 3.8), layout='constrained')
    for anchor, color in ((1200., ORANGE), (1600., GREEN), (2000., BLUE)):
        value = translated_prior(anchor)
        ax.plot(grid, value['prior_weights'], color=color, lw=2, label=f'Common account anchor {anchor:.0f}')
    ax.set(xlim=(200, 3000), ylim=(0, 1.04), title='Common prior translation')
    ax.set_xticks(np.arange(200, 3001, 400))
    style(ax, 'Native rating', 'Unnormalized prior weight')
    ax.legend(loc='lower center', frameon=False, fontsize=9)
    save(figure, 'translated-prior.svg')

    figure, ax = plt.subplots(figsize=(6, 3.8), layout='constrained')
    for variance, color in ((0., '#777777'), (4., GREEN), (16., ORANGE)):
        changed = affine_moments(curve, variance, prior)
        action = prior['prior_mean'] + changed['affine_slope'] * (values-changed['accuracy_mean'])
        ax.plot(values, action, color=color, lw=2, label=f'Conditional variance {variance:.0f} pp²; slope {changed["affine_slope"]:.1f}')
    ax.scatter(moments['accuracy_mean'], prior['prior_mean'], color='black', s=26, zorder=3)
    ax.set(xlim=(76, 98), ylim=(200, 3000), title='Larger conditional variance reduces the slope')
    style(ax, 'Observed arithmetic accuracy (%)', 'Estimated native rating')
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    save(figure, 'conditional-variance.svg')
    print({**{key: value for key, value in prior.items() if np.ndim(value) == 0},
           **moments, 'estimates': points})


if __name__ == '__main__':
    main()
