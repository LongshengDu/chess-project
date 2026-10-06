"""Export rating curves, priors and estimator-specific decisions as figures."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from analysis.cache import write_json
from analysis.player_rating.scale import from_native, native_jacobian, scale_code


COLORS = {'White': '#16828a', 'Black': '#bd6334'}
DISPLAY_RATING_RANGE = (200., 3000.)
PANEL_SIZE = (7.5, 5.4)
LEGACY_IMAGES = ('curve-and-prior.png', 'curve-and-prior.svg', 'posterior.png',
                 'posterior.svg', 'analysis.png', 'prior.png')


def _accuracy_intersection(ratings, accuracy, observed):
    """Invert the plotted monotone curve, retaining a range for flat crossings."""
    if observed < accuracy[0] or observed > accuracy[-1]:
        return None
    equal = np.flatnonzero(np.isclose(accuracy, observed, rtol=0., atol=1e-10))
    if len(equal):
        return float(ratings[equal[0]]), float(ratings[equal[-1]])
    rating = float(np.interp(observed, accuracy, ratings))
    return rating, rating


def _accuracy_settings(fit, curve):
    """Describe recorded likelihood settings without inventing historical values."""
    details = []
    selection = curve.get('selection', {})
    probability = curve.get('top_probability', selection.get('top_probability',
        fit.get('parameters', {}).get('top_probability')))
    if isinstance(probability, (int, float)) and not isinstance(probability, bool) and 0 < probability <= 1:
        details.append('All legal moves' if probability == 1 else f'Top {probability:.0%} probability')
    likelihood = curve.get('likelihood', {})
    sigma, scale = likelihood.get('accuracy_sigma'), likelihood.get('sigma_scale')
    if isinstance(sigma, (int, float)) and not isinstance(sigma, bool) and np.isfinite(sigma) and sigma >= 0:
        label = f'σ = {sigma:.3f} accuracy points'
        if isinstance(scale, (int, float)) and not isinstance(scale, bool) and np.isfinite(scale) and scale > 0:
            label += f' (scale ×{scale:g})'
        details.append(label)
    return ' · '.join(details)


def _component_estimates(axis, fit, rating_scale, display_range):
    """Show a point estimator's saved components without inventing a posterior."""
    rows = ((4, 'Final estimate', None), (3, 'Local curve mean', 'local_mean'),
            (2, 'Population curve mean', 'population_mean'),
            (1, 'Actual Elo supplied', 'actual_rating'),
            (0, 'Shared account anchor', 'common_account_rating'))
    explanation = 'Local and population means precede coverage and account adjustment'
    components = fit['diagnostics']['components']
    for side, offset in (('White', .13), ('Black', -.13)):
        labelled = False
        for level, _, key in rows:
            component = components.get(side, {})
            value = (component.get('unrounded_estimate') if key is None else component.get(key))
            native_value = value is not None
            if key is None and value is None:
                value = fit['players'][side]['estimate']
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{side} component estimates must be finite numbers or absent.')
            if native_value:
                value = from_native(value, rating_scale)
            location = float(np.clip(value, *display_range))
            marker = ('D' if key is None else 's' if key in
                      ('account_rating_used', 'actual_rating', 'common_account_rating') else 'o')
            outside = location != value
            if outside:
                marker = '<' if value < location else '>'
            axis.scatter(location, level+offset, color=COLORS[side], marker=marker,
                         s=55 if key is None else 35, zorder=4,
                         label=side if not labelled else None)
            labelled = True
            left = location > display_range[1]-.09*(display_range[1]-display_range[0])
            label = f'{value:,.0f}'+(' (outside axis)' if outside else '')
            axis.annotate(label, (location, level+offset), xytext=(-7 if left else 7, 0),
                          textcoords='offset points', ha='right' if left else 'left',
                          va='center', fontsize=9, color=COLORS[side])
    axis.set_yticks([row[0] for row in rows], [row[1] for row in rows], fontsize=9)
    axis.set_ylim(-.5, rows[0][0]+.8)
    axis.set_title('Component point estimates · no uncertainty interval\n'+explanation, fontsize=11)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=9, loc='upper left', ncols=2)
    unavailable = [side for side in ('White', 'Black') if fit['players'][side]['estimate'] is None]
    if unavailable:
        axis.text(.98, .98, 'No fitted estimate: '+', '.join(unavailable), fontsize=8,
                  ha='right', va='top', transform=axis.transAxes)




def _shared_affine_curve(axis, fit, rating_scale, *, noise_band=False):
    """Show the current game's curve and its conditional measurement spread."""
    diagnostics = fit['diagnostics']
    curve, model = diagnostics['curve'], diagnostics['model']
    x = from_native(np.asarray(curve['fine_ratings'], dtype=float), rating_scale)
    accuracy = np.asarray(curve['shared_accuracy'], dtype=float)
    if accuracy.shape != x.shape or not np.isfinite(accuracy).all():
        raise ValueError('The shared accuracy curve must match the recorded finite rating grid.')
    axis.plot(x, accuracy, color='#344154', lw=2, label='Game shared curve C')
    variance = model.get('measurement_variance')
    if noise_band and variance is not None:
        sigma = np.sqrt(variance)
        axis.fill_between(x, np.maximum(0., accuracy-sigma), np.minimum(100., accuracy+sigma),
                          color='#344154', alpha=.12,
                          label=f'Conditional measurement ±1σ ({sigma:.2f} accuracy points)')


def _shared_affine_variance(axis, fit):
    """Explain the affine denominator without a population calibration term."""
    affine, model = fit['diagnostics']['affine'], fit['diagnostics']['model']
    noise = model['measurement_variance']
    signal = max(0., affine['accuracy_variance']-noise)
    total = signal+noise
    labels = ('Curve variation under prior Var(C)', 'Game measurement variance v', 'Affine denominator Var(C) + v')
    values = (signal, noise, total)
    maximum = max(values) or 1.
    for y, value in enumerate(values):
        axis.barh(y, value, color=('#344154', '#647b96', '#217758')[y], height=.55)
        axis.text(value+.025*maximum, y, f'{value:.3f}', va='center', fontsize=10)
    axis.set_yticks(range(3), labels, fontsize=9)
    axis.invert_yaxis()
    axis.set(xlim=(0., 1.25*maximum), xlabel='Variance (accuracy points²)',
             title='Accuracy evidence and measurement uncertainty')
    axis.text(.5, -.25, 'Affine slope b = Cov(R, C) / (Var(C) + v)\n'
              'All curve and measurement moments come from this game.',
              transform=axis.transAxes, ha='center', fontsize=9)
    axis.spines[['top', 'right']].set_visible(False)


def _affine_decomposition(axis, fit, rating_scale, display_range):
    """An additive native decision, with endpoint differences on other scales."""
    affine = fit['diagnostics']['affine']
    if affine.get('prior_mean') is None:
        axis.text(.5, .5, 'Insufficient information for an affine estimate', ha='center', transform=axis.transAxes)
        return
    origin = from_native(affine['prior_mean'], rating_scale)
    converted = scale_code(rating_scale) != 'lb'
    for side, y in (('White', 1.), ('Black', 0.)):
        player = fit['players'][side]
        point = player.get('canonical_estimate', player.get('unrounded_estimate', player.get('estimate')))
        if point is None:
            axis.text(.5, y/2+.15, f'{side}: no estimate', transform=axis.transAxes)
            continue
        target = from_native(point, rating_scale)
        start, stop = (float(np.clip(value, *display_range)) for value in (origin, target))
        axis.annotate('', (stop, y), xytext=(start, y),
                      arrowprops={'arrowstyle': '->', 'lw': 2, 'color': COLORS[side]})
        axis.scatter(start, y, marker='s', color='#647b96', s=45, zorder=4)
        axis.scatter(stop, y, marker='D', color=COLORS[side], s=55, zorder=4)
        difference = target-origin
        axis.text(.5, .85 if y else .39,
                  f'{side}: {origin:,.0f} {"+" if difference >= 0 else "−"} {abs(difference):,.0f} ≈ {target:,.0f}',
                  transform=axis.transAxes, ha='center', fontsize=12, color=COLORS[side], fontweight='bold')
        component = fit['diagnostics']['components'][side]
        residual = component.get('observed_accuracy', player.get('average_accuracy'))-affine['accuracy_mean']
        axis.text(.5, .68 if y else .20, f'Observed accuracy residual: {residual:+.2f} points',
                  transform=axis.transAxes, ha='center', fontsize=9, color=COLORS[side])
        if start != origin or stop != target:
            axis.text(.5, .62 if y else .13, 'An endpoint is outside the displayed rating axis',
                      transform=axis.transAxes, ha='center', fontsize=8)
    axis.set_yticks([1., 0.], ['White', 'Black'])
    axis.set_ylim(-.65, 1.55)
    subtitle = ('Native affine decision, then converted; arrows show displayed differences' if converted
                else 'Prior mean + accuracy adjustment = final estimate')
    axis.set_title('One common affine decision · no uncertainty interval\n'+subtitle, fontsize=10)




def _affine_mapping(axis, fit, rating_scale):
    diagnostics = fit['diagnostics']
    affine, curve = diagnostics['affine'], diagnostics['curve']
    observed = [player['average_accuracy'] for player in fit['players'].values()
                if player.get('average_accuracy') is not None]
    if not observed or affine.get('affine_slope') is None:
        axis.text(.5, .5, 'Insufficient information for a common accuracy mapping', ha='center', transform=axis.transAxes)
        return
    low = max(0., min(*observed, affine['accuracy_mean'])-8.)
    high = min(100., max(*observed, affine['accuracy_mean'])+5.)
    accuracy = np.linspace(low, high, 401)
    support = fit.get('canonical_rating_range', fit.get('parameters', {}).get('rating_range', (0., 3200.)))
    predictions = np.clip(affine['prior_mean']+affine['affine_slope']*(accuracy-affine['accuracy_mean']), *support)
    native_grid = np.asarray(curve['fine_ratings'], dtype=float)
    axis.plot(curve['shared_accuracy'], from_native(native_grid, rating_scale), ls='--', color='#647b96',
              label='Raw shared-curve inverse (where it exists)')
    axis.plot(accuracy, from_native(predictions, rating_scale), color='#217758', lw=2,
              label='Common affine decision, then scale conversion')
    origin = from_native(affine['prior_mean'], rating_scale)
    axis.scatter(affine['accuracy_mean'], origin, marker='s', color='#647b96', s=35, label='Prior mean accuracy / rating')
    for side, offset in (('White', (8, 15)), ('Black', (8, -23))):
        player = fit['players'][side]
        native = player.get('canonical_estimate', player.get('unrounded_estimate', player.get('estimate')))
        if native is None:
            continue
        point = from_native(native, rating_scale)
        observed = player['average_accuracy']
        axis.scatter(observed, point, color=COLORS[side], s=35, zorder=5)
        # Right-bound observations (e.g. perfect accuracy) need inward labels.
        near_right = observed > high-.2*(high-low)
        axis.annotate(f'{side}: {point:,.0f}', (observed, point),
                      xytext=(-8 if near_right else offset[0], offset[1]), textcoords='offset points',
                      ha='right' if near_right else 'left', color=COLORS[side], fontsize=9,
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 2})
    axis.set(xlim=(low, high), ylim=from_native(DISPLAY_RATING_RANGE, rating_scale),
             xlabel='Observed arithmetic accuracy (%)', ylabel='Estimated rating')
    axis.set_title(f'Native affine slope: {affine["affine_slope"]:.2f} Elo per accuracy point', fontsize=11)
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(fontsize=8, loc='upper left')


def _affine_prior(axis, fit, rating_scale):
    diagnostics = fit['diagnostics']
    prior_source = diagnostics['model']
    curve, affine = diagnostics['curve'], diagnostics['affine']
    grid = np.asarray(curve['fine_ratings'], dtype=float)
    x, jacobian = from_native(grid, rating_scale), native_jacobian(grid, rating_scale)
    densities = [np.asarray(values, dtype=float)/jacobian for values in
                 (prior_source['base_prior_density'], curve['prior_density'])]
    peak = max(float(values.max()) for values in densities) or 1.
    for label, values, color, style in zip(('Unshifted prior', 'Actual-rating translated prior'), densities,
            ('#aab4c1', '#647b96'), ('--', '-'), strict=True):
        axis.plot(x, values/peak, color=color, ls=style, lw=2, label=label)
    axis.fill_between(x, densities[1]/peak, alpha=.12, color='#647b96')
    for key, label, color in (('account_anchor', 'Common account anchor', '#8f79a8'), ('prior_mean', 'Prior mean after truncation', '#217758')):
        if affine.get(key) is not None:
            point = from_native(affine[key], rating_scale)
            axis.axvline(point, ls=':', color=color, lw=1.2, label=f'{label}: {point:,.0f}')
    axis.set(ylim=(0., 1.08), xlim=from_native(DISPLAY_RATING_RANGE, rating_scale), ylabel='Relative prior density')
    axis.set_title('Translate the common prior; truncate and normalize', fontsize=11)
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(fontsize=8, loc='upper left')


def export_figures(fit, output_dir, *, title=None):
    """Save fit.json, a combined analysis SVG and a panel-sized prior SVG.

    This renderer consumes estimator output; it never recalculates a posterior
    or reads external ratings, reference ratings, PGNs, engines or model APIs.
    Other rating methods can omit curve diagnostics and still export fit.json.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {'fit': output / 'fit.json'}
    write_json(paths['fit'], fit)
    diagnostics = fit.get('diagnostics', {})
    curve = diagnostics.get('curve', {})
    shared_affine = diagnostics.get('kind') == 'shared_curve_affine'
    point_estimator = shared_affine or diagnostics.get('kind') == 'arithmetic_coverage'
    required = ('rating_grid', 'fine_ratings', 'shared_accuracy', 'prior_density')
    available = 'components' in diagnostics if point_estimator else 'posterior_densities' in curve
    if not available or not all(key in curve for key in required):
        for name in (*LEGACY_IMAGES, 'analysis.svg', 'prior.svg', 'method-explanation.svg'):
            (output/name).unlink(missing_ok=True)
        return paths
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import rc_context

    rating_scale = fit.get('rating_scale', {'scale': 'lb', 'name': 'Lichess Blitz'})
    native_x = np.asarray(curve['fine_ratings'], dtype=float)
    x = from_native(native_x, rating_scale)
    jacobian = native_jacobian(native_x, rating_scale)
    accuracy = np.asarray(curve['shared_accuracy'], dtype=float)
    prior = np.asarray(curve['prior_density'], dtype=float)
    players = fit['players']
    central = fit.get('central_interval')
    measured = from_native(np.asarray(curve['rating_grid'], dtype=float), rating_scale)
    low, high = float(measured[0]), float(measured[-1])
    display_range = from_native(DISPLAY_RATING_RANGE, rating_scale)
    display_low, display_high = display_range
    if (x.ndim != 1 or len(x) < 2 or accuracy.shape != x.shape or prior.shape != x.shape
            or not np.isfinite([x, accuracy, prior]).all() or np.any(np.diff(x) <= 0)
            or np.any(np.diff(accuracy) < -1e-10)):
        raise ValueError('Rating figures require finite monotone accuracy on an increasing grid.')
    if jacobian.shape != x.shape or not np.isfinite(jacobian).all() or np.any(jacobian <= 0):
        raise ValueError('Rating conversion must have a finite positive derivative on the figure grid.')
    prior = prior/jacobian
    raw_prior = curve.get('prior_weights')
    prior_weights = (np.asarray(raw_prior, dtype=float) if raw_prior is not None
                     else prior / prior.max() if prior.max() > 0 else prior.copy())
    if (prior_weights.shape != x.shape or not np.isfinite(prior_weights).all()
            or np.any((prior_weights < 0) | (prior_weights > 1))):
        raise ValueError('Prior weights must be finite values from zero to one on the figure grid.')
    converted_scale = scale_code(rating_scale) != 'lb'
    if converted_scale:
        prior_weights = prior/prior.max() if prior.max() > 0 else prior.copy()

    def figure(size):
        fig = Figure(figsize=size, layout='constrained')
        FigureCanvasAgg(fig)
        return fig

    def decorate(axis, ylabel):
        scale_name = rating_scale.get('name', scale_code(rating_scale)) if isinstance(rating_scale, dict) else rating_scale
        axis.set(xlim=display_range, xlabel=f'Rating ({scale_name})', ylabel=ylabel)
        axis.set_xticks(np.arange(np.ceil(display_low/200)*200, display_high+1, 200))
        axis.tick_params(axis='x', labelsize=8)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)

    def shade_extrapolation(axis):
        axis.axvspan(display_low, low, color='#dce1e6', alpha=.55)
        axis.axvspan(high, display_high, color='#dce1e6', alpha=.55)

    def save(fig, stem):
        path = output / f'{stem}.svg'
        temporary = None
        try:
            with NamedTemporaryFile(dir=output, prefix=f'.{stem}-', suffix='.svg', delete=False) as file:
                temporary = Path(file.name)
            with rc_context({'svg.fonttype': 'none'}):
                fig.savefig(temporary, facecolor='white', format='svg')
            temporary.replace(path)
            paths[f'{stem}_svg'] = path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            fig.clear()

    fig = figure((2*PANEL_SIZE[0], PANEL_SIZE[1]))
    fig.suptitle(title or fit.get('name', 'Player-rating analysis'), fontsize=15, fontweight='bold')
    axis, posterior_axis = fig.subplots(1, 2)
    shade_extrapolation(axis)
    if shared_affine:
        _shared_affine_curve(axis, fit, rating_scale)
    else:
        axis.plot(x, accuracy, color='#344154', lw=2, label='Shared expected accuracy')
    knots = curve.get('monotone_expected_accuracy')
    if knots is not None and len(knots) == len(measured):
        axis.scatter(measured, knots, color='#344154', s=13, zorder=4,
                     label='Monotone Maia accuracy knots')
    native_low, native_high = DISPLAY_RATING_RANGE
    visible_native = np.r_[native_low, native_x[(native_x > native_low) & (native_x < native_high)], native_high]
    visible_accuracy = np.interp(visible_native, native_x, accuracy)
    for index, (side, player) in enumerate(players.items()):
        observed = player.get('average_accuracy')
        if observed is None:
            continue
        color = COLORS[side]
        axis.axhline(observed, color=color, ls='--', lw=1.4,
                     label=f'{side} played: {observed:.2f}%')
        crossing = _accuracy_intersection(visible_native, visible_accuracy, observed)
        label_y = .53 if index == 0 else .30
        if crossing is None:
            axis.text(.50, label_y, f'{side}: no intersection in {display_low:.0f}–{display_high:.0f}',
                      color=color, fontsize=9, ha='center', transform=axis.transAxes,
                      bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white', 'edgecolor': color, 'alpha': .95})
            continue
        start, end = from_native(crossing, rating_scale)
        rating = from_native(sum(crossing)/2, rating_scale)
        value = f'{rating:,.0f}' if end-start < 1e-8 else f'{start:,.0f}–{end:,.0f}'
        extrapolated = start < low or end > high
        label = f'{side} intersection: {value}\nAccuracy: {observed:.2f}%'
        if extrapolated:
            label += '\nExtrapolated'
        if end-start >= 1e-8:
            axis.plot([start, end], [observed, observed], color=color, lw=4, alpha=.7)
        axis.scatter([rating], [observed], color=color, s=38, zorder=5, edgecolor='white')
        axis.vlines(rating, 50, observed, color=color, ls=':', lw=1, alpha=.65)
        label_x = float(np.clip((rating-display_low)/(display_high-display_low), .20, .76))
        axis.annotate(label, (rating, observed), xytext=(label_x, label_y),
                      textcoords='axes fraction', ha='center', fontsize=9, color=color,
                      bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white', 'edgecolor': color, 'alpha': .95},
                      arrowprops={'arrowstyle': '-', 'color': color, 'lw': .9})
    settings = _accuracy_settings(fit, curve)
    accuracy_title = ('Game shared accuracy curve' if shared_affine else
                      'Arithmetic reference curve' if point_estimator else 'Shared accuracy')+' · gray regions are extrapolated'
    if settings:
        accuracy_title += f'\n{settings}'
    axis.set_title(accuracy_title, fontsize=11)
    decorate(axis, 'Arithmetic move accuracy (%)')
    axis.set_ylim(50, 100)
    axis.legend(fontsize=8, loc='lower left', framealpha=.95)

    axis = posterior_axis
    if point_estimator:
        decorate(axis, '')
        if shared_affine:
            _affine_decomposition(axis, fit, rating_scale, display_range)
        else:
            _component_estimates(axis, fit, rating_scale, display_range)
    else:
        _posterior_estimates(axis, fit, x, jacobian, central, shade_extrapolation, decorate)
    save(fig, 'analysis')

    fig = figure(PANEL_SIZE)
    axis = fig.subplots()
    axis.fill_between(x, prior_weights, color='#647b96', alpha=.16)
    axis.plot(x, prior_weights, color='#647b96', lw=2)
    prior_title = ('Translated rating prior before normalization' if shared_affine else
                   'Component rating prior before normalization' if point_estimator else 'Rating prior before normalization')
    axis.set_title('Converted rating prior density (peak = 1)' if converted_scale else
                   prior_title if raw_prior is not None else 'Relative rating prior (peak = 1)', fontsize=11)
    decorate(axis, 'Relative prior density' if converted_scale else 'Prior weight')
    axis.set_ylim(0., 1.)
    axis.set_yticks(np.linspace(0., 1., 6))
    axis.yaxis.set_major_formatter('{x:.1f}')
    save(fig, 'prior')
    if shared_affine and diagnostics['affine'].get('affine_slope') is not None:
        fig = figure((2*PANEL_SIZE[0], 2*PANEL_SIZE[1]))
        fig.suptitle((title or fit.get('name', 'Player-rating analysis'))+' · method explanation', fontsize=15, fontweight='bold')
        variance_axis, curve_axis, prior_axis, mapping_axis = fig.subplots(2, 2).flat
        _shared_affine_variance(variance_axis, fit)
        _shared_affine_curve(curve_axis, fit, rating_scale, noise_band=True)
        for side, player in players.items():
            if player.get('average_accuracy') is not None:
                curve_axis.axhline(player['average_accuracy'], ls=':', color=COLORS[side], lw=1,
                                   label=f'{side}: {player["average_accuracy"]:.2f}%')
        decorate(curve_axis, 'Arithmetic accuracy (%)')
        curve_axis.set_ylim(50., 100.)
        curve_axis.set_title('Current-game accuracy evidence and conditional spread', fontsize=11)
        curve_axis.legend(fontsize=8, loc='lower left')
        decorate(prior_axis, 'Relative prior density')
        _affine_prior(prior_axis, fit, rating_scale)
        _affine_mapping(mapping_axis, fit, rating_scale)
        scale_name = rating_scale.get('name', 'Lichess Blitz') if isinstance(rating_scale, dict) else rating_scale
        mapping_axis.set_ylabel(f'Estimated rating ({scale_name})')
        save(fig, 'method-explanation')
    else:
        (output/'method-explanation.svg').unlink(missing_ok=True)
    for name in LEGACY_IMAGES:
        (output/name).unlink(missing_ok=True)
    return paths


def _posterior_estimates(axis, fit, x, jacobian, central, shade_extrapolation, decorate):
    """Keep the Bayesian method's posterior and central-interval rendering."""
    curve, players = fit['diagnostics']['curve'], fit['players']
    shade_extrapolation(axis)
    unavailable = []
    for side in ('White', 'Black'):
        player = players[side]
        raw_density = curve['posterior_densities'][side]
        if raw_density is None:
            unavailable.append(f'{side}: insufficient information for a rating estimate')
            continue
        density = np.asarray(raw_density, dtype=float)
        if density.shape != x.shape or not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError(f'{side} posterior must be a nonnegative density on the figure grid.')
        density = density/jacobian
        estimate = player['estimate']
        label = side
        if estimate is not None:
            lower, upper = player['interval']
            # Insert exact displayed endpoints so shading does not lose a grid cell.
            inner = np.r_[lower, x[(x > lower) & (x < upper)], upper]
            axis.fill_between(inner, np.interp(inner, x, density), color=COLORS[side], alpha=.22)
            axis.axvline(estimate, color=COLORS[side], ls='--', lw=1.2)
            label = f'{side}: {estimate:,.0f} [{lower:,.0f}–{upper:,.0f}] · {player["moves_used"]} moves'
        axis.plot(x, density, color=COLORS[side], lw=2, label=label)
    if unavailable:
        axis.text(.04, .05, '\n'.join(unavailable), fontsize=9, transform=axis.transAxes)
    axis.set_title(f'White and Black rating estimates · shaded central {central:.0%}', fontsize=11)
    axis.set_ylim(0, max(axis.get_ylim()[1], 1e-12)*1.18)
    decorate(axis, 'Posterior density (per Elo)')
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=9, loc='best')


def export_saved_figures(analysis, output_dir, *, title=None):
    """Render the compact rating result stored in a full-game analysis."""
    fit = {
        'players': {side.title(): values for side, values in analysis['played_elo'].items()},
        'method_id': analysis['played_elo_method'],
        'name': analysis.get('played_elo_name', analysis['played_elo_method']),
        'central_interval': analysis['played_elo_central_interval'],
        'rating_range': analysis['played_elo_rating_range'],
        'prior': analysis['played_elo_prior'],
        'diagnostics': analysis.get('played_elo_diagnostics', {}),
        'rating_fit': analysis['rating_fit'],
        'interval_scope': analysis.get('played_elo_interval_scope'),
        'rating_scale': analysis.get('played_elo_scale', {'scale': 'lb', 'name': 'Lichess Blitz'}),
        'canonical_rating_range': analysis.get('played_elo_canonical_rating_range', analysis['played_elo_rating_range']),
    }
    return export_figures(fit, output_dir, title=title)
