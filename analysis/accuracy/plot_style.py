"""Consistent colors and selectable SVG typography for accuracy plots."""

SIDE_COLORS = {'white': '#b25b36', 'black': '#167d95'}
SHARED_COLOR = '#403d8c'

SVG_STYLE = {
    'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    # Common, metrically compatible fonts keep browser text inside its legend.
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
}
