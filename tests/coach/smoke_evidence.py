"""Minimal evidence for a single-position report test."""
from coach.tools_evidence import rounded


def smoke_evidence(position, comparison):
    return rounded({'ply': comparison['ply'], 'label': comparison['label'], 'fen': comparison['fen'],
        'material': position['material'], 'diagram': comparison['comparison_diagram'],
        'baseline': comparison.get('baseline') or position['baseline'],
        'played': comparison['played'], 'alternative': comparison['candidate']})
