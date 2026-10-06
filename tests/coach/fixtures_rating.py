"""A deliberately simple replacement estimator for application wiring tests."""
from contextlib import contextmanager
import importlib
import inspect
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import analysis.player_rating as rating_package
from analysis.settings import CONFIG


def fixed_fit(evidence, *, central_interval=.5):
    players = {}
    for side, value in (('White', 1300), ('Black', 1700)):
        count = len(evidence[side]['observations'])
        players[side] = {
            'estimate': value if count else None,
            'uncertainty': 100 if count else None,
            'interval': [value-100, value+100] if count else [600, 2600],
            'moves_used': count, 'identifiable': bool(count), 'at_rating_limit': False,
        }
    return {'players': players, 'prior': {'kind': 'test'},
            'rating_range': [600, 2600], 'central_interval': central_interval,
            'interval_scope': 'Fixture interval for integration tests.',
            'method': 'Test-only fixed values verify estimator dispatch.',
            'diagnostics': {'central_interval': central_interval}}


@contextmanager
def replacement_estimator():
    """Drop a real module onto the package search path, without a registry edit."""
    module_name = 'analysis.player_rating.test_replacement'
    absent = object()
    previous_module = sys.modules.pop(module_name, absent)
    previous_attribute = getattr(rating_package, 'test_replacement', absent)
    source = ('from analysis.player_rating.interface import PlayerRating\n\n'
              + inspect.getsource(fixed_fit)
              + "\n\nclass Rating(PlayerRating):\n"
                "    name = 'Test replacement fit'\n\n"
                "    def fit(self, evidence):\n"
                "        return fixed_fit(evidence)\n")
    try:
        with tempfile.TemporaryDirectory(prefix='chess-rating-method-') as folder:
            path = Path(folder) / 'test_replacement.py'
            path.write_text(source, encoding='utf-8')
            with patch.object(rating_package, '__path__', [folder, *rating_package.__path__]), \
                 patch.dict(CONFIG['ANALYSIS']['PLAYER_RATING'], METHOD='test_replacement'):
                importlib.invalidate_caches()
                yield path
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not absent:
            sys.modules[module_name] = previous_module
        if previous_attribute is absent:
            if hasattr(rating_package, 'test_replacement'):
                delattr(rating_package, 'test_replacement')
        else:
            rating_package.test_replacement = previous_attribute
        importlib.invalidate_caches()
