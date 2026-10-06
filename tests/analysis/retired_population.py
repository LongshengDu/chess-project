"""Prevent withdrawn research runners from reusing test games as a population.

The historical numerical helpers remain available for synthetic unit tests.
Their benchmark experiments are withdrawn: omitting reference labels does not
make test positions independent training or population-calibration evidence.
"""


REASON = ('Test games cannot be used as population/training inputs. '
          'This historical experiment is withdrawn; use a single-game estimator '
          'that does not consume a benchmark-derived corpus.')


def reject_benchmark_population():
    """Fail before a withdrawn runner reads games, builds assets, or writes fits."""
    raise ValueError(REASON)
