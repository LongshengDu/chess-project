"""Synthetic move-policy evidence for rating-method unit tests; no game assets."""


def evidence_fixture():
    evidence = {}
    for side, played in (('White', 0), ('Black', 1)):
        rows = []
        for move in range(6):
            policy = [[.15+.02*r, .55-.015*r, .30-.005*r] for r in range(21)]
            rows.append({'played_index': played, 'qualities': {'position': [100., 75.+move, 30.+move],
                                                               'root': [100., 75.+move, 30.+move]},
                         'weight': 1., 'maia_probabilities': policy,
                         'position_win_probability': .45+.05*move})
        evidence[side] = {'observations': rows}
    return evidence
