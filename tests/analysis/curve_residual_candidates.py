"""Prespecified, reference-free residual models for the shared-curve study.

The common account mean only supplies the curve expansion point. Rating order
is inherited from observed arithmetic accuracy, not from either account or a
reference label. These are research assumptions, not calibrated human models.
"""
from __future__ import annotations

import numpy as np

from analysis.player_rating.bayesian_shared_curve import SharedCurve, fit_pair

SIDES = ('White', 'Black')
NATIVE_VARIANCE = (2600.-600.)**2/12.
ELO_LOGIT_SCALE = 400./np.log(10.)
ENDPOINT_EPSILON = .005  # Half the existing 0.01-accuracy observation bin.


def describe():
    formulas = {
        'residual_secant_ridge': 'Expand at mean actual Elo m, use the native 600–2600 curve secant s, and r=m+[tau²*s/(v+tau²*s²)]*(A-C(m)); tau² is uniform native-rating variance.',
        'residual_logloss_ridge': 'Same ridge in negative log accuracy-loss coordinates; transform v with the delta method at C(m), and use the transformed native secant.',
        'residual_elo_logodds': 'r=m+(400/log(10))*(logit(A/100)-logit(C(m)/100)); an explicit Elo-logistic analogy for quality odds, not a claim that accuracy is a win probability.',
    }
    return {name: text+' Clip to200–3000; clipping is nondecreasing but can collapse distinct extremes to a tie.'
            for name, text in formulas.items()}


def prepare(evidence, actual_ratings=None):
    curve = fit_pair(evidence['White'], evidence['Black'])
    return {'model': SharedCurve(curve['monotone_expected_accuracy']),
            'variance': float(curve['likelihood']['accuracy_variance']),
            'accuracy': np.array([player['average_accuracy'] for player in curve['players']])}


def predict_prepared(prepared, actual_ratings):
    m = float(np.mean([actual_ratings[side] for side in SIDES]))
    model, v, accuracy = prepared['model'], prepared['variance'], prepared['accuracy']
    anchor = float(model(m))
    endpoints = np.asarray(model([600., 2600.]))
    slope = float(np.diff(endpoints)[0]/2000.)
    gain = NATIVE_VARIANCE*slope/max(v+NATIVE_VARIANCE*slope*slope, 1e-12)
    linear = m+gain*(accuracy-anchor)

    def bounded(a):
        return np.clip(a, ENDPOINT_EPSILON, 100.-ENDPOINT_EPSILON)

    loss = lambda a: -np.log(100.-bounded(a))
    s_loss = float(np.diff(loss(endpoints))[0]/2000.)
    v_loss = v/(100.-bounded(anchor))**2
    gain_loss = NATIVE_VARIANCE*s_loss/max(v_loss+NATIVE_VARIANCE*s_loss*s_loss, 1e-12)
    logloss = m+gain_loss*(loss(accuracy)-loss(anchor))
    odds = lambda a: np.log(bounded(a)/(100.-bounded(a)))
    logodds = m+ELO_LOGIT_SCALE*(odds(accuracy)-odds(anchor))
    return {name: dict(zip(SIDES, map(float, np.clip(value, 200., 3000.)), strict=True))
            for name, value in zip(describe(), (linear, logloss, logodds), strict=True)}


def predict(evidence, actual_ratings):
    return predict_prepared(prepare(evidence), actual_ratings)
