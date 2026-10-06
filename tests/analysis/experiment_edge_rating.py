"""Evaluate fixed edge estimators without using commercial labels for fitting.

The comparison has 16 paired games, not 32 independent games. Cross-game
calibration uses other games' Maia policies only. Reference ratings are exposed
only to the diagnostic/scoring code after predictions have been completed.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analysis.cache import write_json
from analysis.game.study import load_game
from analysis.player_rating.evidence import validate_evidence
from analysis.player_rating.service import evidence_cache_path
from analysis.settings import CONFIG
from tests.analysis.compare_shared_curve_games import audit_evidence, read_json
from tests.analysis.experiment_shared_curve_lichess import _digest, _output_path, _source_paths
from tests.analysis.experiment_shared_curve_sweep import _write_csv
from tests.analysis import edge_quality_likelihood, edge_model_discrepancy, edge_global_quality, edge_distribution_transport
from tests.analysis import edge_posterior_consensus
from tests.analysis import edge_bounded_accuracy, edge_predictive_accuracy
from tests.analysis import edge_predictive_mixture
from tests.analysis import edge_monotone_quality

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'tests/analysis/output/shared-curve-edge-methods'
SIDES = ('White', 'Black')


def load_cases(games_dir):
    cases, sources = [], _source_paths()
    sources.extend(Path(module.__file__) for module in
                   (edge_quality_likelihood, edge_model_discrepancy, edge_global_quality,
                    edge_distribution_transport, edge_posterior_consensus,
                    edge_bounded_accuracy, edge_predictive_accuracy, edge_predictive_mixture,
                    edge_monotone_quality))
    sources.append(Path(__file__))
    sources.extend(Path(__file__).with_name(name+'.py') for name in
                   ('compare_shared_curve_games', 'experiment_shared_curve_lichess',
                    'experiment_shared_curve_sweep'))
    for pgn in sorted(Path(games_dir).glob('game*.pgn'), key=lambda p: int(p.stem[4:])):
        game = load_game(pgn)
        folder = pgn.parent/'output'/f'{pgn.stem}-full'
        analysis = read_json(folder/'analysis.json')
        fit = read_json(folder/'player-rating/fit.json')
        cache = evidence_cache_path(CONFIG['ANALYSIS']['CACHE_DIR'], analysis['rating_fit']['evidence_key'])
        evidence = validate_evidence(read_json(cache))
        audit = audit_evidence(game, analysis, evidence)
        ratings = {side: float(game.headers[f'{side}Elo']) for side in SIDES}
        cases.append({'game': pgn.stem, 'input': {'evidence': evidence, 'fit': fit, 'ratings': ratings},
                      'references': {side: int(game.headers[f'{side}EloEstimate']) for side in SIDES},
                      'performance': analysis['performance']['players'], 'audit': audit})
        sources.extend([pgn, cache, folder/'analysis.json', *(folder/'player-rating').glob('*')])
    if not cases:
        raise ValueError('Saved PGNs and full analysis are required.')
    return cases, sources


def predict(case, calibration_cases, ratings=None):
    data = case['input']
    evidence, fit = data['evidence'], data['fit']
    ratings = ratings if ratings is not None else data['ratings']
    result = {'current': {side: fit['players'][side]['estimate'] for side in SIDES}}
    for module in (edge_quality_likelihood, edge_model_discrepancy):
        result.update(module.predict(evidence, fit, ratings))
    result.update(edge_global_quality.predict(evidence, fit, ratings, calibration_cases))
    result.update({name+'_all': pair for name, pair in
                   edge_global_quality.predict(evidence, fit, ratings, calibration_cases, edge_only=False).items()})
    result.update(edge_distribution_transport.predict(evidence, fit, ratings, calibration_cases))
    result.update(edge_posterior_consensus.predict(evidence, fit, ratings, calibration_cases))
    result.update(edge_predictive_mixture.predict(evidence, fit, ratings, calibration_cases))
    result.update(edge_monotone_quality.predict(evidence, fit, ratings, calibration_cases))
    for module in (edge_bounded_accuracy, edge_predictive_accuracy):
        result.update(module.predict(evidence, fit, ratings, calibration_cases))
        result.update({name+'_all': pair for name,pair in
                       module.predict(evidence, fit, ratings, calibration_cases, edge_only=False).items()})
    for name, pair in result.items():
        for side in SIDES:
            if not np.isfinite(pair[side]) or not 0 <= pair[side] <= 3200:
                raise ValueError(f'{name}: invalid rating for {side}.')
            if name != 'current' and not name.endswith('_all') and not edge_quality_likelihood.is_edge(fit, side):
                if round(pair[side]) != fit['players'][side]['estimate']:
                    raise AssertionError(f'{name} changed an intersecting estimate.')
    return result


def partial_correlation(x, y, control):
    """Descriptive conditional association, not a fitted rating predictor."""
    a, b, c = (float(np.corrcoef(left, right)[0, 1]) for left, right in
               ((x, y), (x, control), (y, control)))
    return (a-b*c)/np.sqrt(max((1-b*b)*(1-c*c), 1e-15))


def associations(rows):
    reference = np.array([row['reference'] for row in rows])
    actual = np.array([row['actual'] for row in rows])
    result = {}
    for key in ('average_accuracy', 'lichess_accuracy', 'actual', 'current'):
        values = np.array([row[key] for row in rows])
        result[key] = {'pearson': float(pearsonr(values, reference).statistic),
                       'spearman': float(spearmanr(values, reference).statistic)}
        if key != 'actual':
            result[key]['partial_given_actual'] = float(partial_correlation(values, reference, actual))
            result[key]['actual_partial_given_feature'] = float(partial_correlation(actual, reference, values))
    return result


def theory(rows):
    pairs = []
    for left, right in combinations(rows, 2):
        if left['game'] == right['game']:
            continue
        if abs(left['actual']-right['actual']) <= 100 and abs(left['average_accuracy']-right['average_accuracy']) <= .25:
            pairs.append({'first': left['game']+left['side'][0], 'second': right['game']+right['side'][0],
                          'accuracy_gap': abs(left['average_accuracy']-right['average_accuracy']),
                          'actual_gap': abs(left['actual']-right['actual']),
                          'reference_gap': abs(left['reference']-right['reference']),
                          'lichess_accuracy_gap': abs(left['lichess_accuracy']-right['lichess_accuracy'])})
    # Resample games together, preserving White/Black dependence.
    rng = np.random.default_rng(20261004)
    groups = [rows[i:i+2] for i in range(0, len(rows), 2)]
    boot = {key: [] for key in ('accuracy', 'actual_given_accuracy', 'lichess')}
    for _ in range(2000):
        sample = [row for i in rng.integers(0, len(groups), len(groups)) for row in groups[i]]
        accuracy, actual, reference, lichess = (np.array([r[k] for r in sample]) for k in
                                               ('average_accuracy', 'actual', 'reference', 'lichess_accuracy'))
        boot['accuracy'].append(np.corrcoef(accuracy, reference)[0, 1])
        boot['actual_given_accuracy'].append(partial_correlation(actual, reference, accuracy))
        boot['lichess'].append(np.corrcoef(lichess, reference)[0, 1])
    return {'all': associations(rows), 'non_edge': associations([r for r in rows if not r['edge']]),
            'edge': associations([r for r in rows if r['edge']]),
            'paired_game_bootstrap_95_percent': {k: np.quantile(v, [.025, .975]).tolist() for k,v in boot.items()},
            'similar_inputs': sorted(pairs, key=lambda p: -p['reference_gap']),
            'actual_range': [min(r['actual'] for r in rows), max(r['actual'] for r in rows)],
            'caveat': 'Associations are descriptive; 16 selected games cannot identify a commercial formula or establish causation.'}


def rank(rows):
    result = []
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        errors = np.array([r['estimate']-r['reference'] for r in selected])
        edges = [r for r in selected if r['edge']]
        matches = sum(np.sign(selected[i]['estimate']-selected[i+1]['estimate']) ==
                      np.sign(selected[i]['reference']-selected[i+1]['reference']) for i in range(0,len(selected),2))
        result.append({'method': method, 'mae': float(np.mean(abs(errors))),
                       'rmse': float(np.sqrt(np.mean(errors**2))), 'maximum_error': float(max(abs(errors))),
                       'edge_mae': float(np.mean([abs(r['estimate']-r['reference']) for r in edges])),
                       'non_edge_mae': float(np.mean([abs(r['estimate']-r['reference']) for r in selected if not r['edge']])),
                       'own_elo_max_change': float(max(r['own_elo_max_change'] for r in selected)),
                       'own_elo_max_span': float(max(r['own_elo_span'] for r in selected)),
                       'opponent_elo_max_change': float(max(r['opponent_elo_max_change'] for r in selected)),
                       'ordering_matches': int(matches), 'games': len(selected)//2})
        result[-1]['edge_ordering_pairs'] = int(sum(
            np.sign(left['estimate']-right['estimate']) == np.sign(left['reference']-right['reference'])
            for left,right in combinations(edges,2)))
        result[-1]['edge_ordering_total'] = len(edges)*(len(edges)-1)//2
    return sorted(result, key=lambda r: (r['mae'],r['maximum_error'],r['method']))


def plots(observations, rows, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import rc_context
    fig=Figure(figsize=(12,5),layout='constrained'); FigureCanvasAgg(fig)
    axes=fig.subplots(1,2)
    for axis,key,label in zip(axes,('average_accuracy','lichess_accuracy'),('Arithmetic move accuracy (%)','Lichess game accuracy (%)')):
        scatter=axis.scatter([r[key] for r in observations],[r['reference'] for r in observations],
                             c=[r['actual'] for r in observations],cmap='viridis',s=40)
        for row in observations:
            if row['edge']:
                axis.annotate(row['game'].replace('game','')+row['side'][0],(row[key],row['reference']),xytext=(4,4),textcoords='offset points',fontsize=8)
        axis.set(xlabel=label,ylabel='Commercial reference',title='Five curve-edge observations labelled')
        axis.grid(alpha=.2)
    fig.colorbar(scatter,ax=list(axes),label='Actual Elo',shrink=.8)
    with rc_context({'svg.fonttype':'none'}): fig.savefig(output/'accuracy-associations.svg')
    fig.clear()
    choices={'current':'Current shared curve',
             'bounded_beta_account_5pct':'Beta edge fallback + 5% actual Elo',
             'predictive_accuracy_fft':'Full-policy predictive edge fallback',
             'monotone_predictive_mixture_account_5pct_all':'Monotone mixture + 5% actual Elo'}
    edges=sorted([r for r in observations if r['edge']],key=lambda r:-r['average_accuracy'])
    fig=Figure(figsize=(10,5),layout='constrained'); FigureCanvasAgg(fig)
    axis=fig.subplots(); index=np.arange(len(edges))
    axis.plot(index,[r['reference'] for r in edges],'kx--',label='Commercial reference')
    for method in choices:
        values={(r['game'],r['side']):r['estimate'] for r in rows if r['method']==method}
        axis.plot(index,[values[r['game'],r['side']] for r in edges],marker='o',label=choices[method])
    axis.set_xticks(index,[r['game']+r['side'][0]+'\n'+f'{r["average_accuracy"]:.2f}%' for r in edges])
    axis.set(ylabel='Played-strength estimate',xlabel='Descending observed arithmetic accuracy',ylim=(200,3000))
    axis.legend(fontsize=8);axis.grid(alpha=.2)
    with rc_context({'svg.fonttype':'none'}):fig.savefig(output/'edge-comparison.svg')
    fig.clear()


def run(games_dir=ROOT/'games', output=OUTPUT):
    from tests.analysis.retired_population import reject_benchmark_population
    reject_benchmark_population()
    started=perf_counter(); output=_output_path(output); output.mkdir(parents=True,exist_ok=True)
    cases,sources=load_cases(games_dir); before=_digest(sources)
    rows,observations=[],[]
    for case in cases:
        calibration=[other['input'] for other in cases if other is not case]
        baseline=predict(case,calibration)
        perturbed={}
        for side in SIDES:
            for shift in (-200,-100,100,200):
                ratings=dict(case['input']['ratings']); ratings[side]+=shift
                perturbed[side,shift]=predict(case,calibration,ratings)
        for side in SIDES:
            fit=case['input']['fit']; player=fit['players'][side]
            curve=fit['diagnostics']['curve']; performance=case['performance'][side.lower()]
            observation={'game':case['game'],'side':side,'actual':case['input']['ratings'][side],
                         'reference':case['references'][side],'average_accuracy':player['average_accuracy'],
                         'lichess_accuracy':performance['accuracy'],'current':player['estimate'],
                         'curve_low':curve['monotone_expected_accuracy'][0],
                         'curve_high':curve['monotone_expected_accuracy'][-1],
                         'sigma':curve['likelihood']['accuracy_sigma'],
                         'moves':player['moves_used'],'edge':bool(edge_quality_likelihood.is_edge(fit,side)),
                         'acpl':performance['average_centipawn_loss'],'blunders':performance['blunders']}
            observations.append(observation)
            for method,pair in baseline.items():
                own=[perturbed[side,shift][method][side] for shift in (-200,-100,100,200)]+[pair[side]]
                other='Black' if side=='White' else 'White'
                opponent=[perturbed[other,shift][method][side] for shift in (-200,-100,100,200)]
                rows.append({**observation,'method':method,'estimate':round(pair[side]),'unrounded_estimate':pair[side],
                             'own_elo_max_change':max(abs(v-pair[side]) for v in own),
                             'own_elo_span':max(own)-min(own),
                             'opponent_elo_max_change':max(abs(v-pair[side]) for v in opponent)})
        print(f'{case["game"]}: {len(baseline)} methods; actual Elo sensitivity checked',flush=True)
    ranking=rank(rows); diagnostic=theory(observations)
    after=_digest(sources)
    if before!=after: raise AssertionError('Production files or evidence changed.')
    result={'ranking':ranking,'theory':diagnostic,'observations':observations,'players':rows,
            'scope':{'games':len(cases),'players':len(observations),'edge_players':sum(r['edge'] for r in observations),
                     'calibration':'Leave-one-game-out Maia curves; no commercial labels or actual played accuracies in calibration.',
                     'parameters':'Fixed mathematical assumptions; no coefficient/threshold search against reference labels.',
                     'sensitivity':'One supplied own/opponent rating changes at a time by -200,-100,+100,+200.',
                     'production_unchanged':True,'protected_files':len(before)},
            'elapsed_seconds':perf_counter()-started}
    write_json(output/'comparison.json',result)
    _write_csv(output/'observations.csv',observations); _write_csv(output/'players.csv',rows); _write_csv(output/'ranking.csv',ranking)
    selected_methods=('current','bounded_beta_account_5pct','predictive_accuracy_fft',
                      'global_residual_transport_all','consensus_posterior_account_all',
                      'predictive_mixture_account_5pct_all','monotone_predictive_mixture_account_5pct_all')
    game_table=[]
    for case in cases:
        entry={'game':case['game'],'commercial':' / '.join(str(case['references'][s]) for s in SIDES)}
        for method in selected_methods:
            selected=[r for r in rows if r['game']==case['game'] and r['method']==method]
            entry[method]=' / '.join(str(r['estimate']) for r in selected)
        game_table.append(entry)
    _write_csv(output/'game-comparison.csv',game_table)
    write_json(output/'source-hashes.json',{'before':before,'after':after})
    plots(observations,rows,output)
    print(json.dumps({'ranking':ranking,'theory':diagnostic,'elapsed_seconds':result['elapsed_seconds']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-dir',type=Path,default=ROOT/'games')
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    options=parser.parse_args(); run(options.games_dir,options.output_dir)
