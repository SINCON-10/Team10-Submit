"""Three learned combinations, fitted only to chronological OOF predictions."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import KEYS, dump, read_json, log_predictions, rmsle
from config import LEARNED, candidates


def fit_global(frame, experts, regularization):
    Z, y = frame[experts].to_numpy(dtype=float), frame.y.to_numpy(dtype=float)
    m = len(experts)
    prior = np.full(m, 1/m)
    # Work with sufficient statistics; avoid an N x M multiplication per step.
    gram, cross = Z.T@Z/len(y), Z.T@y/len(y)
    def objective(w):
        return float(w@gram@w-2*cross@w+regularization*np.square(w-prior).sum())
    def jacobian(w):
        return 2*(gram@w-cross+regularization*(w-prior))
    result = minimize(objective, prior, jac=jacobian, method='SLSQP',
        bounds=[(0., 1.)]*m, constraints={'type': 'eq', 'fun': lambda w: w.sum()-1., 'jac': lambda w: np.ones(m)},
        options={'ftol': 1e-12, 'maxiter': 2000})
    if not result.success:
        raise RuntimeError('가중치 최적화 실패: '+result.message)
    weights = np.maximum(result.x, 0.)
    return weights/weights.sum()


def fit_one(name, parameters, train, valid, experts, config, amount=None, prior=None):
    if name == 'learned_global_weights':
        weights = fit_global(train, experts, parameters['regularization'])
        prediction = log_predictions(valid[experts].to_numpy()@weights)
        return dict(model=weights, prediction=prediction,
                    recipe=dict(name=name, backend='global_weights', experts=experts, weights=weights.tolist(),
                                parameters=parameters, amount=1), curve=pd.DataFrame())
    if name == 'ridge_stacking':
        model = make_pipeline(StandardScaler(), Ridge(alpha=parameters['alpha']))
        model.fit(train[experts].to_numpy(), train.y)
        return dict(model=model, prediction=log_predictions(model.predict(valid[experts].to_numpy())),
                    recipe=dict(name=name, backend='ridge_stacking', experts=experts, parameters=parameters, amount=1),
                    curve=pd.DataFrame())
    if name == 'context_gate':
        from torch_models import fit_gate
        return fit_gate(parameters, train, valid, experts, prior, config, monitor=amount is None, amount=amount)
    raise ValueError(name)


def save(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if result['recipe']['backend'] == 'context_gate':
        from torch_models import save_ft
        save_ft(result, folder)
    else:
        joblib.dump(result['model'], folder/'model.joblib', compress=3)
        dump(folder/'recipe.json', result['recipe'])


def predict_saved(folder, frame, config):
    folder = Path(folder)
    recipe = read_json(folder/'recipe.json')
    if recipe['backend'] == 'context_gate':
        from torch_models import predict_gate
        return predict_gate(folder, frame, config)
    Z = frame[recipe['experts']].to_numpy(dtype=float)
    if recipe['backend'] == 'global_weights':
        return log_predictions(Z@np.asarray(recipe['weights']))
    return log_predictions(joblib.load(folder/'model.joblib').predict(Z))


def develop(train, valid, experts, config, folder):
    """Earlier calibration dates fit weights; later dates select settings/steps."""
    folder = Path(folder)
    assert train.date.max() < valid.date.min()
    completed = folder/'selections.json'
    if completed.is_file():
        selected = read_json(completed)
        predictions = {name: predict_saved(folder/name/'best_model', valid, config) for name in LEARNED}
        print('[앙상블 탐색 재개] 완료된 설정과 모델 재사용', flush=True)
        return selected, predictions
    prior = fit_global(train, experts, .001)
    selected, rows, predictions = {}, [], {}
    for name in LEARNED:
        best, chosen = np.inf, None
        for number, parameters in enumerate(candidates(name, config, meta=True)):
            result = fit_one(name, parameters, train, valid, experts, config, prior=prior)
            score = rmsle(valid.y, result['prediction'])
            trial = folder/name/f'trial_{number:03d}'
            trial.mkdir(parents=True, exist_ok=True)
            dump(trial/'result.json', dict(parameters=parameters, rmsle=score, recipe=result['recipe']))
            if len(result['curve']):
                result['curve'].to_csv(trial/'learning_curve.csv', index=False)
            rows.append(dict(model=name, trial=number, validation_rmsle=score, parameters=str(parameters),
                             amount=result['recipe']['amount'], hit_cap=result['recipe'].get('hit_cap', False)))
            print(f'[앙상블 탐색] {name} #{number+1}: RMSLE={score:.6f}', flush=True)
            if score < best:
                best, chosen = score, result['recipe']
                save(result, folder/name/'best_model')
                predictions[name] = result['prediction'].copy()
        selected[name] = dict(parameters=chosen['parameters'], amount=chosen['amount'],
                              validation_rmsle=best, experts=experts)
        dump(folder/name/'selection.json', selected[name])
    pd.DataFrame(rows).to_csv(folder/'trials.csv', index=False)
    dump(folder/'selections.json', selected)
    return selected, predictions


def refit(selected, train, future, experts, config, folder):
    results = {}
    prior = fit_global(train, experts, .001)
    for name in LEARNED:
        spec = selected[name]
        result = fit_one(name, spec['parameters'], train, future, experts, config,
                         amount=spec['amount'], prior=prior)
        save(result, Path(folder)/name)
        replayed = predict_saved(Path(folder)/name, future, config)
        np.testing.assert_allclose(replayed, result['prediction'], rtol=1e-6, atol=2e-6)
        results[name] = replayed
    return results


def event_library(train):
    keys = ['station_id', 'night', 'dt3', 'rel']
    columns = keys+['uplift', 'event_days']
    e = train.loc[train.ev.eq(1) & train.ev_rel.notna()].copy()
    if e.empty:
        return pd.DataFrame(columns=columns)
    e['origin'] = e.date-pd.to_timedelta(e.horizon, unit='D')
    e = e.sort_values('origin').drop_duplicates(KEYS, keep='last')
    e['rel'] = e.ev_rel.round().astype(int)
    e['night'] = (e.hour-e.ev_rel >= 19).astype(int)
    e['excess'] = e.y-e.base_v2
    e = e.loc[e.rel.between(-3, 2)]
    tables = []
    for group in [keys, ['station_id', 'night', 'rel']]:
        t = e.groupby(group, as_index=False).agg(median=('excess', 'median'), event_days=('date', 'nunique'))
        t = t.loc[t.event_days >= 3].copy()
        t['uplift'] = t['median'].clip(lower=0)*t.event_days/(t.event_days+3)
        if 'dt3' not in t:
            t['dt3'] = -1
        tables.append(t[columns])
    return pd.concat(tables, ignore_index=True)


def event_uplift(frame, library):
    if library.empty:
        return np.zeros(len(frame)), np.zeros(len(frame), dtype=bool)
    keys = ['station_id', 'night', 'dt3', 'rel']
    query = frame[['station_id', 'dt3']].copy()
    query['night'] = (frame.hour-frame.ev_rel >= 19).astype(int)
    query['rel'] = frame.ev_rel.fillna(999).round().astype(int)
    table = library.set_index(keys)
    a = table.reindex(pd.MultiIndex.from_frame(query[keys]))
    query['dt3'] = -1
    b = table.reindex(pd.MultiIndex.from_frame(query[keys]))
    value = np.nan_to_num(np.where(a.uplift.notna(), a.uplift, b.uplift), nan=0.)
    count = np.nan_to_num(np.where(a.event_days.notna(), a.event_days, b.event_days), nan=0.)
    gate = frame.ev.eq(1).to_numpy() & (count >= 3) & (value > 0)
    return value, gate
