"""Tree/linear experts, learning curves, and replayable selected models."""
from pathlib import Path
import copy
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from common import dump, log_predictions, rmsle


def model_inputs(name, fold):
    modular = name in ['role_lgb', 'role_cat']
    names = ([c for c in fold.features if c not in ['ev', 'ev_rel']] +
             ['base_v2', 'base_minus_lvl8']) if modular else fold.features
    train = fold.train.loc[fold.train.ev.eq(0)] if modular else fold.train
    base_name = 'base_v2' if modular else 'lvl8'
    X = train[names].to_numpy(dtype=np.float32)
    V = fold.future[names].to_numpy(dtype=np.float32)
    y = (train.y-train[base_name]).to_numpy(dtype=float)
    return X, y, V, fold.future[base_name].to_numpy(dtype=float), names, base_name


def factory(name, params, names, config, rounds):
    seed, threads = config['seed'], config['threads']
    kind = 'lightgbm' if name in ['role_lgb', 'lightgbm_fixed500'] else 'catboost' if name == 'role_cat' else name
    if kind.startswith('histgb'):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**params, max_iter=rounds,
            categorical_features=[names.index('sid')], early_stopping=False, random_state=seed)
    if kind == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, n_estimators=rounds, objective='regression',
            subsample=.8, subsample_freq=1, max_bin=127, deterministic=True,
            force_col_wise=True, random_state=seed, n_jobs=threads, verbosity=-1)
    if kind == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params, iterations=rounds, loss_function='RMSE',
            random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False)
    if kind == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(**params, n_estimators=rounds, objective='reg:squarederror',
            subsample=.8, colsample_bytree=.9, tree_method='hist', device='cpu',
            random_state=seed, n_jobs=threads)
    if kind == 'ridge':
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        return make_pipeline(SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True),
                             StandardScaler(), Ridge(**params))
    raise ValueError(name)


def fit(name, params, fold, config, monitor=False, amount=None):
    X, y, V, base, names, base_name = model_inputs(name, fold)
    if not monitor and amount is None:
        raise ValueError('재학습에는 검증으로 선택한 학습량을 지정해야 합니다.')
    cap = int(config['max_rounds'] if monitor else amount)
    patience = int(config['boost_patience'])
    kind = 'lightgbm' if name in ['role_lgb', 'lightgbm_fixed500'] else 'catboost' if name == 'role_cat' else name
    if name == 'lightgbm_fixed500':
        cap, monitor = (3 if config['smoke'] else 500), False
    model = factory(name, params, names, config, cap)
    curve, started = [], time.perf_counter()
    actual = fold.future.y.to_numpy() if monitor else None
    with threadpool_limits(limits=config['threads']):
        if kind.startswith('histgb') and monitor:
            model.set_params(warm_start=True)
            best, best_model, stale, selected = np.inf, None, 0, 0
            every = int(config['hgb_check_every'])
            checkpoints = sorted(set(list(range(min(every, cap), cap+1, every)) + [cap]))
            for step in checkpoints:
                model.set_params(max_iter=step)
                model.fit(X, y)
                value = rmsle(actual, base+model.predict(V))
                train_rmse = float(np.sqrt(np.mean((y-model.predict(X))**2)))
                curve.append(dict(step=step, train_residual_rmse=train_rmse, validation_rmsle=value))
                if value < best-1e-10:
                    best, best_model, stale, selected = value, copy.deepcopy(model), 0, step
                else:
                    stale += step-(curve[-2]['step'] if len(curve)>1 else 0)
                if stale >= patience:
                    break
            model = best_model
            model.set_params(warm_start=False)
            amount, stopped = selected, curve[-1]['step']
        elif kind == 'lightgbm' and monitor:
            import lightgbm as lgb
            record = {}
            def metric(y_true, predicted):
                return 'rmsle', rmsle(actual, base+predicted), False
            model.set_params(metric='None')
            model.fit(X, y, feature_name=names, categorical_feature=['sid'],
                eval_set=[(V, actual-base)], eval_names=['validation'], eval_metric=metric,
                callbacks=[lgb.record_evaluation(record), lgb.early_stopping(patience, first_metric_only=True, verbose=False),
                           lgb.log_evaluation(0)])
            values = record['validation']['rmsle']
            curve = [dict(step=i+1, validation_rmsle=float(v)) for i, v in enumerate(values)]
            amount, stopped = int(model.best_iteration_), len(values)
        elif kind == 'catboost' and monitor:
            model.fit(X, y, eval_set=(V, actual-base), early_stopping_rounds=patience, use_best_model=False)
            record = model.get_evals_result()
            rmse_values = record['validation']['RMSE']
            # Native stopping monitors log-residual RMSE; choose the checkpoint
            # by exact clipped RMSLE over ALL rounds actually reached.
            best, selected = np.inf, 1
            for i, pred in enumerate(model.staged_predict(V), start=1):
                score = rmsle(actual, base+pred)
                curve.append(dict(step=i, validation_rmsle=score,
                                  native_validation_rmse=float(rmse_values[i-1])))
                if score < best:
                    best, selected = score, i
            stopped, amount = model.tree_count_, selected
            model.shrink(ntree_end=selected)
        elif kind == 'xgboost' and monitor:
            def metric(y_true, predicted):
                return rmsle(actual, base+predicted)
            metric.__name__ = 'rmsle'
            model.set_params(eval_metric=metric, early_stopping_rounds=patience)
            model.fit(X, y, eval_set=[(V, actual-base)], verbose=False)
            values = model.evals_result()['validation_0']['rmsle']
            curve = [dict(step=i+1, validation_rmsle=float(v)) for i, v in enumerate(values)]
            amount, stopped = int(model.best_iteration)+1, len(values)
            # Persist only the selected trees and remove a nonserializable closure.
            model._Booster = model.get_booster()[:amount]
            model.set_params(eval_metric='rmse', early_stopping_rounds=None)
        else:
            if kind == 'lightgbm':
                model.fit(X, y, feature_name=names, categorical_feature=['sid'])
            else:
                model.fit(X, y)
            amount = 1 if kind == 'ridge' else cap
            stopped = amount
            if monitor:
                curve = [dict(step=amount, validation_rmsle=rmsle(actual, base+model.predict(V)))]
        pred = log_predictions(base+model.predict(V))
    recipe = dict(name=name, parameters=params, amount=int(amount), stopped=int(stopped),
                  hit_cap=bool(monitor and stopped >= cap), features=names, base=base_name,
                  backend='joblib', seconds=time.perf_counter()-started,
                  checkpoint_metric='clipped log RMSE (RMSLE)',
                  stopping_metric='log residual RMSE' if kind == 'catboost' else 'RMSLE',
                  selected_rmsle=rmsle(actual, pred) if monitor else None)
    return dict(model=model, prediction=pred, recipe=recipe, curve=pd.DataFrame(curve))


def save(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(result['model'], folder/'model.joblib', compress=3)
    dump(folder/'recipe.json', result['recipe'])
    if len(result['curve']):
        result['curve'].to_csv(folder/'learning_curve.csv', index=False)


def predict_saved(folder, frame):
    from common import read_json
    folder = Path(folder)
    recipe = read_json(folder/'recipe.json')
    model = joblib.load(folder/'model.joblib')
    with threadpool_limits(limits=2):
        result = frame[recipe['base']].to_numpy()+model.predict(frame[recipe['features']].to_numpy(dtype=np.float32))
    return log_predictions(result)
