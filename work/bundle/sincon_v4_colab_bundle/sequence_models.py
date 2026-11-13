"""NHITS/TFT/TiDE with a 123-day monitor and best-weight restoration."""
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS, TFT, TiDE

from common import HORIZON, KEYS, dump, read_json, rmsle, log_predictions, seed_all


class ClippedLogMSE(MSE):
    def __call__(self, y, y_hat, mask=None, y_insample=None):
        return super().__call__(y=y, y_hat=y_hat.clamp_min(0), mask=mask)


class BestValidationState(Callback):
    def __init__(self):
        super().__init__()
        self.rows, self.best, self.best_step, self.weights = [], np.inf, 0, None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get('ptl/val_loss')
        if value is None:
            return
        score = float(value.detach().cpu())**.5
        step = int(trainer.global_step)
        self.rows.append(dict(step=step, validation_rmsle=score))
        if score < self.best:
            self.best, self.best_step = score, step
            self.weights = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}


def align_predictions(nf, history, future, static, fold, name):
    forecast = nf.predict(df=history, futr_df=future, static_df=static, verbose=False)
    if 'unique_id' not in forecast:
        forecast = forecast.reset_index()
    codes = forecast.unique_id.str.split('_', expand=True)
    forecast['station_id'], forecast['hour'] = codes[0].astype(int), codes[1].astype(int)
    forecast = forecast.rename(columns={'ds': 'date', name: 'prediction'})
    aligned = fold.future[KEYS + ['lvl8']].merge(forecast[KEYS+['prediction']], on=KEYS,
                                               how='left', validate='one_to_one')
    fallback = aligned.prediction.isna()
    return log_predictions(aligned.prediction.fillna(aligned.lvl8)), int(fallback.sum())


def fit(name, params, fold, engine, config, monitor=False, amount=None):
    seed_all(config['seed'], config['threads'])
    history, future, static, exog = engine.sequence_inputs(fold, config)
    if not monitor and amount is None:
        raise ValueError('시계열 재학습에는 검증으로 선택한 학습량을 지정해야 합니다.')
    cap = int(config['max_steps'] if monitor else amount)
    check = min(int(config['check_steps']), cap)
    callback = BestValidationState()
    n_series = history.unique_id.nunique()
    valid_batch = max(i for i in range(1, min(32, n_series)+1) if n_series % i == 0)
    architecture = {k: v for k, v in params.items() if k not in ['input_size', 'width']}
    if name == 'nhits':
        architecture['mlp_units'] = [[params['width'], params['width']]]*3
    if name == 'tft':
        architecture['n_head'] = 4
    common = dict(h=HORIZON, input_size=params['input_size'], futr_exog_list=exog,
                  stat_exog_list=['sid_code', 'hour_code', 'line_code'], loss=MSE(),
                  valid_loss=ClippedLogMSE(), max_steps=cap, val_check_steps=check,
                  early_stop_patience_steps=config['deep_patience'] if monitor else -1,
                  num_lr_decays=-1, scaler_type='robust', start_padding_enabled=True,
                  batch_size=config['neural_batch_size'], valid_batch_size=valid_batch,
                  windows_batch_size=config['neural_windows_batch_size'], inference_windows_batch_size=32,
                  random_seed=42, alias=name, accelerator='gpu' if config['use_gpu'] and torch.cuda.is_available() else 'cpu',
                  devices=1, precision='32-true', logger=False, enable_checkpointing=False,
                  enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
                  deterministic='warn', callbacks=[callback] if monitor else [])
    model = {'nhits': NHITS, 'tft': TFT, 'tide': TiDE}[name](**architecture, **common)
    nf = NeuralForecast(models=[model], freq='D')
    started = time.perf_counter()
    frame = history
    if monitor:
        labels = fold.future[KEYS+['y']].copy()
        labels['unique_id'] = labels.station_id.astype(str)+'_'+labels.hour.astype(str)
        validation = future.merge(labels.rename(columns={'date': 'ds'})[['unique_id', 'ds', 'y']],
                                  on=['unique_id', 'ds'], how='left', validate='one_to_one')
        assert validation.y.notna().all()
        frame = pd.concat([history, validation], ignore_index=True)
    # val_size masks the last 123 days out of EVERY training window. A separate
    # history-only DataFrame is always passed again for forecast inference.
    nf.fit(df=frame, static_df=static, val_size=HORIZON if monitor else 0, verbose=False)
    if monitor:
        # NeuralForecast clones the initialized model (including its callbacks).
        # Read the callback attached to the fitted clone, not the original one.
        callback = next(c for c in nf.models[0].trainer_kwargs['callbacks'] if isinstance(c, BestValidationState))
        if callback.weights is None or not math.isfinite(callback.best):
            raise ValueError(f'{name}: 중간 검증 체크포인트가 없습니다.')
        nf.models[0].load_state_dict(callback.weights)
        selected = callback.best_step
        stopped = max(r['step'] for r in callback.rows)
    else:
        selected, stopped = cap, cap
    pred, fallback = align_predictions(nf, history, future, static, fold, name)
    # Callback state is not part of the inference artifact.
    nf.models[0].trainer_kwargs['callbacks'] = []
    nf.models[0].hparams['callbacks'] = []
    recipe = dict(name=name, backend='neuralforecast', parameters=params, amount=selected,
                  stopped=stopped, hit_cap=bool(monitor and stopped >= cap),
                  selected_rmsle=rmsle(fold.future.y, pred) if monitor else None,
                  checkpoint_metric='RMSLE, eligible series; level fallback is constant across checkpoints',
                  seconds=time.perf_counter()-started, horizon=HORIZON, num_lr_decays=-1,
                  future_exogenous=exog, static_features=['sid_code', 'hour_code', 'line_code'],
                  feature_policy='daily log passenger history + v4 calendar/weather/static (not the 55-column row matrix)',
                  cold_or_short_series_fallback_rows=fallback, best_weights_restored=bool(monitor),
                  fit_validation_size=HORIZON if monitor else 0,
                  inference_passenger_max=str(fold.origin.date()),
                  start_padding_enabled=True)
    return dict(model=nf, prediction=pred, recipe=recipe, curve=pd.DataFrame(callback.rows))


def save(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result['model'].save(path=str(folder/'neuralforecast'), overwrite=True, save_dataset=False)
    dump(folder/'recipe.json', result['recipe'])
    if len(result['curve']):
        result['curve'].to_csv(folder/'learning_curve.csv', index=False)


def predict_saved(folder, fold, engine, config):
    folder = Path(folder)
    recipe = read_json(folder/'recipe.json')
    nf = NeuralForecast.load(path=str(folder/'neuralforecast'))
    for model in nf.models:
        model.trainer_kwargs.update(accelerator='gpu' if config['use_gpu'] and torch.cuda.is_available() else 'cpu',
                                    devices=1, logger=False, enable_progress_bar=False, callbacks=[])
    history, future, static, _ = engine.sequence_inputs(fold, config)
    return align_predictions(nf, history, future, static, fold, recipe['name'])[0]
