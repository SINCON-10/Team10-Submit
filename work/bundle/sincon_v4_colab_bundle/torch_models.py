"""Small FT-Transformer adaptation and a context-dependent softmax gate."""
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import dump, read_json, rmsle, log_predictions, seed_all

CAT_SIZES = {'sid': 71, 'hour': 24, 'line': 10, 'dow': 7, 'month': 13, 'dt3': 3}
CONTEXT = ['sid', 'hour', 'line', 'dow', 'month', 'dt3', 'horizon', 'lvl8', 'cnt8',
           'is_hol', 'is_major', 'offday', 'univ_vac', 'school_vac', 'stn_age_cut',
           'w_temp', 'w_rain', 'w_rain3', 'w_rain_day', 'w_tmax', 'w_snow_day']


class TabularTransform:
    def fit(self, frame, names):
        self.names = list(names)
        self.cat = [c for c in names if c in CAT_SIZES]
        self.num = [c for c in names if c not in self.cat]
        self.cards = [CAT_SIZES[c] for c in self.cat]
        a = frame[self.num].to_numpy(dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.mean = np.nanmean(a, axis=0)
            self.std = np.nanstd(a, axis=0)
        self.mean = np.nan_to_num(self.mean, nan=0.)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-5), self.std, 1.)
        return self

    def transform(self, frame):
        a = frame[self.num].to_numpy(dtype=np.float32)
        missing = ~np.isfinite(a)
        a = np.clip(np.nan_to_num((a-self.mean)/self.std, nan=0.), -20, 20)
        cat = frame[self.cat].fillna(0).to_numpy(dtype=np.int64)
        for j, card in enumerate(self.cards):
            if not ((cat[:, j] >= 0) & (cat[:, j] < card)).all():
                raise ValueError(f'{self.cat[j]} 범주 코드가 범위를 벗어났습니다.')
        return a, cat, missing.astype(np.float32)

    def dense(self, frame):
        a, cat, missing = self.transform(frame)
        parts = [a, missing] + [np.eye(card, dtype=np.float32)[cat[:, j]] for j, card in enumerate(self.cards)]
        return np.concatenate(parts, axis=1)


class FTTransformer(nn.Module):
    def __init__(self, n_num, cards, d_token=32, n_layers=2, dropout=.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d_token)*.02)
        self.bias = nn.Parameter(torch.zeros(n_num, d_token))
        self.missing = nn.Parameter(torch.randn(n_num, d_token)*.02)
        self.embeddings = nn.ModuleList([nn.Embedding(n, d_token) for n in cards])
        self.cls = nn.Parameter(torch.randn(1, 1, d_token)*.02)
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=4, dim_feedforward=d_token*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, numeric, categorical, missing):
        tokens = [numeric.unsqueeze(-1)*self.weight + self.bias + missing.unsqueeze(-1)*self.missing]
        tokens += [embedding(categorical[:, j]).unsqueeze(1) for j, embedding in enumerate(self.embeddings)]
        x = torch.cat([self.cls.expand(len(numeric), -1, -1)] + tokens, dim=1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


class ContextGate(nn.Module):
    def __init__(self, input_dim, prior, hidden_size=16, mix_strength=.5):
        super().__init__()
        m = len(prior)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, m)) if hidden_size else nn.Linear(input_dim, m)
        final = self.net[-1] if hidden_size else self.net
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32))
        self.mix_strength = mix_strength

    def weights(self, x):
        flexible = torch.softmax(torch.log(self.prior.clamp_min(1e-5)) + self.net(x), dim=1)
        return (1-self.mix_strength)*self.prior + self.mix_strength*flexible

    def forward(self, x, experts):
        return (self.weights(x)*experts).sum(dim=1)


def device_for(config):
    return torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _ft_predict(model, arrays, base, device, batch_size):
    model.eval()
    result = []
    with torch.inference_mode():
        for lo in range(0, len(base), batch_size):
            tensors = [torch.as_tensor(a[lo:lo+batch_size], device=device) for a in arrays]
            result.append(model(*tensors).detach().cpu().numpy())
    return log_predictions(np.concatenate(result).astype(float)+base)


def fit_ft(params, fold, config, monitor=False, amount=None):
    seed_all(config['seed'], config['threads'])
    transform = TabularTransform().fit(fold.train, fold.features)
    arrays = transform.transform(fold.train)
    valid_arrays = transform.transform(fold.future)
    target = (fold.train.y-fold.train.lvl8).to_numpy(dtype=np.float32)
    base = fold.future.lvl8.to_numpy(dtype=float)
    architecture = {k: params[k] for k in ['d_token', 'n_layers', 'dropout']}
    device = device_for(config)
    model = FTTransformer(len(transform.num), transform.cards, **architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    if not monitor and amount is None:
        raise ValueError('FT 재학습에는 검증으로 선택한 학습량을 지정해야 합니다.')
    cap = int(config['max_steps'] if monitor else amount)
    check, patience = int(config['check_steps']), int(config['deep_patience'])
    rng = np.random.default_rng(42)
    best, best_step, best_state, stale, curve = np.inf, cap, None, 0, []
    started = time.perf_counter()
    for step in range(1, cap+1):
        model.train()
        ix = rng.integers(0, len(target), size=min(config['batch_size'], len(target)))
        tensors = [torch.as_tensor(a[ix], device=device) for a in arrays]
        expected = torch.as_tensor(target[ix], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model(*tensors)-expected).mean()
        if not torch.isfinite(loss):
            raise ValueError('FT-Transformer 학습 손실이 유한하지 않습니다.')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if monitor and (step % check == 0 or step == cap):
            predicted = _ft_predict(model, valid_arrays, base, device, config['inference_batch_size'])
            score = rmsle(fold.future.y, predicted)
            curve.append(dict(step=step, train_batch_rmse=float(loss.detach().sqrt().cpu()), validation_rmsle=score))
            print(f'  FT step {step}: 중간 검증 RMSLE={score:.6f}', flush=True)
            if score < best-1e-10:
                best, best_step, best_state, stale = score, step, cpu_state(model), 0
            else:
                stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    prediction = _ft_predict(model, valid_arrays, base, device, config['inference_batch_size'])
    recipe = dict(name='ft_transformer', backend='ft_transformer', parameters=params, architecture=architecture,
                  features=fold.features, base='lvl8', amount=best_step if monitor else cap,
                  stopped=step, hit_cap=bool(monitor and step >= cap), seconds=time.perf_counter()-started,
                  selected_rmsle=rmsle(fold.future.y, prediction) if monitor else None,
                  checkpoint_metric='RMSLE', device=str(device),
                  implementation='small FT-Transformer adaptation, feature tokens + missing tokens; residual target')
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(model=model, transform=transform, prediction=prediction, recipe=recipe, curve=pd.DataFrame(curve))


def save_ft(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(result['model']), folder/'model.pt')
    joblib.dump(result['transform'], folder/'transform.joblib')
    dump(folder/'recipe.json', result['recipe'])
    if len(result['curve']):
        result['curve'].to_csv(folder/'learning_curve.csv', index=False)


def predict_ft(folder, frame, config):
    folder = Path(folder)
    recipe = read_json(folder/'recipe.json')
    transform = joblib.load(folder/'transform.joblib')
    model = FTTransformer(len(transform.num), transform.cards, **recipe['architecture'])
    model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
    device = device_for(config)
    model.to(device)
    return _ft_predict(model, transform.transform(frame), frame[recipe['base']].to_numpy(), device, config['inference_batch_size'])


def gate_predict(model, X, Z, device, batch_size=8192, return_weights=False):
    model.eval()
    result = []
    with torch.inference_mode():
        for lo in range(0, len(X), batch_size):
            x = torch.as_tensor(X[lo:lo+batch_size], device=device)
            if return_weights:
                value = model.weights(x)
            else:
                z = torch.as_tensor(Z[lo:lo+batch_size], dtype=torch.float32, device=device)
                value = model(x, z)
            result.append(value.detach().cpu().numpy())
    values = np.concatenate(result)
    return values if return_weights else log_predictions(values)


def fit_gate(params, train, valid, experts, prior, config, monitor=True, amount=None):
    seed_all(config['seed'], config['threads'])
    transform = TabularTransform().fit(train, CONTEXT)
    X, V = transform.dense(train), transform.dense(valid)
    Z = train[experts].to_numpy(dtype=np.float32)
    Zvalid = valid[experts].to_numpy(dtype=np.float32)
    y = train.y.to_numpy(dtype=np.float32)
    architecture = dict(input_dim=X.shape[1], prior=list(np.asarray(prior, dtype=float)),
                        hidden_size=params['hidden_size'], mix_strength=params['mix_strength'])
    device = device_for(config)
    model = ContextGate(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=.001)
    cap = int(config['gate_max_steps'] if monitor else amount)
    best, best_step, best_state, stale, curve = np.inf, cap, None, 0, []
    rng = np.random.default_rng(42)
    for step in range(1, cap+1):
        ix = rng.integers(0, len(y), size=min(2048, len(y)))
        x = torch.as_tensor(X[ix], device=device)
        z = torch.as_tensor(Z[ix], device=device)
        target = torch.as_tensor(y[ix], device=device)
        optimizer.zero_grad(set_to_none=True)
        w = model.weights(x)
        loss = ((w*z).sum(1)-target).square().mean() + params['regularization']*(w-model.prior).square().sum(1).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if monitor and (step % config['gate_check_steps'] == 0 or step == cap):
            score = rmsle(valid.y, gate_predict(model, V, Zvalid, device))
            curve.append(dict(step=step, validation_rmsle=score))
            if score < best-1e-10:
                best, best_step, best_state, stale = score, step, cpu_state(model), 0
            else:
                stale += 1
            if stale >= config['gate_patience']:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    prediction = gate_predict(model, V, Zvalid, device)
    model.cpu()
    recipe = dict(name='context_gate', backend='context_gate', parameters=params, experts=experts,
                  architecture=architecture, amount=best_step if monitor else cap, stopped=step,
                  hit_cap=bool(monitor and step >= cap), features=CONTEXT,
                  selected_rmsle=rmsle(valid.y, prediction) if monitor else None,
                  method='FFORMA-inspired context gate; direct log-MSE objective, not a literal FFORMA reproduction')
    return dict(model=model, transform=transform, prediction=prediction, recipe=recipe, curve=pd.DataFrame(curve))


def predict_gate(folder, frame, config, return_weights=False):
    folder = Path(folder)
    recipe = read_json(folder/'recipe.json')
    transform = joblib.load(folder/'transform.joblib')
    model = ContextGate(**recipe['architecture'])
    model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
    device = device_for(config)
    model.to(device)
    return gate_predict(model, transform.dense(frame), frame[recipe['experts']].to_numpy(dtype=np.float32),
                        device, return_weights=return_weights)
