"""Finite, reproducible parameter spaces; the default is a budgeted search."""
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import numpy as np

CORE = ['baseline', 'histgb_basic', 'histgb', 'histgb_tuned', 'catboost',
        'lightgbm', 'lightgbm_fixed500', 'ridge', 'xgboost', 'nhits', 'tft',
        'ft_transformer', 'tide']
MODULAR = ['structure_v2', 'structure_residual_lgb', 'structure_residual_blend', 'role_ensemble']
LEARNED = ['learned_global_weights', 'ridge_stacking', 'context_gate']
FIXED = {
    'ensemble_hgb_cat': [('histgb', .3), ('catboost', .7)],
    'ensemble_hgb_ridge': [('histgb', .7), ('ridge', .3)],
    'ensemble_cat_lgb': [('catboost', .25), ('lightgbm', .75)],
    'ensemble_current_lgb': [('ensemble_hgb_cat', .5), ('lightgbm', .5)],
    'ensemble_lgb90_tft10': [('lightgbm', .9), ('tft', .1)],
}
EXPERTS = ['lightgbm', 'catboost', 'histgb', 'xgboost', 'ridge', 'nhits', 'tft',
           'ft_transformer', 'tide', 'structure_v2', 'structure_residual_lgb',
           'structure_residual_blend', 'role_ensemble']

SPACES = {
    'histgb_basic': {'learning_rate': [.025, .05, .08], 'max_leaf_nodes': [31, 63, 127],
                     'min_samples_leaf': [60, 120], 'l2_regularization': [1., 5.]},
    'histgb': {'learning_rate': [.02, .04, .06], 'max_leaf_nodes': [31, 63, 127],
               'min_samples_leaf': [80, 150], 'l2_regularization': [1., 5.]},
    'histgb_tuned': {'learning_rate': [.015, .025, .05], 'max_leaf_nodes': [31, 63, 127],
                     'min_samples_leaf': [60, 120], 'l2_regularization': [2., 8.]},
    'lightgbm': {'learning_rate': [.025, .05, .08], 'num_leaves': [15, 31, 63],
                 'min_child_samples': [80, 150], 'reg_lambda': [2., 10.],
                 'colsample_bytree': [.8, 1.]},
    'catboost': {'learning_rate': [.025, .05, .08], 'depth': [4, 6, 8],
                 'l2_leaf_reg': [3., 10.]},
    'xgboost': {'learning_rate': [.025, .05, .08], 'max_depth': [4, 6, 8],
                'min_child_weight': [50, 100], 'reg_lambda': [5., 15.]},
    'ridge': {'alpha': [.1, 1., 10., 100., 1000., 10000.]},
    'ft_transformer': {'learning_rate': [.0003, .001], 'd_token': [32, 64],
                       'n_layers': [1, 2], 'dropout': [.1, .2], 'weight_decay': [.00001, .001]},
    'nhits': {'learning_rate': [.0003, .001], 'input_size': [56, 84],
               'width': [64, 128], 'dropout_prob_theta': [0., .1]},
    'tft': {'learning_rate': [.0003, .001], 'input_size': [56, 84],
             'hidden_size': [16, 32, 64], 'dropout': [.1, .2]},
    'tide': {'learning_rate': [.0003, .001], 'input_size': [56, 84],
              'hidden_size': [64, 128], 'decoder_output_dim': [8, 16],
              'dropout': [.1, .3], 'num_encoder_layers': [1, 2]},
    'learned_global_weights': {'regularization': [0., .00001, .0001, .001, .01, .1]},
    'ridge_stacking': {'alpha': [.1, 1., 10., 100., 1000., 10000.]},
    'context_gate': {'hidden_size': [0, 16, 32], 'learning_rate': [.001, .003],
                     'regularization': [.001, .01], 'mix_strength': [.25, .5, 1.]},
}
DEFAULTS = {
    'histgb_basic': {'learning_rate': .05, 'max_leaf_nodes': 127, 'min_samples_leaf': 80, 'l2_regularization': 1.},
    'histgb': {'learning_rate': .04, 'max_leaf_nodes': 127, 'min_samples_leaf': 80, 'l2_regularization': 1.},
    'histgb_tuned': {'learning_rate': .025, 'max_leaf_nodes': 127, 'min_samples_leaf': 60, 'l2_regularization': 2.},
    'lightgbm': {'learning_rate': .05, 'num_leaves': 31, 'min_child_samples': 100, 'reg_lambda': 5., 'colsample_bytree': .9},
    'catboost': {'learning_rate': .05, 'depth': 6, 'l2_leaf_reg': 3.},
    'xgboost': {'learning_rate': .05, 'max_depth': 6, 'min_child_weight': 100, 'reg_lambda': 5.},
    'ridge': {'alpha': 10.},
    'ft_transformer': {'learning_rate': .001, 'd_token': 32, 'n_layers': 2, 'dropout': .1, 'weight_decay': .00001},
    'nhits': {'learning_rate': .001, 'input_size': 56, 'width': 128, 'dropout_prob_theta': 0.},
    'tft': {'learning_rate': .001, 'input_size': 56, 'hidden_size': 32, 'dropout': .1},
    'tide': {'learning_rate': .001, 'input_size': 56, 'hidden_size': 128,
             'decoder_output_dim': 16, 'dropout': .1, 'num_encoder_layers': 1},
    'learned_global_weights': {'regularization': .001},
    'ridge_stacking': {'alpha': 10.},
    'context_gate': {'hidden_size': 16, 'learning_rate': .001, 'regularization': .01, 'mix_strength': .5},
}


def default_config():
    return dict(seed=42, threads=2, use_gpu=True, models=CORE.copy(), include_modular=True,
                search_mode='random', trials_per_model=6, meta_trials=6,
                spaces=deepcopy(SPACES), max_rounds=2000, boost_patience=100,
                hgb_check_every=25, max_steps=2000, check_steps=100, deep_patience=5,
                batch_size=512, inference_batch_size=2048, neural_batch_size=16,
                neural_windows_batch_size=32, meta_train_days=82,
                gate_max_steps=1500, gate_check_steps=50, gate_patience=5,
                final_refit=True, final_models='winner', allow_model_failures=False,
                smoke=False, smoke_stations=[3110, 3127], smoke_train_rows=5000)


def _unique(specifications):
    result = []
    for specification in specifications:
        if specification not in result:
            result.append(deepcopy(specification))
    return result


def candidates(name, config, meta=False):
    """Cover every LR first, then balance unique trials across LR values.

    For a dict space, the first sweep changes only LR from the original preset.
    An explicit list keeps its exact configurations; no new ones are invented.
    Non-smoke runs reject budgets too small for complete LR coverage.
    """
    key = 'lightgbm' if name == 'role_lgb' else 'catboost' if name == 'role_cat' else name
    base = deepcopy(DEFAULTS[key])
    raw = config['spaces'].get(key)
    if raw is None:
        raise ValueError(f'{key}: 탐색 공간이 없습니다.')
    explicit = isinstance(raw, list)
    # Explicit lists contain exact configurations/overrides, in user order.
    if isinstance(raw, list):
        if not raw or any(not isinstance(v, dict) or set(v)-set(SPACES[key]) for v in raw):
            raise ValueError(f'{key}: 후보 사전 목록과 허용 파라미터를 확인하세요.')
        pool = [{**base, **v} for v in raw]
    elif isinstance(raw, dict):
        unknown = set(raw) - set(SPACES[key])
        if unknown:
            raise ValueError(f'{key}: 지원하지 않는 파라미터 {sorted(unknown)}')
        pool = [{**base, **v} for v in ParameterGrid(raw)]
    else:
        raise ValueError('탐색 공간은 사전 또는 후보 사전들의 목록이어야 합니다.')
    limit = int(config['meta_trials' if meta else 'trials_per_model'])
    if limit < 1:
        raise ValueError('후보 수는 1 이상이어야 합니다.')
    pool = _unique(pool)
    if config['search_mode'] == 'grid':
        return _unique([base] + pool)
    if config['search_mode'] != 'random':
        raise ValueError('search_mode는 random 또는 grid입니다.')
    rng = np.random.default_rng(42)
    if 'learning_rate' not in base:
        ordered = pool if explicit else [pool[int(i)] for i in rng.permutation(len(pool))]
        return _unique([base]+ordered)[:limit]

    rates = []
    for p in [base]+pool:
        rate = p['learning_rate']
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f'{key}: learning_rate는 양수여야 합니다.')
        if rate not in rates:
            rates.append(rate)
    if limit < len(rates) and not config.get('smoke', False):
        field = 'META_TRIALS' if meta else 'TRIALS_PER_MODEL'
        raise ValueError(f'{key}: 원래 설정을 포함한 learning rate {len(rates)}종을 '
                         f'모두 비교하려면 {field}를 {len(rates)} 이상으로 설정하세요. 현재 {limit}.')
    if explicit:
        universe = _unique([base]+pool)
        first = [base]+[next(p for p in universe if p['learning_rate'] == rate) for rate in rates[1:]]
    else:
        # Controlled LR comparison: architecture and regularization stay fixed.
        first = [{**base, 'learning_rate': rate} for rate in rates]
        universe = _unique(first+pool)
    budget = min(limit, len(universe))
    chosen = _unique(first)[:budget]
    ordered = universe if explicit else [universe[int(i)] for i in rng.permutation(len(universe))]
    remaining = [p for p in ordered if p not in chosen]
    while len(chosen) < budget:
        available = [rate for rate in rates if any(p['learning_rate'] == rate for p in remaining)]
        rate = min(available, key=lambda r: (sum(p['learning_rate'] == r for p in chosen), rates.index(r)))
        index = next(i for i,p in enumerate(remaining) if p['learning_rate'] == rate)
        chosen.append(remaining.pop(index))
    return chosen
