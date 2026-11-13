"""Shared, portable utilities for the v4 laboratory."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import random
import sys

import numpy as np

VERSION = 'sincon-lab-v4.2'
KEYS = ['date', 'station_id', 'hour']
HORIZON = 123
LAST_LABEL = '2026-06-30'
SPLITS = {
    'calibration': '2025-10-27',
    'evaluation': '2026-02-27',
    'final': '2026-06-30',
}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def versions():
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'catboost',
                'xgboost', 'torch', 'neuralforecast', 'pytorch-lightning', 'torchmetrics', 'pyarrow']
    result = {'python': sys.version, 'platform': sys.platform}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'NOT INSTALLED'
    return result


def seed_all(seed=42, threads=2):
    if seed != 42:
        raise ValueError('이 실험실의 비교 시드는 42로 고정합니다.')
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def log_predictions(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('예측에 NaN 또는 무한대가 있습니다.')
    return np.maximum(values, 0.)


def rmsle(actual_log, prediction_log):
    a, b = np.asarray(actual_log, dtype=np.float64), log_predictions(prediction_log)
    if a.shape != b.shape or not np.isfinite(a).all() or a.size == 0:
        raise ValueError('평가 정답의 크기/값을 확인하세요.')
    return float(np.sqrt(np.mean(np.square(a - b))))
