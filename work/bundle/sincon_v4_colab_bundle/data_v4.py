"""v4-only inputs; passenger history is physically truncated before features."""
from dataclasses import dataclass
from pathlib import Path
import contextlib
import importlib.util
import io
import warnings

import numpy as np
import pandas as pd

from common import KEYS, HORIZON, LAST_LABEL, dump, sha


def import_file(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Fold:
    name: str
    origin: pd.Timestamp
    train: pd.DataFrame
    future: pd.DataFrame
    history: pd.DataFrame
    features: list
    cube: object
    metadata: dict


class DataEngine:
    REQUIRED = ['preprocess.py', 'train_processed.csv.gz', 'test_processed.csv.gz', 'weather_incheon.csv']

    def __init__(self, data_dir, source_dir=None):
        self.data_dir = Path(data_dir).resolve()
        self.source_dir = Path(source_dir or Path(__file__).parent)
        for name in self.REQUIRED:
            if not (self.data_dir / name).is_file():
                raise FileNotFoundError(f'v4 입력 파일이 없습니다: {self.data_dir / name}')
        self.prep = import_file(self.data_dir / 'preprocess.py', 'v4_preprocessing_source')
        self.v2 = import_file(self.source_dir / 'vendor/sincon_v2.py', 'sincon_structural_source')
        self.raw = pd.read_csv(self.data_dir / 'train_processed.csv.gz', parse_dates=['date'])
        self.test = pd.read_csv(self.data_dir / 'test_processed.csv.gz', parse_dates=['date'])
        if self.raw.date.max() != pd.Timestamp(LAST_LABEL):
            raise ValueError('학습 승차량의 마지막 날짜는 2026-06-30이어야 합니다.')
        if self.raw.date.min() != self.prep.START or self.raw.duplicated(KEYS).any():
            raise ValueError('v4 학습 날짜/행 키를 확인하세요.')
        if any(c in self.test for c in ['y', 'boardings', 'alightings', 'log_boardings', 'log_alightings']):
            raise ValueError('테스트 입력에 실제 이용객 정답 열이 있습니다.')
        if len(self.test) != 157194 or not self.test.example_id.is_unique:
            raise ValueError('v4 테스트 행 수/ID가 다릅니다.')
        self.meta = pd.concat([self.raw, self.test], ignore_index=True)[
            ['station_id', 'station_name', 'canonical_line']].drop_duplicates().sort_values('station_id')
        if len(self.meta) != 71 or not self.meta.station_id.is_unique:
            raise ValueError('역 메타 정보는 71개 역이어야 합니다.')
        self.stations = self.meta.station_id.to_numpy(dtype=int)
        self.cal = self.prep.build_calendar()
        self.input_hashes = {name: sha(self.data_dir / name) for name in self.REQUIRED}
        self.input_hashes['vendor/sincon_v2.py'] = sha(self.source_dir / 'vendor/sincon_v2.py')
        # The supplied target-derived flags are isolated from all input builders.
        self.diagnostics = self.raw[KEYS + ['is_event']].rename(columns={'is_event': 'diagnostic_event'})
        self.passengers = self.raw[KEYS + ['boardings', 'alightings']].copy()

    def cube_at(self, origin):
        origin = pd.Timestamp(origin)
        if origin > pd.Timestamp(LAST_LABEL):
            raise ValueError('이용객 기록은 2026-06-30까지만 읽을 수 있습니다.')
        prefix = self.passengers.loc[self.passengers.date.le(origin)].copy()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube = self.prep.Cube(prefix, self.test[KEYS], self.meta, self.cal)
            self.prep.attach_weather(cube, self.data_dir / 'weather_incheon.csv')
        c = int((origin - self.prep.START).days)
        assert np.isnan(cube.Y[c + 1:]).all(), 'Passenger values escaped the cutoff boundary'
        return cube, prefix

    def _cold_start(self, frame):
        out = frame.copy()
        missing = out.lvl8.isna()
        if missing.any():
            # Forecast inputs only: same-day type/hour peers, then all-line peers.
            reference = out.loc[~missing]
            means = reference.groupby(['date', 'line', 'hour']).lvl8.median()
            values = means.reindex(pd.MultiIndex.from_frame(out.loc[missing, ['date', 'line', 'hour']])).to_numpy()
            broad = reference.groupby(['date', 'hour']).lvl8.median().reindex(
                pd.MultiIndex.from_frame(out.loc[missing, ['date', 'hour']])).to_numpy()
            out.loc[missing, 'lvl8'] = np.where(np.isfinite(values), values, broad)
            out.loc[missing, 'lvl8_filled'] = 1.
        if not np.isfinite(out.lvl8).all():
            raise ValueError('과거 기록만으로 기준 수준을 만들 수 없습니다.')
        return out, int(missing.sum())

    def build(self, name, origin, config, with_train=True):
        origin = pd.Timestamp(origin)
        end = origin + pd.Timedelta(days=HORIZON)
        if name != 'final' and end > pd.Timestamp(LAST_LABEL):
            raise ValueError('검증 정답은 2026-06-30 이전으로 한정합니다.')
        cube, history = self.cube_at(origin)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            future = self.prep.cutoff_table(self.prep.build_rows(
                cube, origin, origin + pd.Timedelta(days=1), end,
                require_target=False, avail_end=origin), cube).drop(columns='y')
            train = self.prep.cutoff_table(self.prep.make_training_set(cube, origin), cube) if with_train else pd.DataFrame()
        future, cold = self._cold_start(future)
        # For historical periods, unknown events are absent in the prefix cube.
        if name != 'final':
            assert future.ev.eq(0).all() and future.ev_rel.isna().all()
            # Attach labels strictly AFTER building forecast covariates.
            labels = self.passengers.loc[self.passengers.date.gt(origin) & self.passengers.date.le(end),
                                         KEYS + ['boardings']].copy()
            labels['y'] = np.log1p(labels.pop('boardings'))
            future = future.merge(labels, on=KEYS, how='inner', validate='one_to_one')
            future = future.merge(self.diagnostics, on=KEYS, how='left', validate='one_to_one')
        else:
            future = self.test[['example_id'] + KEYS].merge(future, on=KEYS, how='left', validate='one_to_one')
            assert future.example_id.tolist() == self.test.example_id.tolist()
            future['diagnostic_event'] = future.ev.astype(int)
        features = [c for c in future if c not in ['date', 'station_id', 'y', 'example_id', 'diagnostic_event']]
        assert len(features) == 55, (len(features), features)
        assert not set(features) & {'y', 'boardings', 'alightings', 'diagnostic_event'}
        if with_train:
            assert train.date.max() <= origin and (train.horizon > 0).all()
            assert np.isfinite(train[['y', 'lvl8']]).all().all()
            assert not train.assign(origin=train.date-pd.to_timedelta(train.horizon, unit='D')).duplicated(['origin'] + KEYS).any()
        assert future.date.nunique() == HORIZON and not future.duplicated(KEYS).any()
        metadata = dict(name=name, origin=str(origin.date()), start=str((origin+pd.Timedelta(days=1)).date()),
                        end=str(end.date()), days=HORIZON, features=features, feature_count=len(features),
                        rows=len(future), stations=future.station_id.nunique(), train_rows=len(train),
                        cold_start_rows=cold, history_max=str(history.date.max().date()),
                        no_passengers_after_origin=True, historical_events='unknown => ev=0, ev_rel=NaN',
                        weather='v4 measured weather, allowed future exogenous information',
                        expected_missing_targets='new stations before opening; no invented zero targets')
        if config.get('smoke'):
            stations = config['smoke_stations']
            future = future.loc[future.station_id.isin(stations)].reset_index(drop=True)
            if with_train:
                train = train.loc[train.station_id.isin(stations)].groupby('station_id', group_keys=False).tail(
                    config['smoke_train_rows']).reset_index(drop=True)
            metadata.update(smoke=True, scored_rows=len(future), fitted_rows=len(train))
        return Fold(name, origin, train, future, history, features, cube, metadata)

    def structural_values(self, fold, frame, clean=True):
        origins = frame.date - pd.to_timedelta(frame.horizon, unit='D')
        values = np.empty(len(frame), dtype=np.float32)
        si = frame.station_id.map(fold.cube.sidx).to_numpy(dtype=int)
        hi = frame.hour.to_numpy(dtype=int) - 6
        for origin in sorted(origins.unique()):
            c = int((pd.Timestamp(origin) - self.prep.START).days)
            prefix_cube, _ = self.cube_at(origin)
            history = prefix_cube.Y[:c+1].copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                model = self.v2.Forecaster(history, use_events=False)
                if clean:
                    model.Y = history.copy()
                    model.Y[prefix_cube.ev[:c+1]] = np.nan
                    model._prepare()
                ix = np.flatnonzero(origins.eq(origin).to_numpy())
                steps = (frame.date.iloc[ix] - pd.Timestamp(origin)).dt.days.to_numpy(dtype=int)
                days = np.arange(c+1, c+int(steps.max())+1)
                pred = model.predict(days)
                values[ix] = pred[steps-1, si[ix], hi[ix]]
        return np.where(np.isfinite(values), values, frame.lvl8.to_numpy()).astype(np.float32)

    def augment_structure(self, fold):
        for frame in [fold.train, fold.future]:
            if len(frame) and 'base_v2' not in frame:
                frame['base_v2'] = self.structural_values(fold, frame)
                frame['base_minus_lvl8'] = frame.base_v2 - frame.lvl8
        return fold

    def sequence_inputs(self, fold, config):
        # Neural models get daily station-hour series, the same v4 calendar and
        # weather source, and static station/hour/line codes. No held-out counts.
        raw = fold.history.copy()
        if config.get('smoke'):
            raw = raw.loc[raw.station_id.isin(config['smoke_stations'])]
        raw['unique_id'] = raw.station_id.astype(str) + '_' + raw.hour.astype(str)
        eligible = raw.groupby('unique_id').size()
        eligible = eligible.index[eligible >= HORIZON + 1]
        if config.get('smoke'):
            eligible = eligible[[0, -1]]
        raw = raw.loc[raw.unique_id.isin(eligible)].copy()
        if raw.empty:
            raise ValueError('123일 시계열 모델을 학습할 과거 기록이 부족합니다.')
        pairs = raw[['unique_id', 'station_id', 'hour']].drop_duplicates()
        dates = pd.DataFrame({'date': pd.date_range(fold.origin+pd.Timedelta(days=1), periods=HORIZON)})
        future = dates.merge(pairs, how='cross')
        external = self.prep.CAL_FEATS + self.prep.WEATHER_FEATS

        def attach(frame):
            out = frame.merge(self.cal[['date'] + self.prep.CAL_FEATS], on='date', validate='many_to_one')
            di = (out.date-self.prep.START).dt.days.to_numpy(dtype=int)
            hi = out.hour.to_numpy(dtype=int)-6
            for name in self.prep.WEATHER_FEATS:
                out[name] = fold.cube.W[name][di, hi]
            # Only independently supplied test schedules enter future exogenous
            # features. Historical auto-detected events are NOT future inputs.
            out['scheduled_event'] = 0.
            out['scheduled_rel'] = 0.
            if fold.name == 'final':
                ev = fold.future[KEYS + ['ev', 'ev_rel']]
                out = out.merge(ev, on=KEYS, how='left', validate='one_to_one')
                out['scheduled_event'] = out.pop('ev').fillna(0)
                out['scheduled_rel'] = out.pop('ev_rel').fillna(0)
            return out

        raw, future = attach(raw), attach(future)
        # Do not train constant-zero historical schedule columns then introduce
        # unseen positive values at prediction time. Event experts handle them.
        raw['y'] = np.log1p(raw.boardings)
        history = raw.rename(columns={'date': 'ds'})[['unique_id', 'ds', 'y'] + external]
        future = future.rename(columns={'date': 'ds'})[['unique_id', 'ds'] + external]
        static = pairs.merge(self.meta[['station_id', 'canonical_line']], on='station_id', validate='many_to_one')
        static['sid_code'] = static.station_id.map(fold.cube.sidx) / 70
        static['hour_code'] = (static.hour-6)/17
        static['line_code'] = static.canonical_line / self.meta.canonical_line.max()
        static = static[['unique_id', 'sid_code', 'hour_code', 'line_code']]
        assert history.ds.max() <= fold.origin < future.ds.min()
        assert np.isfinite(history[external+['y']]).all().all() and np.isfinite(future[external]).all().all()
        return history, future, static, external
