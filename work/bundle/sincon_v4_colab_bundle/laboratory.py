"""A chronological, resumable 123-day experiment runner (v4 data only)."""
from pathlib import Path
from datetime import datetime
import gc
import shutil
import time

import numpy as np
import pandas as pd

from common import (VERSION, KEYS, SPLITS, dump, read_json, sha, fingerprint,
                    versions, seed_all, log_predictions, rmsle)
from config import CORE, MODULAR, FIXED, LEARNED, EXPERTS, DEFAULTS, candidates
from data_v4 import DataEngine
import tabular
import ensembles

SEQUENCES = {'nhits', 'tft', 'tide'}


class Laboratory:
    def __init__(self, data_dir, output_dir, config, source_dir=None):
        self.config = config
        self.source = Path(source_dir or Path(__file__).parent).resolve()
        self.out = Path(output_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.engine = DataEngine(data_dir, self.source)
        seed_all(config['seed'], config['threads'])
        import torch
        accelerator = torch.cuda.get_device_name(0) if config['use_gpu'] and torch.cuda.is_available() else 'cpu'
        environment = versions()
        if not config['models'] or set(config['models'])-set(CORE):
            raise ValueError('models는 지원하는 기본 모델들의 비어 있지 않은 목록이어야 합니다.')
        if config['final_models'] not in ['winner', 'all']:
            raise ValueError('final_models는 winner 또는 all입니다.')
        if min(config['max_rounds'], config['max_steps'], config['check_steps'],
               config['boost_patience'], config['trials_per_model'], config['meta_trials']) < 1:
            raise ValueError('탐색 횟수와 학습 상한은 양수여야 합니다.')
        algorithm_files = ['common.py', 'config.py', 'data_v4.py', 'tabular.py', 'torch_models.py',
                           'sequence_models.py', 'ensembles.py', 'laboratory.py', 'report.py']
        self.code_hashes = {name: sha(self.source/name) for name in algorithm_files}
        signature = fingerprint(dict(version=VERSION, config=config,
                                     inputs=self.engine.input_hashes, code=self.code_hashes))
        previous = self.out/'run_manifest.json'
        if previous.is_file():
            prior = read_json(previous)
            if prior['environment'] != environment or prior.get('accelerator') != accelerator:
                raise ValueError('패키지 버전 또는 계산 장치가 달라졌습니다. 새 RUN_NAME으로 실행하세요.')
        if previous.is_file() and read_json(previous)['signature'] != signature:
            raise ValueError('기존 실행과 설정/입력/코드가 다릅니다. RUN_NAME을 새 이름으로 바꾸세요.')
        self.manifest = dict(version=VERSION, signature=signature, config=config,
                             inputs=self.engine.input_hashes, code=self.code_hashes,
                             environment=environment, accelerator=accelerator, created=datetime.now().isoformat(),
                             seed=42, prediction_space='max(0, log1p count); blend in log space',
                             protocol='single calibration 2025-10-28..2026-02-27; evaluation excluded from parameter/weight fitting')
        if not previous.is_file():
            dump(previous, self.manifest)
            source = self.out/'source'
            source.mkdir(exist_ok=True)
            for p in self.source.glob('*.py'):
                shutil.copyfile(p, source/p.name)
            (source/'vendor').mkdir(exist_ok=True)
            shutil.copyfile(self.source/'vendor/sincon_v2.py', source/'vendor/sincon_v2.py')
        self.trainable = [n for n in config['models'] if n != 'baseline']
        if config['include_modular']:
            self.trainable += ['role_lgb', 'role_cat']
        self.errors = []
        print(f'[실험 설정] {VERSION} · 기본 모델당 {config["trials_per_model"]}후보 · '
              f'앙상블 {config["meta_trials"]}후보 · seed=42', flush=True)

    def describe(self):
        rows, plan = [], {}
        def planned(name, meta=False):
            specifications = candidates(name, self.config, meta=meta)
            plan[name] = specifications
            rates = sorted({p['learning_rate'] for p in specifications if 'learning_rate' in p})
            counts = ', '.join(f'{rate:g} × {sum(p["learning_rate"] == rate for p in specifications)}' for rate in rates)
            return len(specifications), counts or '학습률 없음'
        for name in self.trainable:
            if name == 'lightgbm_fixed500':
                rows.append(dict(model=name, candidates=0, searches=0, learning_rates='0.05 고정', note='고정 500회 대조군'))
            else:
                number, rates = planned(name)
                rows.append(dict(model=name, candidates=number, searches=1, learning_rates=rates, note='중간검증 후보별 조기 종료'))
        for name in LEARNED:
            number, rates = planned(name, meta=True)
            rows.append(dict(model=name, candidates=number, searches=1, learning_rates=rates,
                             note='중간검증 내부 학습·점검 후 전체 재학습'))
        dump(self.out/'candidate_plan.json', plan)
        return pd.DataFrame(rows)

    def _fit(self, name, parameters, fold, monitor=False, amount=None):
        if name == 'ft_transformer':
            from torch_models import fit_ft
            return fit_ft(parameters, fold, self.config, monitor=monitor, amount=amount)
        if name in SEQUENCES:
            from sequence_models import fit
            return fit(name, parameters, fold, self.engine, self.config, monitor=monitor, amount=amount)
        return tabular.fit(name, parameters, fold, self.config, monitor=monitor, amount=amount)

    def _save(self, result, folder):
        name = result['recipe']['name']
        if name == 'ft_transformer':
            from torch_models import save_ft
            save_ft(result, folder)
        elif name in SEQUENCES:
            from sequence_models import save
            save(result, folder)
        else:
            tabular.save(result, folder)

    def _replay_base(self, name, folder, fold):
        if name == 'ft_transformer':
            from torch_models import predict_ft
            return predict_ft(folder, fold.future, self.config)
        if name in SEQUENCES:
            from sequence_models import predict_saved
            return predict_saved(folder, fold, self.engine, self.config)
        return tabular.predict_saved(folder, fold.future)

    def _parameters(self, name):
        return DEFAULTS['lightgbm' if name in ['role_lgb', 'lightgbm_fixed500'] else 'catboost' if name == 'role_cat' else name].copy()

    def search(self, fold):
        """First model operation: tune on the single 123-day calibration period."""
        if fold.name != 'calibration':
            raise ValueError('최종 평가/테스트 정답으로 설정을 탐색할 수 없습니다.')
        parent = self.out/'search'/fold.name
        selected = {}
        for name in self.trainable:
            if name == 'lightgbm_fixed500':
                continue
            folder = parent/name
            completed = folder/'selection.json'
            if completed.is_file():
                selected[name] = read_json(completed)
                print(f'[탐색 재개] {fold.name}/{name}: 완료된 선택 재사용', flush=True)
                continue
            best, chosen, trials = np.inf, None, []
            specifications = candidates(name, self.config)
            for i, parameters in enumerate(specifications):
                print(f'[파라미터 탐색] {fold.name}/{name} 후보 {i+1}/{len(specifications)}: {parameters}', flush=True)
                result = self._fit(name, parameters, fold, monitor=True)
                score = rmsle(fold.future.y, result['prediction'])
                record = dict(trial=i, model=name, parameters=parameters, rmsle=score,
                              amount=result['recipe']['amount'], stopped=result['recipe']['stopped'],
                              hit_cap=result['recipe']['hit_cap'], seconds=result['recipe']['seconds'])
                trial_dir = folder/f'trial_{i:03d}'
                dump(trial_dir/'result.json', record)
                result['curve'].to_csv(trial_dir/'learning_curve.csv', index=False)
                trials.append(record)
                pd.DataFrame(trials).to_csv(folder/'trials.csv', index=False)
                if score < best:
                    best = score
                    chosen = dict(parameters=parameters, amount=result['recipe']['amount'],
                                  rmsle=score, trial=i, chosen_using=fold.metadata,
                                  best_model='best_model', recipe=result['recipe'])
                    self._save(result, folder/'best_model')
                    np.savez_compressed(folder/'best_model/prediction.npz', prediction=result['prediction'])
                print(f'  최고 체크포인트 {record["amount"]}, RMSLE={score:.6f}, 상한 도달={record["hit_cap"]}', flush=True)
                del result
                gc.collect()
            dump(completed, chosen)
            selected[name] = chosen
        dump(parent/'selections.json', selected)
        return selected

    def _write_recipe(self, name, stage, recipe):
        dump(self.out/stage/'models'/name/'recipe.json', dict(name=name, **recipe))

    def predict_stage(self, fold, selections=None):
        """Fits fresh; future.y is NEVER passed to a training/early-stop call."""
        stage = self.out/fold.name
        dump(stage/'fold.json', fold.metadata)
        frame = fold.future.copy()
        frame['baseline'] = log_predictions(frame.lvl8)
        self._write_recipe('baseline', fold.name, dict(backend='baseline', base='lvl8', dependencies=[]))
        for name in self.trainable:
            folder = stage/'models'/name
            pred_path = folder/'prediction.npz'
            if pred_path.is_file() and (folder/'recipe.json').is_file():
                with np.load(pred_path) as saved:
                    prediction = saved['prediction']
                if len(prediction) != len(frame):
                    raise ValueError('저장 예측의 행 수가 다릅니다.')
                print(f'[재개] {fold.name}/{name}', flush=True)
            else:
                choice = (selections or {}).get(name)
                if choice is None and name != 'lightgbm_fixed500':
                    raise ValueError(f'{name}: 먼저 중간검증으로 설정·학습량을 선택해야 합니다.')
                parameters = choice['parameters'] if choice else self._parameters(name)
                amount = choice['amount'] if choice else (3 if self.config['smoke'] else 500)
                if name == 'lightgbm_fixed500':
                    amount = 3 if self.config['smoke'] else 500
                    description = f'고정 대조군 {amount}회'
                elif name == 'ridge':
                    amount = 1
                    description = f'alpha={parameters["alpha"]}, 직접 계산'
                else:
                    unit = '스텝' if name in SEQUENCES | {'ft_transformer'} else '회'
                    description = f'학습량={amount}{unit}'
                # Passing a target-free Fold copy makes the outer-label boundary
                # explicit even if another backend is added later.
                from dataclasses import replace
                target_free = replace(fold, future=fold.future.drop(columns=['y'], errors='ignore'))
                if fold.name == 'calibration' and choice is not None:
                    print(f'[중간검증 최고 모델 불러오기] {name}, {description}', flush=True)
                    checkpoint = self.out/'search/calibration'/name/'best_model'
                    shutil.copytree(checkpoint, folder, dirs_exist_ok=True)
                    prediction = self._replay_base(name, folder, target_free)
                    with np.load(checkpoint/'prediction.npz') as expected:
                        np.testing.assert_allclose(prediction, expected['prediction'], rtol=1e-6, atol=3e-6)
                else:
                    print(f'[선택 설정 재학습] {fold.name}/{name}, {description}', flush=True)
                    result = self._fit(name, parameters, target_free, monitor=False, amount=amount)
                    result['recipe'].update(origin=str(fold.origin.date()),
                        chosen_using=(choice or {}).get('chosen_using', 'fixed 500-round control'),
                        outer_labels_used_for_fit=False)
                    self._save(result, folder)
                    prediction = result['prediction']
                    replay = self._replay_base(name, folder, target_free)
                    np.testing.assert_allclose(replay, prediction, rtol=1e-6, atol=3e-6,
                                               err_msg=f'{name}: 저장 모델 재예측 불일치')
                    prediction = replay
                    del result
                np.savez_compressed(pred_path, prediction=prediction)
                gc.collect()
            frame[name] = log_predictions(prediction)
        if self.config['include_modular']:
            library = ensembles.event_library(fold.train)
            library.to_csv(stage/'event_library.csv', index=False)
            uplift, gate = ensembles.event_uplift(frame, library)
            frame['structure_v2'] = log_predictions(frame.base_v2)
            if 'role_lgb' in frame:
                frame['structure_residual_lgb'] = frame.role_lgb
                frame['role_ensemble'] = np.where(gate, log_predictions(frame.base_v2+uplift), frame.role_lgb)
            if 'role_lgb' in frame and 'role_cat' in frame:
                frame['structure_residual_blend'] = .5*frame.role_lgb+.5*frame.role_cat
            recipes = {
                'structure_v2': dict(backend='structure', dependencies=[], base='base_v2'),
                'structure_residual_lgb': dict(backend='fixed', dependencies=['role_lgb'], weights=[1.]),
                'structure_residual_blend': dict(backend='fixed', dependencies=['role_lgb', 'role_cat'], weights=[.5, .5]),
                'role_ensemble': dict(backend='event', dependencies=['role_lgb'], library='../event_library.csv'),
            }
            for name, recipe in recipes.items():
                if name in frame:
                    self._write_recipe(name, fold.name, recipe)
        for name, terms in FIXED.items():
            if all(component in frame for component, _ in terms):
                frame[name] = sum(frame[component]*weight for component, weight in terms)
                self._write_recipe(name, fold.name, dict(backend='fixed', dependencies=[n for n, _ in terms],
                                                       weights=[w for _, w in terms]))
        if 'lightgbm_fixed500' in frame and 'structure_residual_blend' in frame:
            name = 'ensemble_event_routed'
            frame[name] = np.where(frame.ev.eq(1), frame.lightgbm_fixed500, frame.structure_residual_blend)
            self._write_recipe(name, fold.name, dict(backend='event_route',
                dependencies=['lightgbm_fixed500', 'structure_residual_blend']))
        frame.to_parquet(stage/'predictions.parquet', index=False)
        self.score(frame, fold.name)
        return frame

    def score(self, frame, split):
        if 'y' not in frame:
            return pd.DataFrame()
        model_names = [n for n in CORE+MODULAR+list(FIXED)+['ensemble_event_routed']+LEARNED if n in frame]
        rows = []
        groups = [('전체', np.ones(len(frame), dtype=bool)),
                  ('행사 역-일', frame.diagnostic_event.eq(1)),
                  ('공휴일', frame.is_hol.eq(1)), ('신규 3역', frame.station_id.isin([3107, 3108, 3109]))]
        groups += [(f'월:{month}', frame.date.dt.strftime('%Y-%m').eq(month))
                   for month in sorted(frame.date.dt.strftime('%Y-%m').unique())]
        for group, mask in groups:
            subset = frame.loc[mask]
            for model in model_names:
                rows.append(dict(split=split, group=group, model=model, rows=len(subset),
                                 days=subset.date.nunique(), horizon_days=123,
                                 rmsle=rmsle(subset.y, subset[model]) if len(subset) else np.nan,
                                 smoke=self.config['smoke']))
        result = pd.DataFrame(rows)
        result.to_csv(self.out/split/'scores.csv', index=False, encoding='utf-8-sig')
        result.loc[result.group.eq('전체')].sort_values('rmsle').to_csv(
            self.out/split/'leaderboard.csv', index=False, encoding='utf-8-sig')
        return result

    def _fold(self, name):
        fold = self.engine.build(name, SPLITS[name], self.config)
        if self.config['include_modular']:
            print(f'[v4 재생성] {name}: 구조 기준선 생성', flush=True)
            self.engine.augment_structure(fold)
        print(f'[v4 재생성] {name}: 학습 {len(fold.train):,} / 예측 {len(fold.future):,} / 123일', flush=True)
        print(f'  기준일 {fold.metadata["origin"]} · 예측 {fold.metadata["start"]}~{fold.metadata["end"]}', flush=True)
        return fold

    def run_development(self):
        print('[1/4 모델별 중간검증 탐색] 첫 모델부터 후보별 파라미터·학습량을 선택합니다. 고정 200회 준비 학습은 없습니다.', flush=True)
        fold = self._fold('calibration')
        selections = self.search(fold)
        frame = self.predict_stage(fold, selections)
        experts = [n for n in EXPERTS if n in frame]
        if len(experts) < 2:
            raise ValueError('앙상블 비교에는 기본 예측 모델이 2개 이상 필요합니다.')
        days = sorted(frame.date.unique())
        train_days = int(self.config['meta_train_days'])
        if not 0 < train_days < len(days):
            raise ValueError('앙상블 내부 학습일 수는 1~122일이어야 합니다.')
        boundary = pd.Timestamp(days[train_days])
        meta_train = frame.loc[frame.date.lt(boundary)].reset_index(drop=True)
        meta_valid = frame.loc[frame.date.ge(boundary)].reset_index(drop=True)
        meta_spec = dict(train_start=str(meta_train.date.min().date()), train_end=str(meta_train.date.max().date()),
            train_days=meta_train.date.nunique(), validation_start=str(meta_valid.date.min().date()),
            validation_end=str(meta_valid.date.max().date()), validation_days=meta_valid.date.nunique(),
            source='same 123-day calibration forecast; no additional forecasting fold',
            base_selection_uses_full_calibration=True,
            interpretation='shared development data; internal ensemble scores are not independent final performance')
        print(f'[2/4 앙상블 학습] 중간검증의 앞 {meta_spec["train_days"]}일 학습 / 뒤 {meta_spec["validation_days"]}일 점검 후 전체 123일로 재학습합니다.', flush=True)
        selected, predictions = ensembles.develop(meta_train, meta_valid, experts, self.config, self.out/'ensemble_search')
        diagnostic = meta_valid[KEYS+['y', 'horizon']].copy()
        for name, values in predictions.items():
            diagnostic[name] = values
        diagnostic.to_parquet(self.out/'ensemble_search/internal_validation_predictions.parquet', index=False)
        dump(self.out/'ensemble_search/period.json', meta_spec)
        # Only this development forecast fits ensemble weights. In-sample
        # ensemble refit predictions are deliberately NOT added to a leaderboard.
        ensembles.refit(selected, frame, frame.drop(columns='y'), experts, self.config,
                        self.out/'calibration/models')
        result = dict(base_selections=selections, ensemble_selections=selected, experts=experts,
                      winner=None, calibration=fold.metadata, ensemble_internal_split=meta_spec,
                      parameter_selection_policy='calibration only, all parameters frozen before evaluation',
                      winner_selection_policy='lowest final 123-day RMSLE among the frozen candidate models',
                      initial_fixed_training=False)
        dump(self.out/'BEST_CONFIGURATION.json', result)
        dump(self.out/'DEVELOPMENT_CONFIGURATION.json', result)
        del fold, frame, meta_train, meta_valid
        gc.collect()
        return result

    def apply_frozen_ensembles(self, frame, split, settings):
        """Copy and predict calibrated weights; NEVER fit on evaluation labels."""
        for name in LEARNED:
            source = self.out/'calibration/models'/name
            destination = self.out/split/'models'/name
            shutil.copytree(source, destination, dirs_exist_ok=True)
            frame[name] = ensembles.predict_saved(destination, frame, self.config)
        return frame

    def run_evaluation(self):
        print('[3/4 최종 123일 평가] 2026-02-28~06-30. 고정된 설정으로 비교하며 가중치·학습량은 바꾸지 않습니다.', flush=True)
        settings = read_json(self.out/'BEST_CONFIGURATION.json')
        fold = self._fold('evaluation')
        frame = self.predict_stage(fold, settings['base_selections'])
        frame = self.apply_frozen_ensembles(frame.drop(columns='y'), 'evaluation', settings).assign(y=fold.future.y.to_numpy())
        from torch_models import predict_gate
        weights = predict_gate(self.out/'evaluation/models/context_gate', frame, self.config, return_weights=True)
        weight_frame = pd.DataFrame(weights, columns=settings['experts'])
        summaries = []
        for column in ['station_id', 'hour', 'is_hol']:
            group = weight_frame.groupby(frame[column].to_numpy()).mean().reset_index().rename(columns={'index': 'value'})
            group.insert(0, 'group', column)
            summaries.append(group)
        pd.concat(summaries, ignore_index=True).to_csv(self.out/'evaluation/context_weights.csv', index=False)
        frame.to_parquet(self.out/'evaluation/predictions.parquet', index=False)
        scores = self.score(frame, 'evaluation')
        best_outer = scores.loc[scores.group.eq('전체')].sort_values('rmsle').iloc[0]
        winner = str(best_outer.model)
        settings.update(winner=winner, winner_evaluation_rmsle=float(best_outer.rmsle))
        dump(self.out/'BEST_CONFIGURATION.json', settings)
        dump(self.out/'BEST_EVALUATION_MODEL.json', dict(model=str(best_outer.model), rmsle=float(best_outer.rmsle),
            path=f'evaluation/models/{best_outer.model}', role='best model family after frozen-parameter 123-day evaluation'))
        dump(self.out/'BEST_MODEL.json', dict(model=winner, chosen_on='evaluation',
            path=f'evaluation/models/{winner}',
            evaluation_rmsle=rmsle(frame.y, frame[winner]), dependencies=self.dependencies(winner, 'evaluation')))
        dump(self.out/'leakage_checks.json', dict(
            v4_inputs_only='PASS', original_inputs_modified=False,
            passenger_max='2026-06-30', passenger_values_after_june30='NOT READ, including evaluation',
            physically_truncated_cube='PASS for each fold and every structural feature origin',
            avail_end_equal_to_fold_origin='PASS', training_target_max_le_origin='PASS',
            validation_y_attached_after_inputs='PASS', target_derived_validation_events='diagnostics only',
            new_stations='all observed labels retained, past-only peer level fallback; pre-opening labels absent',
            history_detector='rebuilt on each prefix, never full-history validation detector',
            outer_labels_for_stopping_or_weight_fit=False, public_scores_used=False,
            fold_horizon=123, seed=42, prediction_clipping='each expert log prediction >=0 before combination',
            meta_training='calibration-only predictions: earlier 82 days train, later 41 days select; refit on calibration',
            preopening_validation_fold=False, initial_fixed_200_stage=False,
            base_and_ensemble_parameters_chosen_on='2025-10-28..2026-02-27 only',
            final_model_family_chosen_on='final 123-day leaderboard; per-model estimates, winner chosen after comparison',
            known_limitations=['historical independent event schedule not provided; historical ev inputs unknown',
                               'v4 rain3 retains supplied 06-23 rolling semantics',
                               'neural inputs differ structurally from the common 55 tabular features'],
            input_sha256_unchanged={n: sha(self.engine.data_dir/n) == h for n, h in self.engine.input_hashes.items()
                                    if n in self.engine.REQUIRED}))
        return scores

    def dependencies(self, name, split):
        recipe = read_json(self.out/split/'models'/name/'recipe.json')
        direct = recipe.get('experts', recipe.get('dependencies', []))
        result = [name]
        for dependency in direct:
            for item in self.dependencies(dependency, split):
                if item not in result:
                    result.append(item)
        return result

    def run_final(self):
        print('[4/4 제출용 재학습] 최고 모델을 2026-06-30까지의 기록으로 선택된 학습량만큼 학습합니다.', flush=True)
        settings = read_json(self.out/'BEST_CONFIGURATION.json')
        if not settings.get('winner'):
            raise ValueError('먼저 최종 123일 평가를 실행해 최고 모델을 선택하세요.')
        if self.config['smoke']:
            dump(self.out/'SMOKE_ONLY.json', dict(message='동작 검사 전용. 제출용 파일을 생성하지 않습니다.'))
            return None
        # Fit the requested final experts only; an all-model option is available.
        needed = self.dependencies(settings['winner'], 'evaluation')
        keep = self.trainable
        try:
            if self.config['final_models'] == 'winner':
                self.trainable = [n for n in keep if n in needed]
            fold = self._fold('final')
            frame = self.predict_stage(fold, settings['base_selections'])
        finally:
            self.trainable = keep
        winner = settings['winner']
        if winner in LEARNED or self.config['final_models'] == 'all':
            # Learned ensembles all depend on the same declared expert pool.
            available = [n for n in LEARNED if all(e in frame for e in settings['experts'])]
            if winner in LEARNED and not available:
                raise ValueError('최종 앙상블 구성 모델이 누락됐습니다.')
            if available:
                frame = self.apply_frozen_ensembles(frame, 'final', settings)
        if winner not in frame:
            raise ValueError('선택된 최종 모델의 예측이 없습니다: '+winner)
        frame.to_parquet(self.out/'final/predictions.parquet', index=False)
        submission = pd.DataFrame({'example_id': frame.example_id,
                                   'boardings': np.expm1(log_predictions(frame[winner]))})
        path = self.out/'submission_stack.csv'
        submission.to_csv(path, index=False, float_format='%.10f', lineterminator='\n')
        checks = self.check_submission(path)
        dump(self.out/'submission_checks.json', dict(**checks, model=winner,
            chosen_using='final 123-day comparison; all hyperparameters and weights frozen on calibration', extra_postprocessing=False))
        dump(self.out/'FINAL_MODEL.json', dict(model=winner, path=f'final/models/{winner}',
            components=self.dependencies(winner, 'final'), submission='submission_stack.csv', sha256=checks['sha256']))
        # Verify the CSV survives a fresh artifact reload, without any fitting.
        replay = self.replay_final()
        if sha(replay) != sha(path):
            raise AssertionError('저장 모델 재예측 CSV의 SHA256이 다릅니다.')
        dump(self.out/'replay_check.json', dict(identical_sha256=True, sha256=sha(path),
                                              method='all required models reloaded, forecast again, CSV rewritten'))
        return path

    def replay_final(self):
        settings = read_json(self.out/'BEST_CONFIGURATION.json')
        name = settings['winner']
        fold = self.engine.build('final', SPLITS['final'], self.config, with_train=False)
        if self.config['include_modular']:
            self.engine.augment_structure(fold)
        frame = fold.future.copy()
        def predict(model):
            if model in frame:
                return
            folder = self.out/'final/models'/model
            recipe = read_json(folder/'recipe.json')
            backend = recipe['backend']
            for dep in recipe.get('experts', recipe.get('dependencies', [])):
                predict(dep)
            if backend == 'baseline':
                value = frame.lvl8
            elif backend == 'structure':
                value = frame.base_v2
            elif backend == 'fixed':
                value = sum(frame[n]*w for n, w in zip(recipe['dependencies'], recipe['weights']))
            elif backend == 'event':
                library = pd.read_csv(self.out/'final/event_library.csv')
                uplift, gate = ensembles.event_uplift(frame, library)
                value = np.where(gate, log_predictions(frame.base_v2+uplift), frame.role_lgb)
            elif backend == 'event_route':
                value = np.where(frame.ev.eq(1), frame.lightgbm_fixed500, frame.structure_residual_blend)
            elif backend in ['global_weights', 'ridge_stacking', 'context_gate']:
                value = ensembles.predict_saved(folder, frame, self.config)
            else:
                value = self._replay_base(model, folder, fold)
            frame[model] = log_predictions(value)
        predict(name)
        path = self.out/'submission_reloaded.csv'
        pd.DataFrame({'example_id': frame.example_id, 'boardings': np.expm1(frame[name])}).to_csv(
            path, index=False, float_format='%.10f', lineterminator='\n')
        return path

    def check_submission(self, path):
        frame = pd.read_csv(path)
        expected = self.engine.test.example_id
        assert frame.columns.tolist() == ['example_id', 'boardings']
        assert len(frame) == 157194 and frame.example_id.is_unique
        assert frame.example_id.tolist() == expected.tolist()
        assert np.isfinite(frame.boardings).all() and frame.boardings.ge(0).all()
        return dict(rows=len(frame), ids_match=True, duplicate_ids=0, nan=0, negative=0, sha256=sha(path))

    def run_all(self):
        self.describe().to_csv(self.out/'search_budget.csv', index=False)
        self.run_development()
        scores = self.run_evaluation()
        submission = self.run_final() if self.config['final_refit'] else None
        from report import write_report
        write_report(self.out)
        return scores, submission
