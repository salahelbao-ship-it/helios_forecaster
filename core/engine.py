# core/engine.py
import time, os, gc, hashlib, traceback
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
except Exception: pass

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner 

from PySide6.QtCore import QThread, Signal
from core.datatypes import ExperimentConfig, SiteContext, FoldResult, ExperimentResult, FoldDefinition, CancelledError, DataQualityError
from core.models import build_agfa_model, apply_transfer_learning, MultiHorizonNGBoost, MultiHorizonCopula, BaselineEngine, CancelCallback
from data.loader import DataLoader, DataQualityAuditor
from data.features import FoldPreprocessor, NonLinearFeatureSelector, estimate_solar_elevation
from data.sequences import SequenceBuilder
from data.database import CacheManager, ArtifactStore
from utils.helpers import generate_environment_meta, get_code_hash
from utils.logger import get_logger

try: from utils.config import ConfigManager
except ImportError: ConfigManager = None

logger = get_logger("HeliosEngine")

class EpochProgressCallback(Callback):
    def __init__(self, log_func, model_name, fold_idx, total_epochs):
        super().__init__(); self.log_func, self.m_name, self.f_idx, self.total_epochs = log_func, model_name, fold_idx, total_epochs
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0 or epoch == 0: self.log_func(f"[{self.m_name} F{self.f_idx}] Epoch {epoch+1}/{self.total_epochs} | val_loss: {logs.get('val_loss', 0.0):.4f}")

class Evaluator:
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_quantiles: Dict[float, np.ndarray], cap: float, y_base: np.ndarray = None) -> Dict[str, float]:
        y_p50 = y_quantiles.get(0.5, y_true); y_p10 = y_quantiles.get(0.1, y_true); y_p90 = y_quantiles.get(0.9, y_true)
        rmse = np.sqrt(mean_squared_error(y_true, y_p50))
        mean_y = np.mean(y_true); r2 = r2_score(y_true, y_p50) if np.var(y_true) >= 1e-6 else 0.0
        nrmse_cap = (rmse / cap) * 100 if cap > 0 else 0.0
        nrmse_mean = (rmse / mean_y) * 100 if mean_y > 1e-6 else 0.0
        picp = np.mean((y_true >= y_p10) & (y_true <= y_p90)) * 100
        mpiw = np.mean(y_p90 - y_p10)
        mbe = np.mean(y_p50 - y_true)
        day_mask = y_true > (cap * 0.01)
        rmse_day = np.sqrt(mean_squared_error(y_true[day_mask], y_p50[day_mask])) if np.sum(day_mask) > 0 else rmse
        y_t_diff, y_p_diff = np.diff(y_true), np.diff(y_p50)
        rmse_ramp = np.sqrt(mean_squared_error(y_t_diff, y_p_diff)) if len(y_t_diff) > 0 else 0.0
        skill_day = skill_ramp = 0.0
        if y_base is not None:
            rbd = np.sqrt(mean_squared_error(y_true[day_mask], y_base[day_mask])) if np.sum(day_mask) > 0 else rmse_day
            if rbd > 0: skill_day = (1.0 - (rmse_day / rbd)) * 100.0
            rbr = np.sqrt(mean_squared_error(y_t_diff, np.diff(y_base))) if len(y_t_diff) > 0 else rmse_ramp
            if rbr > 0: skill_ramp = (1.0 - (rmse_ramp / rbr)) * 100.0
        crps = np.mean([np.mean(np.maximum(q*(y_true-y_quantiles.get(q, y_p50)), (q-1)*(y_true-y_quantiles.get(q, y_p50)))) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        return {
            'RMSE': float(rmse), 'RMSE_Day': float(rmse_day), 'RMSE_Ramp': float(rmse_ramp),
            'Skill_Day_%': float(skill_day), 'Skill_Ramp_%': float(skill_ramp), 
            'nRMSE_c%': float(nrmse_cap), 'nRMSE_m%': float(nrmse_mean), 
            'MAE': float(mean_absolute_error(y_true, y_p50)), 'MAPE': float(np.mean(np.abs((y_true - y_p50) / np.maximum(y_true, cap * 0.01))) * 100),
            'sMAPE': float(np.mean(2.0 * np.abs(y_p50 - y_true) / (np.abs(y_true) + np.abs(y_p50) + 1e-8)) * 100),
            'MBE': float(mbe), 'R2': float(r2), 'CRPS_Approx': float(crps), 'PICP': float(picp), 'MPIW': float(mpiw)
        }

    @staticmethod
    def dm_nw_hac_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, h: int = 1) -> Tuple[float, float]:
        d = (np.array(y_true) - np.array(y_pred1))**2 - (np.array(y_true) - np.array(y_pred2))**2
        T = len(d); lag = max(h - 1, int(4 * (T / 100)**(2/9)))
        if T == 0 or np.var(d) == 0: return 0.0, 1.0
        V_hac = np.var(d)
        for j in range(1, lag + 1):
            if T <= j: break
            V_hac += 2 * (1 - j / (lag + 1)) * np.cov(d[j:], d[:-j])[0, 1]
        dm_stat = np.mean(d) / np.sqrt(V_hac / T) if V_hac > 0 else 0.0
        return float(dm_stat), float(2 * (1 - stats.norm.cdf(np.abs(dm_stat))))

    @staticmethod
    def compute_permutation_importance(model_obj, X_val, y_val, feature_cols, q_labels, cap, metric='RMSE'):
        baseline_q = model_obj.predict_quantiles(X_val.reshape(X_val.shape[0], -1), q_labels) if hasattr(model_obj, 'predict_quantiles') else {q: model_obj.predict(X_val, verbose=0).reshape(-1, len(q_labels))[:, i] for i, q in enumerate(q_labels)}
        base_rmse = np.sqrt(mean_squared_error(y_val.flatten(), baseline_q.get(0.5, y_val.flatten())))
        importance_scores = {}
        for feat_idx, feat_name in enumerate(feature_cols):
            X_shuffled = X_val.copy()
            target_shape = X_shuffled[:, :, feat_idx].shape
            flat_feat = X_shuffled[:, :, feat_idx].flatten()
            np.random.shuffle(flat_feat)
            X_shuffled[:, :, feat_idx] = flat_feat.reshape(target_shape)
            shuff_q = model_obj.predict_quantiles(X_shuffled.reshape(X_shuffled.shape[0], -1), q_labels) if hasattr(model_obj, 'predict_quantiles') else {q: model_obj.predict(X_shuffled, verbose=0).reshape(-1, len(q_labels))[:, i] for i, q in enumerate(q_labels)}
            importance_scores[feat_name] = max(0.0, float(np.sqrt(mean_squared_error(y_val.flatten(), shuff_q.get(0.5, y_val.flatten()))) - base_rmse))
        total_importance = sum(importance_scores.values()) + 1e-9
        return {k: (v / total_importance) * 100.0 for k, v in importance_scores.items()}

class TrainingEngine:
    def __init__(self, config: ExperimentConfig, site_ctx: SiteContext, log_cb: Callable, prog_cb: Callable, status_cb: Callable, is_cancelled_func: Callable, model_done_cb: Callable = None):
        self.config = config
        self.site = site_ctx
        self.log_ui = log_cb
        self.prog = prog_cb
        self.status = status_cb
        self.is_cancelled = is_cancelled_func
        self.model_done_cb = model_done_cb
        self.cache = CacheManager()
        self.warnings, self.errors = [], []

    def log(self, msg: str, level: str = "info"): 
        self.log_ui(msg)

    def run(self) -> List[ExperimentResult]:
        try:
            df, qa_report, freq_str, freq_mode = self._ingest_data()
            dt_minutes = max(pd.to_timedelta(freq_str).total_seconds() / 60.0 if freq_str != 'Unknown' else 1.0, 1.0)
            lead_k_map = {t_lead: max(0, int(round(t_lead / dt_minutes)) - 1) for t_lead in self.config.target_leads_minutes}
            self.config.forecast_horizon = max(lead_k_map.values()) + 1
            effective_embargo = max(self.config.embargo_gap, self.config.sequence_length)
            tscv = TimeSeriesSplit(n_splits=self.config.n_folds, gap=effective_embargo)
            models = [m.strip() for m in self.config.selected_models if m.strip() != "Dynamic Ensemble"]
            ensemble_requested = any(m.strip() == "Dynamic Ensemble" for m in self.config.selected_models)
            all_results = []
            total_steps = (len(models) + (1 if ensemble_requested else 0)) * self.config.n_folds
            step_idx = 0

            for m_name in models:
                res = self._process_model(m_name, df, tscv, self.config.get_hash(), freq_str, freq_mode, qa_report, step_idx, total_steps, lead_k_map)
                all_results.append(res)
                step_idx += self.config.n_folds
                if self.model_done_cb: self.model_done_cb(res)
                
            if ensemble_requested and len(all_results) > 1:
                self.status("CALCULATING DYNAMIC ENSEMBLE WEIGHTS")
                ens_res = self._process_dynamic_ensemble(all_results, df, tscv, freq_str, qa_report, lead_k_map)
                all_results.append(ens_res)
                if self.model_done_cb: self.model_done_cb(ens_res)
                
            self.status("EXECUTION COMPLETED")
            return all_results
        except CancelledError: raise
        except Exception as e: self.log(f"CRITICAL FAULT: {traceback.format_exc()}", "error"); raise e

    def _ingest_data(self) -> Tuple[pd.DataFrame, Dict, str, str]:
        self.status(f"AUDITING MINIMAL SENSORS [{self.site.name}]")
        df, qa_report, target_col = DataLoader.load_df(self.config.data_path, self.site.timezone, self.config.modify_target, self.config.strict_qa_mode, self.config.target_col)
        self.config.target_col = self.site.target_col = target_col
        if self.is_cancelled(): raise CancelledError()
        
        t_pan_col = next((c for c in df.columns if 'temp' in c.lower() and ('pan' in c.lower() or 'mod' in c.lower())), None)
        t_amb_col = next((c for c in df.columns if 'temp' in c.lower() and c != t_pan_col), None)
        
        if t_pan_col and t_amb_col:
            self.log("Identified Temperature Pairs: Injecting Bounded TSI & TELR Features.")
            day_mask = df[target_col].values > 50.0
            safe_irrad = np.maximum(df[target_col].values, 50.0)
            safe_irrad_diff = np.maximum(df[target_col].diff().abs().values, 10.0)
            df['TSI'] = np.where(day_mask, (df[t_pan_col] - df[t_amb_col]) / safe_irrad, 0.0)
            df['TELR'] = np.where(day_mask, df[t_pan_col].diff() / safe_irrad_diff, 0.0)
            df['TELR'] = df['TELR'].bfill().fillna(0)
            if 'TSI' not in self.site.exog_cols: self.site.exog_cols.extend(['TSI', 'TELR'])
            
        df['IVI'] = df[target_col].rolling(5, min_periods=1).std().bfill().fillna(0)
        if 'IVI' not in self.site.exog_cols: self.site.exog_cols.append('IVI')

        self.config.dataset_hash = DataLoader.hash_dataset(self.config.data_path)
        self.site.capacity = self.site.capacity or DataQualityAuditor.infer_capacity(df, target_col)
        return df, qa_report, qa_report.get('issues', {}).get('inferred_freq', 'Unknown'), qa_report.get('issues', {}).get('frequency_mode', 'IRREGULAR_UNKNOWN')

    def _process_model(self, m_name: str, df: pd.DataFrame, tscv: TimeSeriesSplit, config_hash: str, freq_str: str, freq_mode: str, qa_report: dict, step_start: int, total_steps: int, lead_k_map: Dict[int, int]) -> ExperimentResult:
        fold_results: List[FoldResult] = []
        total_train_time = 0.0
        run_fingerprint = hashlib.sha256(f"{config_hash}_{self.config.dataset_hash}_{get_code_hash()}".encode()).hexdigest()
        q_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        step_idx = step_start
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            if self.is_cancelled(): raise CancelledError()
            self.status(f"PROCESSING {m_name.upper()} (FOLD {fold_idx}/{self.config.n_folds})")
            
            df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
            fold_def = FoldDefinition(fold_idx, str(df_train.index[0]), str(df_train.index[-1]), str(df_test.index[0]), str(df_test.index[-1]), len(df_train), len(df_test), self.config.embargo_gap, freq_str)
            
            model_obj, prep, best_params, cached, degraded, t_dur, feat_rep = self._train_fold(m_name, fold_idx, df_train, config_hash, freq_str, freq_mode)
            total_train_time += t_dur

            final_seq_len = best_params.get('seq_len', self.config.sequence_length)
            lookback_df = pd.concat([df_train.iloc[-final_seq_len:], df_test])
            df_test_full = prep.transform(lookback_df)
            X_ts_seq, Y_ts_seq_scaled, iss_t, val_t_matrix = SequenceBuilder.build_sequences(df_test_full, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon)
            if len(Y_ts_seq_scaled) == 0: continue

            Y_ts_seq_unscaled = prep.scaler_y.inverse_transform(Y_ts_seq_scaled.reshape(-1, 1)).reshape(Y_ts_seq_scaled.shape)
            df_all_context = pd.concat([df_train, df_test])
            y_base_pers_1d = BaselineEngine.make_persistence(df_all_context, self.config.target_col, iss_t)
            y_base_pers_flat = np.repeat(y_base_pers_1d, self.config.forecast_horizon)
            
            y_quantiles, crossing_pre = self._execute_predictions(m_name, df_train, df_all_context, prep, model_obj, X_ts_seq, iss_t, val_t_matrix, y_base_pers_flat, q_labels)
            
            metrics = {}
            for lead_min, k in lead_k_map.items():
                if k >= self.config.forecast_horizon: continue
                y_true_k = Y_ts_seq_unscaled[:, k]
                y_pred_q_k = {}
                for q in q_labels:
                    raw_q = y_quantiles.get(q, y_quantiles.get(str(q)))
                    if raw_q is not None: y_pred_q_k[q] = raw_q.reshape(len(iss_t), self.config.forecast_horizon)[:, k]
                
                night_mask = estimate_solar_elevation(pd.DatetimeIndex(val_t_matrix[:, k]), self.site.lat, self.site.lon) < 2.0
                for q in y_pred_q_k.keys():
                    y_pred_q_k[q][night_mask] = 0.0
                    y_pred_q_k[q] = np.clip(y_pred_q_k[q], 0, prep.capacity)

                lead_metrics = Evaluator.compute_metrics(y_true_k, y_pred_q_k, prep.capacity, y_base_pers_1d)
                for mk, mv in lead_metrics.items(): metrics[f"{mk}_L{lead_min}"] = mv
                
                dm_stat, p_val = Evaluator.dm_nw_hac_test(y_true_k, y_pred_q_k[0.5], y_base_pers_1d, h=k+1) if 'Persistence' not in m_name else (0.0, 1.0)
                metrics[f"DM_p_L{lead_min}"] = p_val
                metrics[f"DM_stat_L{lead_min}"] = dm_stat

            # --- EXPORT SCHEMA FIX: Format quantile keys as integer strings ('10', '50', '90') ---
            q_dict_formatted = {str(int(round(float(q) * 100))): y_quantiles[q].tolist() for q in y_quantiles.keys()}

            fold_results.append(FoldResult(
                fold_id=fold_idx, issue_times=[t.isoformat() for t in np.repeat(iss_t, self.config.forecast_horizon)], 
                valid_times=[t.isoformat() for t in pd.DatetimeIndex(val_t_matrix.flatten())], 
                y_true=Y_ts_seq_unscaled.flatten().tolist(), 
                y_pred_quantiles=q_dict_formatted, 
                baseline_pers=y_base_pers_flat.tolist(), metrics=metrics, pinball_by_q={}, 
                crossing_rate_pre_repair=float(crossing_pre), params=best_params, fold_def=fold_def, 
                training_duration_sec=t_dur, loaded_from_cache=cached, feature_selection_report=feat_rep
            ))
            
            step_idx += 1
            self.prog(int((step_idx / total_steps) * 100))
            K.clear_session(); gc.collect()

        return ExperimentResult(
            model_name=m_name, site_name=self.site.name, config_id=config_hash, seq_len=self.config.sequence_length, 
            timestamp_created=datetime.now().isoformat(), capacity=float(self.site.capacity), horizon=self.config.forecast_horizon, 
            folds=fold_results, training_time_total=total_train_time, run_fingerprint=run_fingerprint, 
            env_meta=generate_environment_meta(), site_meta=self.site.__dict__, qa_report=qa_report, 
            config_dump=self.config.__dict__, methodology=f"Mode: {self.config.evaluation_mode} | Folds: {self.config.n_folds}", 
            warnings=self.warnings, errors=self.errors
        )

    def _process_dynamic_ensemble(self, previous_results, df, tscv, freq_str, qa_report, lead_k_map):
        fold_results, q_labels = [], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            if self.is_cancelled(): raise CancelledError()
            fold_models = [r.folds[fold_idx-1] for r in previous_results if 'Persistence' not in r.model_name and len(r.folds) >= fold_idx]
            if not fold_models: continue
            
            rmses = np.array([m.metrics.get('RMSE', 1.0) for m in fold_models])
            inv_errors = 1.0 / (rmses + 1e-6)
            weights = inv_errors / inv_errors.sum()
            
            # Use the integer string keys ('10', '50') to map correctly across schemas
            q_str_map = {q: str(int(round(float(q) * 100))) for q in q_labels}
            ens_q = {q_str_map[q]: np.zeros_like(np.array(fold_models[0].y_pred_quantiles.get(q_str_map[q], fold_models[0].y_pred_quantiles.get(str(q))))) for q in q_labels}
            
            for q in q_labels:
                q_s = q_str_map[q]
                fallback_s = str(q)
                for w, m in zip(weights, fold_models): 
                    arr = m.y_pred_quantiles.get(q_s, m.y_pred_quantiles.get(fallback_s))
                    ens_q[q_s] += w * np.array(arr)
                
            y_true_k = np.array(fold_models[0].y_true)
            y_base = np.array(fold_models[0].baseline_pers)
            val_t_matrix = np.array(fold_models[0].valid_times).reshape(-1, self.config.forecast_horizon)
            iss_t = np.array(fold_models[0].issue_times)[::self.config.forecast_horizon]
            
            metrics = {}
            for lead_min, k in lead_k_map.items():
                if k >= self.config.forecast_horizon: continue
                yt_k = y_true_k.reshape(-1, self.config.forecast_horizon)[:, k]
                y_pred_q_k = {q: ens_q[q_str_map[q]].reshape(-1, self.config.forecast_horizon)[:, k] for q in q_labels}
                lead_metrics = Evaluator.compute_metrics(yt_k, y_pred_q_k, float(self.site.capacity), y_base.reshape(-1, self.config.forecast_horizon)[:, k])
                for mk, mv in lead_metrics.items(): metrics[f"{mk}_L{lead_min}"] = mv
                
            fold_results.append(FoldResult(
                fold_id=fold_idx, issue_times=fold_models[0].issue_times, valid_times=fold_models[0].valid_times, 
                y_true=y_true_k.tolist(), y_pred_quantiles={k: v.tolist() for k, v in ens_q.items()}, 
                baseline_pers=y_base.tolist(), metrics=metrics, pinball_by_q={}, crossing_rate_pre_repair=0.0, 
                params={'ensemble_weights': {previous_results[i].model_name: float(weights[i]) for i in range(len(weights))}}, 
                fold_def=fold_models[0].fold_def, training_duration_sec=0.1, loaded_from_cache=False, feature_selection_report={}
            ))
            
        return ExperimentResult(model_name="Dynamic Ensemble", site_name=self.site.name, config_id=self.config.get_hash(), seq_len=self.config.sequence_length, timestamp_created=datetime.now().isoformat(), capacity=float(self.site.capacity), horizon=self.config.forecast_horizon, folds=fold_results, training_time_total=1.0, run_fingerprint="ensemble_"+hashlib.sha256(self.config.get_hash().encode()).hexdigest(), env_meta=generate_environment_meta(), site_meta=self.site.__dict__, qa_report=qa_report, config_dump=self.config.__dict__, methodology="Inverse-RMSE Monte Carlo Fusion", warnings=[], errors=[])

    def _train_fold(self, m_name: str, fold_idx: int, df_train: pd.DataFrame, config_hash: str, freq_str: str, freq_mode: str) -> Tuple[Any, FoldPreprocessor, dict, bool, bool, float, dict]:
        model_obj = None 
        is_ml, is_postprocessor = 'Persistence' not in m_name, m_name.startswith("CopulaBayes")
        cache_key = self.cache.get_cache_key(config_hash, self.config.dataset_hash, self.site.name, m_name, fold_idx)
        cached, degraded, t_duration, feat_report = False, False, 0.0, {}
        q_labels, best_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], {'units': 64, 'dropout': 0.2, 'lr': 0.001, 'seq_len': self.config.sequence_length}
        manifest = self.cache.check_cache(cache_key, self.config, self.site.name, m_name, fold_idx)

        is_wpd_active = "WPD-" in m_name
        sel_exog, feat_report = NonLinearFeatureSelector.select(df_train, self.config.target_col, self.site.exog_cols, self.site.lat, self.site.lon, self.config.forecast_horizon, self.config.exog_availability) if "Auto" in self.config.feature_selection_mode else ([c for c in self.site.exog_cols if c in df_train.columns], {'mode': 'manual'})

        prep = FoldPreprocessor(self.config.target_col, self.config.use_advanced_features, self.config.use_ramp_features, is_wpd_active, self.config.forecast_horizon, self.config.exog_availability, freq_str, freq_mode, self.site.lat, self.site.lon, sel_exog)
        prep.feat_report = feat_report
        prep.fit(df_train, self.site.capacity)
        
        wpd_indices = [i for i, c in enumerate(prep.feature_cols) if 'wpd' in c.lower() or 'swt' in c.lower()] if is_wpd_active else []
        gater_target_cols = ['ivi', 'tsi', 'telr', 'ghi', 'ramp', 'tamb', 'tpan']
        gater_indices = [i for i, c in enumerate(prep.feature_cols) if any(g in c.lower() for g in gater_target_cols)]
        context_indices = [i for i, c in enumerate(prep.feature_cols) if i not in wpd_indices]

        if is_ml and self.config.use_cache and manifest:
            try:
                n_feats, best_params = manifest['n_features'], manifest['hyperparams']
                K.clear_session()
                model_obj, prep = self.cache.load_cache(cache_key) if "NGBoost" in m_name or is_postprocessor else self.cache.load_cache(cache_key, build_agfa_model(m_name.replace("WPD-", ""), best_params.get('seq_len', self.config.sequence_length), n_feats, wpd_indices, context_indices, gater_indices, best_params.get('units', 64), best_params.get('dropout', 0.2), best_params.get('lr', 0.001), q_labels, self.config.forecast_horizon))
                self.log(f"[{m_name} F{fold_idx}] AGFA Weights restored securely from cache.")
                return model_obj, prep, best_params, True, degraded, 0.0, getattr(prep, 'feat_report', {})
            except Exception as e: self.log(f"[{m_name} F{fold_idx}] Cache load failed: {e}. Re-training.")
        
        df_t_sub_trans, df_v_sub_trans = prep.transform(df_train.iloc[:int(len(df_train)*0.8)]), prep.transform(df_train.iloc[int(len(df_train)*0.8):])
        final_seq_len, n_features = best_params.get('seq_len', self.config.sequence_length), len(prep.feature_cols)

        if is_ml and not is_postprocessor and getattr(self.config, 'transfer_weights_path', None) and os.path.exists(self.config.transfer_weights_path):
            self.status(f"TRANSFER LEARNING: Fine-tuning {m_name.upper()} (FOLD {fold_idx})")
            t_start = time.time(); K.clear_session()
            model_obj = apply_transfer_learning(build_agfa_model(m_name.replace("WPD-", ""), final_seq_len, n_features, wpd_indices, context_indices, gater_indices, best_params.get('units', 64), best_params.get('dropout', 0.2), 1e-5, q_labels, self.config.forecast_horizon), self.config.transfer_weights_path, freeze_base=getattr(self.config, 'freeze_base_layers', True))
            
            # --- TL STABILITY FIX: Elite Callbacks injected into fine-tuning to prevent divergence ---
            tl_callbacks = [
                CancelCallback(self.is_cancelled),
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                EpochProgressCallback(self.log, f"TL-{m_name}", fold_idx, getattr(self.config, 'fine_tune_epochs', 50))
            ]
            
            ds_train, _ = SequenceBuilder.build_tf_dataset(df_t_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon, 32, True)
            ds_val, _ = SequenceBuilder.build_tf_dataset(df_v_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon, 32)
            model_obj.fit(ds_train, validation_data=ds_val, epochs=getattr(self.config, 'fine_tune_epochs', 50), verbose=0, callbacks=tl_callbacks)
            t_duration = time.time() - t_start

        elif is_ml and not is_postprocessor and "NGBoost" not in m_name and self.config.model_optuna.get(m_name, self.config.use_optuna):
            self.status(f"TUNING {m_name.upper()} (FOLD {fold_idx})")
            optuna.logging.set_verbosity(optuna.logging.ERROR) 

            def obj(trial):
                if self.is_cancelled(): raise optuna.exceptions.OptunaError("Cancelled")
                K.clear_session()
                
                if any(rnn in m_name for rnn in ["LSTM", "BiLSTM", "GRU"]):
                    u = trial.suggest_int('units', 64, 128, step=32)
                    d = trial.suggest_float('dropout', 0.1, 0.3)
                    trial_seq_len = trial.suggest_categorical('seq_len', [48, 96]) 
                elif "TCN-iTransformer" in m_name:
                    u = trial.suggest_int('units', 64, 256, step=64)
                    d = trial.suggest_float('dropout', 0.2, 0.4)
                    trial_seq_len = trial.suggest_categorical('seq_len', [72, 144]) 
                else:
                    u = trial.suggest_int('units', 64, 128, step=32)
                    d = trial.suggest_float('dropout', 0.1, 0.3)
                    trial_seq_len = trial.suggest_categorical('seq_len', [48, 96])
                    
                l = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
                
                ds_train, n_tr = SequenceBuilder.build_tf_dataset(df_t_sub_trans, prep.feature_cols, self.config.target_col, trial_seq_len, self.config.forecast_horizon, 32, True)
                ds_val, n_v = SequenceBuilder.build_tf_dataset(df_v_sub_trans, prep.feature_cols, self.config.target_col, trial_seq_len, self.config.forecast_horizon, 32)
                if n_tr == 0 or n_v == 0: raise optuna.exceptions.TrialPruned()
                
                m = build_agfa_model(m_name.replace("WPD-", ""), trial_seq_len, len(prep.feature_cols), wpd_indices, context_indices, gater_indices, u, d, l, q_labels, self.config.forecast_horizon)
                h = m.fit(ds_train, validation_data=ds_val, epochs=20, verbose=0, callbacks=[
                    optuna.integration.TFKerasPruningCallback(trial, 'val_loss'), 
                    CancelCallback(self.is_cancelled), 
                    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=False, min_delta=1e-4) 
                ])
                
                val_loss_min = min(h.history['val_loss'])
                del m, ds_train, ds_val; gc.collect()
                return val_loss_min

            try:
                study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42, multivariate=True), pruner=HyperbandPruner(min_resource=3, max_resource=20, reduction_factor=3))
                study.optimize(obj, n_trials=self.config.n_trials, callbacks=[lambda s, t: self.log(f"[{m_name} F{fold_idx}] NAS Trial {t.number} | Loss: {t.value:.4f}")])
                best_params.update(study.best_params)
            except Exception: degraded = True
            
            final_seq_len, n_features = best_params.get('seq_len', self.config.sequence_length), len(prep.feature_cols)
            t_start = time.time()
            ds_train, _ = SequenceBuilder.build_tf_dataset(df_t_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon, 32, True)
            K.clear_session()
            ds_val, _ = SequenceBuilder.build_tf_dataset(df_v_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon, 32)
            model_obj = build_agfa_model(m_name.replace("WPD-", ""), final_seq_len, n_features, wpd_indices, context_indices, gater_indices, best_params.get('units', 64), best_params.get('dropout', 0.2), best_params.get('lr', 0.001), q_labels, self.config.forecast_horizon)
            self.log(f"[{m_name} F{fold_idx}] Commencing Final Full-Scale Training...", "info")
            
            elite_callbacks = [
                CancelCallback(self.is_cancelled), 
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), 
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                EpochProgressCallback(self.log, m_name, fold_idx, 100)
            ]
            
            model_obj.fit(ds_train, validation_data=ds_val, epochs=100, verbose=0, callbacks=elite_callbacks)
            t_duration = time.time() - t_start

        elif is_ml:
            self.log(f"[{m_name} F{fold_idx}] Booting TensorFlow...", "info")
            t_start = time.time()
            if "NGBoost" in m_name:
                X_tr_seq, Y_tr_seq_sc, _, _ = SequenceBuilder.build_sequences(df_t_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon)
                model_obj = MultiHorizonNGBoost(self.config.forecast_horizon)
                model_obj.fit(X_tr_seq.reshape(X_tr_seq.shape[0], -1), Y_tr_seq_sc)
            else:
                ds_train, _ = SequenceBuilder.build_tf_dataset(df_t_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon, 32, True)
                K.clear_session()
                ds_val, _ = SequenceBuilder.build_tf_dataset(df_v_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon, 32)
                model_obj = build_agfa_model(m_name.replace("WPD-", ""), final_seq_len, n_features, wpd_indices, context_indices, gater_indices, best_params.get('units', 64), best_params.get('dropout', 0.2), best_params.get('lr', 0.001), q_labels, self.config.forecast_horizon)
                
                elite_callbacks = [
                    CancelCallback(self.is_cancelled), 
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), 
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                    EpochProgressCallback(self.log, m_name, fold_idx, 100)
                ]
                
                model_obj.fit(ds_train, validation_data=ds_val, epochs=100, verbose=0, callbacks=elite_callbacks)
            t_duration = time.time() - t_start

        if is_ml and not is_postprocessor:
            try:
                X_v_seq, Y_v_seq_sc, _, _ = SequenceBuilder.build_sequences(df_v_sub_trans, prep.feature_cols, self.config.target_col, final_seq_len, self.config.forecast_horizon)
                if len(Y_v_seq_sc) > 0:
                    feat_report['xai_importance'] = Evaluator.compute_permutation_importance(model_obj, X_v_seq, Y_v_seq_sc, prep.feature_cols, q_labels, prep.capacity)
            except Exception as e: self.log(f"XAI Extraction bypassed: {e}", "warning")

        if not self.is_cancelled() and is_ml:
            manifest_to_save = {
                'feature_cols': list(prep.feature_cols),
                'hyperparams': best_params,
                'xai_report': feat_report,
                'n_features': n_features
            }
            self.cache.save_cache(cache_key, model_obj, prep, self.config, self.site.name, m_name, fold_idx, n_features, q_labels, manifest_to_save)

        return model_obj, prep, best_params, cached, degraded, t_duration, feat_report
    
    def _execute_predictions(self, m_name: str, df_train: pd.DataFrame, df_all: pd.DataFrame, prep: FoldPreprocessor, model_obj: Any, X_ts_seq: np.ndarray, iss_t: pd.DatetimeIndex, val_t_matrix: np.ndarray, y_base_pers: np.ndarray, q_labels: List[float]) -> Tuple[Dict[float, np.ndarray], float]:
        y_quantiles, crossing_pre = {}, 0.0
        if m_name == 'Persistence':
            resid = df_train[self.config.target_col].values - BaselineEngine.make_persistence(df_train, self.config.target_col, df_train.index)
            y_quantiles = BaselineEngine.make_empirical_intervals(resid, y_base_pers, prep.capacity, q_labels)
        elif "NGBoost" in m_name:
            q_dict = model_obj.predict_quantiles(X_ts_seq.reshape(X_ts_seq.shape[0], -1), q_labels)
            Q_raw_flat = np.zeros((len(X_ts_seq) * self.config.forecast_horizon, len(q_labels)))
            for i, q in enumerate(q_labels): Q_raw_flat[:, i] = prep.scaler_y.inverse_transform(q_dict[q].flatten().reshape(-1, 1)).flatten()
            for i, q in enumerate(q_labels): y_quantiles[q] = np.sort(Q_raw_flat, axis=-1)[:, i]
        elif m_name.startswith("CopulaBayes"):
            base_name = m_name.split('-', 1)[1] if '-' in m_name else "WPD-LSTM"
            base_key = self.cache.get_cache_key(self.config.get_hash(), self.config.dataset_hash, self.site.name, base_name, 1)
            base_manifest = self.cache.check_cache(base_key, self.config, self.site.name, base_name, 1)
            t_idx = list(base_manifest.get('feature_cols', [])).index(self.config.target_col) if base_manifest and 'feature_cols' in base_manifest and self.config.target_col in base_manifest['feature_cols'] else 0
            base_model_obj, _ = self.cache.load_cache(base_key) if base_manifest else (None, None)
            q_dict_scaled = model_obj.predict_quantiles(base_model_obj.predict(X_ts_seq, verbose=0)[:, :, q_labels.index(0.5)], estimate_solar_elevation(pd.DatetimeIndex(val_t_matrix.flatten()), self.site.lat, self.site.lon).reshape(val_t_matrix.shape), q_labels)
            Q_raw_flat = np.zeros((len(X_ts_seq) * self.config.forecast_horizon, len(q_labels)))
            for i, q in enumerate(q_labels): Q_raw_flat[:, i] = prep.scaler_y.inverse_transform(q_dict_scaled[q].flatten().reshape(-1, 1)).flatten()
            for i, q in enumerate(q_labels): y_quantiles[q] = np.sort(Q_raw_flat, axis=-1)[:, i]
        else: 
            Q_raw_reshaped = model_obj.predict(X_ts_seq, verbose=0).reshape(-1, len(q_labels))
            Q_raw_flat = np.zeros((len(X_ts_seq) * self.config.forecast_horizon, len(q_labels)))
            for i, q in enumerate(q_labels): Q_raw_flat[:, i] = prep.scaler_y.inverse_transform(Q_raw_reshaped[:, i].reshape(-1, 1)).flatten()
            crossing_pre = np.mean(np.any(np.diff(Q_raw_flat, axis=-1) < 0, axis=-1))
            for i, q in enumerate(q_labels): y_quantiles[q] = np.sort(Q_raw_flat, axis=-1)[:, i]
        return y_quantiles, crossing_pre

class WorkerThread(QThread):
    progress, log, status, finished_run, error, model_finished = Signal(int), Signal(str), Signal(str), Signal(list), Signal(str), Signal(object)
    def __init__(self, configs: List[ExperimentConfig], sites: List[SiteContext]) -> None: 
        super().__init__(); self.configs = configs; self.sites = sites; self._is_cancelled = False
    def cancel(self) -> None: self._is_cancelled = True
    def is_cancelled(self) -> bool: return self._is_cancelled
    def run(self) -> None:
        all_res = []
        try:
            for cfg, site in zip(self.configs, self.sites):
                res = TrainingEngine(config=cfg, site_ctx=site, log_cb=self.log.emit, prog_cb=self.progress.emit, status_cb=self.status.emit, is_cancelled_func=self.is_cancelled, model_done_cb=self.model_finished.emit).run()
                all_res.extend(res)
            self.finished_run.emit(all_res)
        except CancelledError: self.log.emit("Execution terminated by operator."); self.finished_run.emit(all_res)
        except Exception as e: logger.error(f"Worker Crash: {traceback.format_exc()}"); self.error.emit(str(e))