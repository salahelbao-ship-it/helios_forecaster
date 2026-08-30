# data/database.py
import os
import json
import sqlite3
import hashlib
import pickle
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from core.datatypes import (
    ExperimentResult, FoldResult, FoldDefinition, ExperimentConfig, SiteContext
)

class CacheManager:
    """Handles deterministic disk caching for ML Models and Preprocessors."""
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        self.manifest = self._load_manifest()
        
    def _load_manifest(self) -> Dict:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f: return json.load(f)
            except Exception: return {}
        return {}
        
    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f: json.dump(self.manifest, f, indent=4)
        
    def get_cache_key(self, config_hash: str, dataset_hash: str, site_name: str, model_name: str, fold_idx: int) -> str:
        raw = f"{config_hash}_{dataset_hash}_{site_name}_{model_name}_F{fold_idx}"
        return hashlib.md5(raw.encode()).hexdigest()
        
    def check_cache(self, cache_key: str, config: ExperimentConfig, site_name: str, model_name: str, fold_idx: int) -> Optional[Dict]:
        if cache_key in self.manifest:
            entry = self.manifest[cache_key]
            if entry['model'] == model_name and entry['fold'] == fold_idx:
                model_path = os.path.join(self.cache_dir, f"{cache_key}.keras")
                prep_path = os.path.join(self.cache_dir, f"{cache_key}_prep.pkl")
                if os.path.exists(model_path) and os.path.exists(prep_path): return entry
        return None
        
    def save_cache(self, cache_key: str, model_obj: Any, prep_obj: Any, config: ExperimentConfig, site_name: str, model_name: str, fold_idx: int, n_features: int, q_labels: List[float], best_params: dict):
        try:
            model_path = os.path.join(self.cache_dir, f"{cache_key}.keras")
            prep_path = os.path.join(self.cache_dir, f"{cache_key}_prep.pkl")
            
            if hasattr(model_obj, 'save'): model_obj.save(model_path)
            else:
                with open(model_path, 'wb') as f: pickle.dump(model_obj, f)
                
            with open(prep_path, 'wb') as f: pickle.dump(prep_obj, f)
            
            self.manifest[cache_key] = {
                'timestamp': datetime.now().isoformat(), 'site': site_name, 'model': model_name,
                'fold': fold_idx, 'n_features': n_features, 'quantiles': q_labels,
                'hyperparams': best_params, 'config_hash': config.get_hash()
            }
            self._save_manifest()
        except Exception as e: print(f"Cache save failed: {e}")
        
    def load_cache(self, cache_key: str, custom_objects=None) -> Tuple[Any, Any]:
        model_path = os.path.join(self.cache_dir, f"{cache_key}.keras")
        prep_path = os.path.join(self.cache_dir, f"{cache_key}_prep.pkl")
        
        with open(prep_path, 'rb') as f: prep_obj = pickle.load(f)
        
        try:
            import tensorflow as tf
            model_obj = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception:
            with open(model_path, 'rb') as f: model_obj = pickle.load(f)
            
        return model_obj, prep_obj

class ArtifactStore:
    """Manages heavy binary artifacts to completely bypass SQLite JSON lag bottlenecks."""
    def __init__(self, storage_dir="artifacts"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
    def store_fold_data(self, exp_fingerprint: str, fold_idx: int, payload: dict) -> str:
        file_name = f"payload_{exp_fingerprint}_f{fold_idx}.bin"
        file_path = os.path.join(self.storage_dir, file_name)
        compressed_data = zlib.compress(pickle.dumps(payload))
        with open(file_path, 'wb') as f:
            f.write(compressed_data)
        return file_name
        
    def retrieve_fold_data(self, file_name: str) -> dict:
        file_path = os.path.join(self.storage_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.loads(zlib.decompress(f.read()))
        return {}

class ExperimentRegistry:
    """SQLite Database Interface for persisting telemetry configurations and experiment metrics."""
    def __init__(self, db_path="experiments.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS sites (
                            name TEXT PRIMARY KEY, file_path TEXT, target_col TEXT,
                            lat REAL, lon REAL, timezone TEXT, capacity REAL, 
                            technology_type TEXT, location_label TEXT, exog_cols TEXT, inferred_flags TEXT
                        )''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS experiments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, run_fingerprint TEXT,
                            model_name TEXT, site_name TEXT, config_id TEXT,
                            timestamp TEXT, capacity REAL, horizon INTEGER, seq_len INTEGER,
                            training_time REAL, rmse_mean REAL, env_meta TEXT, site_meta TEXT, 
                            qa_report TEXT, config_dump TEXT, methodology TEXT, warnings TEXT, errors TEXT
                        )''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS folds (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id INTEGER, fold_idx INTEGER,
                            train_start TEXT, train_end TEXT, test_start TEXT, test_end TEXT,
                            metrics TEXT, pinball TEXT, params TEXT, y_true TEXT, y_pred_q TEXT, 
                            base_pers TEXT, issue_times TEXT, valid_times TEXT, crossing_rate REAL, 
                            train_dur REAL, cached BOOLEAN, degraded BOOLEAN, data_file_path TEXT, 
                            artifact_hash TEXT, feature_report TEXT, file_size INTEGER,
                            FOREIGN KEY(exp_id) REFERENCES experiments(id)
                        )''')
            conn.commit()

    def save_site(self, site: SiteContext):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO sites 
                         (name, file_path, target_col, lat, lon, timezone, capacity, technology_type, location_label, exog_cols, inferred_flags)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (site.name, site.file_path, site.target_col, site.lat, site.lon, site.timezone, site.capacity, 
                       site.technology_type, site.location_label, json.dumps(site.exog_cols), json.dumps(site.inferred_flags)))
            conn.commit()

    def load_sites(self) -> List[SiteContext]:
        sites = []
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM sites")
            for r in c.fetchall():
                sites.append(SiteContext(
                    name=r[0], file_path=r[1], target_col=r[2], lat=r[3], lon=r[4], timezone=r[5], 
                    capacity=r[6], technology_type=r[7], location_label=r[8],
                    exog_cols=json.loads(r[9]), inferred_flags=json.loads(r[10])
                ))
        return sites

    def delete_site(self, name: str):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM sites WHERE name=?", (name,))
            conn.commit()
            
    def save_exp(self, exp: ExperimentResult) -> int:
        import numpy as np 
        
        def get_mean_rmse(f):
            rmses = [v for k, v in f.metrics.items() if 'RMSE_L' in k and v is not None]
            return float(np.mean(rmses)) if rmses else float(f.metrics.get('RMSE', 0.0))
            
        rmse_mean = float(np.mean([get_mean_rmse(f) for f in exp.folds])) if exp.folds else 0.0
        
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO experiments 
                         (run_fingerprint, model_name, site_name, config_id, timestamp, capacity, horizon, seq_len, 
                          training_time, rmse_mean, env_meta, site_meta, qa_report, config_dump, methodology, warnings, errors)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (exp.run_fingerprint, exp.model_name, exp.site_name, exp.config_id, exp.timestamp_created,
                       exp.capacity, exp.horizon, exp.seq_len, exp.training_time_total, rmse_mean,
                       json.dumps(exp.env_meta), json.dumps(exp.site_meta), json.dumps(exp.qa_report),
                       json.dumps(exp.config_dump), exp.methodology, json.dumps(exp.warnings), json.dumps(exp.errors)))
            exp_id = c.lastrowid
            
            store = ArtifactStore()
            
            for f in exp.folds:
                # Bypass SQLite lag for massive arrays using fast compressed binaries
                payload = {
                    'y_true': f.y_true, 'y_pred_q': f.y_pred_quantiles,
                    'base_pers': f.baseline_pers, 'issue_times': f.issue_times, 'valid_times': f.valid_times
                }
                bin_file = store.store_fold_data(exp.run_fingerprint, f.fold_id, payload)
                
                # Utilisation de getattr() pour garantir que l'enregistrement DB ne crashe pas si
                # l'objet FoldDefinition ou FoldResult n'a pas les anciens arguments.
                t_start = getattr(f.fold_def, 'train_start', "")
                t_end = getattr(f.fold_def, 'train_end', "")
                v_start = getattr(f.fold_def, 'val_start', "")
                v_end = getattr(f.fold_def, 'val_end', "")
                
                c.execute('''INSERT INTO folds 
                             (exp_id, fold_idx, train_start, train_end, test_start, test_end,
                              metrics, pinball, params, y_true, y_pred_q, base_pers, issue_times, valid_times,
                              crossing_rate, train_dur, cached, degraded, data_file_path, artifact_hash, feature_report, file_size)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (exp_id, f.fold_id, t_start, t_end, v_start, v_end,
                           json.dumps(f.metrics), json.dumps(f.pinball_by_q), json.dumps(f.params),
                           "", "", "", "", "", # Blanked out string arrays to eliminate database bloat
                           f.crossing_rate_pre_repair, f.training_duration_sec, f.loaded_from_cache, False, bin_file, "",
                           json.dumps(f.feature_selection_report), 0))
            conn.commit()
            return exp_id

    def load(self, exp_id: int) -> ExperimentResult:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,))
            r = c.fetchone()
            if not r: raise ValueError(f"Experiment {exp_id} not found.")
            
            c.execute("SELECT * FROM folds WHERE exp_id=?", (exp_id,))
            folds = []
            store = ArtifactStore()
            
            for fr in c.fetchall():
                # Lightning fast array resurrection
                payload = store.retrieve_fold_data(fr[19])
                if payload:
                    y_true = payload.get('y_true', [])
                    y_pred_safe = {str(k): v for k, v in payload.get('y_pred_q', {}).items()}
                    base_pers = payload.get('base_pers', [])
                    issue_times = payload.get('issue_times', [])
                    valid_times = payload.get('valid_times', [])
                else: # Fallback to standard SQLite
                    y_true = json.loads(fr[10]) if fr[10] else []
                    y_pred_safe = {str(k): v for k, v in (json.loads(fr[11]) if fr[11] else {}).items()}
                    base_pers = json.loads(fr[12]) if fr[12] else []
                    issue_times = json.loads(fr[13]) if fr[13] else []
                    valid_times = json.loads(fr[14]) if fr[14] else []

                # CRITICAL FIX: Ne passer QUE les arguments acceptés par la dataclass moderne.
                # Nous ignorons totalement fr[18] (degraded), fr[19] (data_file), fr[20] (hash), fr[22] (file_size)
                fd = FoldDefinition(
                    fold_id=fr[2], 
                    train_start=fr[3], 
                    train_end=fr[4], 
                    val_start=fr[5], 
                    val_end=fr[6], 
                    train_size=0, 
                    val_size=0, 
                    gap_size=0, 
                    inferred_freq=""
                )
                
                folds.append(FoldResult(
                    fold_id=fr[2], 
                    issue_times=issue_times, 
                    valid_times=valid_times,
                    y_true=y_true, 
                    y_pred_quantiles=y_pred_safe, 
                    baseline_pers=base_pers,
                    metrics=json.loads(fr[7]), 
                    pinball_by_q=json.loads(fr[8]), 
                    params=json.loads(fr[9]),
                    fold_def=fd, 
                    crossing_rate_pre_repair=fr[15], 
                    training_duration_sec=fr[16],
                    loaded_from_cache=bool(fr[17]), 
                    feature_selection_report=json.loads(fr[21])
                ))
            
            return ExperimentResult(
                model_name=r[2], site_name=r[3], config_id=r[4], seq_len=r[8], timestamp_created=r[5],
                capacity=r[6], horizon=r[7], training_time_total=r[9], run_fingerprint=r[1],
                env_meta=json.loads(r[11]), site_meta=json.loads(r[12]), qa_report=json.loads(r[13]),
                config_dump=json.loads(r[14]), methodology=r[15], warnings=json.loads(r[16]),
                errors=json.loads(r[17]), folds=folds
            )

    def list_all(self) -> List[Dict]:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            c = conn.cursor()
            c.execute("SELECT id, timestamp, model_name, site_name, run_fingerprint FROM experiments ORDER BY id DESC")
            return [{'id': r[0], 'timestamp': r[1], 'model': r[2], 'site': r[3], 'run_fingerprint': r[4]} for r in c.fetchall()]