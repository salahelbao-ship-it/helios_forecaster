# data/loader.py
import pandas as pd
import numpy as np
import hashlib
from typing import Tuple, Dict, Any

from core.datatypes import DataQualityError, AlignmentError

class AlignmentGuard:
    """Strict mathematical checks to ensure prediction arrays match valid time indices."""
    @staticmethod
    def validate(y_true: np.ndarray, y_pred: np.ndarray, valid_times: pd.DatetimeIndex, 
                 y_base: np.ndarray = None, base_times: pd.DatetimeIndex = None):
        if len(y_true) != len(y_pred) or len(y_true) != len(valid_times):
            raise AlignmentError(
                f"Length mismatch: y_true({len(y_true)}), y_pred({len(y_pred)}), times({len(valid_times)})"
            )
        if y_base is not None:
            if base_times is None or len(y_pred) != len(y_base) or not valid_times.equals(base_times):
                raise AlignmentError("Timestamp sequence strict mismatch between model and baseline.")

class DataQualityAuditor:
    """Automated QA pipeline to detect spikes, negatives, and frequency shifts."""
    @staticmethod
    def infer_capacity(df: pd.DataFrame, target_col: str) -> float:
        p999 = float(df[target_col].quantile(0.999))
        day_mask = df[target_col] > df[target_col].mean()
        if day_mask.sum() > 100:
            p95_day = float(df.loc[day_mask, target_col].quantile(0.95))
            return max(p999, p95_day, 1.0)
        return max(p999, 1.0)

    @staticmethod
    def run_audit(df: pd.DataFrame, target_col: str, modify_target: bool = True, 
                  strict_mode: bool = False) -> Tuple[pd.DataFrame, Dict]:
        report = {
            'issues': {}, 'actions': {}, 'thresholds': {}, 
            'policy': {'modify_target': modify_target, 'strict_mode': strict_mode}
        }
        df_clean = df.copy()
        
        dups = df_clean.index.duplicated(keep='first')
        report['issues']['duplicates_found'] = int(dups.sum())
        if dups.any(): 
            if strict_mode: raise DataQualityError(f"Strict QA: Found {dups.sum()} duplicates.")
            df_clean = df_clean[~dups]
            report['actions']['duplicates_dropped'] = int(dups.sum())
        
        if not df_clean.index.is_monotonic_increasing:
            report['issues']['non_monotonic_index'] = True
            df_clean = df_clean.sort_index()
            report['actions']['sorted_index'] = True
            
        deltas = df_clean.index.to_series().diff().dropna()
        if not deltas.empty:
            median_delta = deltas.median()
            std_delta = deltas.std()
            deviation = (std_delta.total_seconds() / median_delta.total_seconds()) if median_delta.total_seconds() > 0 else 0
            mode = "REGULAR_VALIDATED" if deviation <= 0.05 else "IRREGULAR_INFERRED"
            report['issues']['interval_cv'] = float(deviation)
            report['issues']['median_interval_sec'] = float(median_delta.total_seconds())
            inferred_freq = str(median_delta)
        else:
            mode, inferred_freq = "IRREGULAR_UNKNOWN", "Unknown"
            
        report['issues']['inferred_freq'] = inferred_freq
        report['issues']['frequency_mode'] = mode
        
        neg_mask = df_clean[target_col] < 0
        report['issues']['negatives_found'] = int(neg_mask.sum())
        if neg_mask.any():
            if strict_mode: raise DataQualityError(f"Strict QA: Found {neg_mask.sum()} negatives.")
            if modify_target:
                df_clean.loc[neg_mask, target_col] = 0.0
                report['actions']['negatives_zeroed'] = int(neg_mask.sum())
            
        exp_p99 = df_clean[target_col].expanding(min_periods=1).quantile(0.99)
        spike_mask = df_clean[target_col] > (exp_p99 * 5.0 + 1.0)
        report['issues']['outlier_spikes_found'] = int(spike_mask.sum())
        report['thresholds']['spike_multiplier'] = 5.0
        if spike_mask.any(): 
            if strict_mode: raise DataQualityError(f"Strict QA: Found {spike_mask.sum()} outlier spikes.")
            if modify_target:
                df_clean.loc[spike_mask, target_col] = exp_p99[spike_mask] * 5.0
                report['actions']['outlier_spikes_clipped'] = int(spike_mask.sum())
                
        report['issues']['missing_pct'] = float(df_clean[target_col].isna().mean() * 100)
        if strict_mode and report['issues']['missing_pct'] > 5.0:
            raise DataQualityError(f"Strict QA: {report['issues']['missing_pct']}% missing target data.")
        
        return df_clean, report

class DataLoader:
    """Robust parser for messy CSV and Excel telemetry files."""
    
    @staticmethod
    def hash_dataset(path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                while chunk := f.read(8192): 
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception: 
            return "unknown_hash"

    @staticmethod
    def _parse_time_robust(t: Any) -> pd.Timedelta:
        if pd.isna(t): return pd.NaT
        if isinstance(t, str):
            t = t.strip()
            try: return pd.Timedelta(t)
            except: pass
        if hasattr(t, 'hour'): 
            return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=getattr(t, 'second', 0))
        return pd.NaT

    @staticmethod
    def load_df(path: str, tz: str = None, modify_target: bool = True, 
                strict_mode: bool = False, target_col: str = None) -> Tuple[pd.DataFrame, dict, str]:
        try:
            if path.endswith('.csv'): df_raw = pd.read_csv(path)
            else:
                xl = pd.ExcelFile(path)
                sheet_name = 'Feuil1' if 'Feuil1' in xl.sheet_names else 0
                df_raw = pd.read_excel(xl, sheet_name=sheet_name)

            df_raw.columns = df_raw.columns.astype(str).str.strip()
            cols_lower = {c.lower(): c for c in df_raw.columns}
            
            # Identify Date/Time columns safely
            date_col = next((c for c in df_raw.columns if c.lower() in ['date', 'date_local', 'datetime', 'timestamp', 'date ']), None)
            time_col = next((c for c in df_raw.columns if 'time' in c.lower()), None)
            
            # Temporal Fusion
            if date_col and time_col:
                dates = pd.to_datetime(df_raw[date_col], errors='coerce')
                times = df_raw[time_col].apply(DataLoader._parse_time_robust)
                dt_index = dates + times
                invalid_mask = dt_index.isna()
                df = df_raw[~invalid_mask].copy()
                df.index = dt_index[~invalid_mask]
                df.drop(columns=[date_col, time_col], inplace=True, errors='ignore')
            elif date_col:
                df_raw.set_index(date_col, inplace=True)
                df_raw.index = pd.to_datetime(df_raw.index, errors='coerce')
                df = df_raw[~df_raw.index.isna()].copy()
            else:
                df_raw.set_index(df_raw.columns[0], inplace=True)
                df_raw.index = pd.to_datetime(df_raw.index, errors='coerce')
                df = df_raw[~df_raw.index.isna()].copy()

            df.sort_index(inplace=True)
            
            # Strip duplicate timestamps immediately
            df = df[~df.index.duplicated(keep='first')]

            # Infer Target Column safely
            if target_col and target_col.lower() in cols_lower:
                t_col = cols_lower[target_col.lower()]
            elif target_col and target_col in df.columns:
                t_col = target_col
            else:
                candidates = [c for c in df.columns if any(k in c.lower() for k in ['pv', 'power', 'dcp', 'target'])]
                t_col = candidates[-1] if candidates else df.columns[-1]

            # Convert all to numeric to prevent strict modeling errors
            df = df.apply(pd.to_numeric, errors='coerce')

            # TZ Localization BEFORE Grid Enforcement
            tz_nats = 0
            if tz and tz.lower() not in ['unknown', '']:
                if df.index.tz is None: 
                    df.index = df.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
                else: 
                    df.index = df.index.tz_convert(tz)
                tz_nats = df.index.isna().sum()
                df = df[~df.index.isna()].copy()

            # --- STRICT TEMPORAL ENFORCEMENT ---
            freq_str = '15min'
            
            if not df.empty:
                # 1. Create a mathematically perfect, continuous timeline
                full_index = pd.date_range(
                    start=df.index.min().floor('D'), 
                    end=df.index.max().ceil('D'), 
                    freq=freq_str, 
                    tz=df.index.tz
                )
                
                # 2. Resample raw jittery data into strict 15-minute bins
                df_resampled = df.resample(freq_str).mean()
                
                # 3. Align the resampled data onto the perfect timeline (Exposing the night gaps)
                df_grid = df_resampled.reindex(full_index)
                
                # 4. Intelligent Gap Filling (Interpolate dropouts up to 1 hour / 4 steps)
                df_grid = df_grid.interpolate(method='time', limit=4)
                
                # 5. Fix the Night-Jump Corruption (Remaining NaNs are nights, set target to 0)
                if t_col in df_grid.columns:
                    df_grid[t_col] = df_grid[t_col].fillna(0.0)
                
                # 6. Forward fill weather variables (Tamb, GHI, Tpan) through the night, then 0
                df_grid = df_grid.ffill().fillna(0.0)
                df = df_grid

            # Run Auditor on the pristine, enforced grid
            df, qa_report = DataQualityAuditor.run_audit(df, t_col, modify_target, strict_mode)
            qa_report['actions']['tz_nats_dropped'] = int(tz_nats)
            qa_report['issues']['frequency_mode'] = 'STRICT_FORCED'
            qa_report['issues']['inferred_freq'] = freq_str
            
            return df, qa_report, t_col
            
        except Exception as e:
            raise DataQualityError(f"Dataset load failed: {str(e)}")