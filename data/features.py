# data/features.py
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

def estimate_solar_elevation(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    if lat is None or lon is None: return np.full(len(times), np.nan)
    times_utc = times.tz_convert('UTC') if times.tz is not None else times.tz_localize('UTC')
    doy = times_utc.dayofyear.values
    hour = times_utc.hour.values + times_utc.minute.values/60.0
    declination = 23.45 * np.sin(np.radians((360/365) * (doy - 81)))
    lst = hour + (lon / 15.0)
    omega = 15 * (lst - 12)
    lat_rad, dec_rad, omega_rad = np.radians(lat), np.radians(declination), np.radians(omega)
    sin_alpha = np.sin(lat_rad)*np.sin(dec_rad) + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(omega_rad)
    return np.degrees(np.arcsin(np.clip(sin_alpha, -1.0, 1.0)))

class NonLinearFeatureSelector:
    @staticmethod
    def select(df_train: pd.DataFrame, target_col: str, candidates: List[str], lat: float, lon: float, horizon: int, exog_avail: dict, threshold: float=0.03, top_k: int=12) -> Tuple[List[str], Dict]:
        report = {'threshold': threshold, 'candidates': candidates, 'selected': [], 'rejected': []}
        if not candidates: return [], report
        
        elev = estimate_solar_elevation(df_train.index, lat, lon) if lat and lon else np.ones(len(df_train))
        day_mask = elev > 5.0
        df_day = df_train[day_mask].copy()
        
        if len(df_day) < 50: return candidates, report
        target_series = df_day[target_col].values
        scored = []
        
        for ex in candidates:
            if ex not in df_train.columns or ex == target_col: continue
            best_mi = 0.0
            for lag in range(1, min(6, horizon * 2) + 1):
                ex_shifted = df_train[ex].shift(lag)[day_mask]
                valid_mask = ~np.isnan(ex_shifted) & ~np.isnan(target_series)
                if valid_mask.sum() < 50: continue
                ex_valid = ex_shifted[valid_mask].values.reshape(-1, 1)
                y_valid = target_series[valid_mask]
                mi_score = mutual_info_regression(ex_valid, y_valid, random_state=42)[0]
                if mi_score > best_mi: best_mi = mi_score
            
            if best_mi >= threshold: scored.append((best_mi, ex))
            else: report['rejected'].append(ex)
            
        scored.sort(reverse=True, key=lambda x: x[0])
        selected = [x[1] for x in scored[:top_k]]
        report['selected'] = selected
        return selected, report

class FoldPreprocessor:
    def __init__(self, target_col: str, use_adv: bool, use_ramp: bool, use_wpd: bool, horizon: int, exog_avail: dict, freq_str: str, freq_mode: str, lat: float, lon: float, exog_cols: List[str]):
        self.target_col = target_col
        self.use_adv = use_adv
        self.use_ramp = use_ramp
        self.use_wpd = use_wpd and HAS_PYWT
        self.horizon = horizon
        self.exog_avail = exog_avail
        self.freq_str = freq_str
        self.freq_mode = freq_mode
        self.lat = lat
        self.lon = lon
        self.exog_cols = exog_cols
        
        # ACADEMIC FIX: Split transformation pipelines
        self.scaler_X_exog = StandardScaler() 
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.iso_forest = IsolationForest(contamination=0.01, random_state=42)
        
        self.feature_cols = []
        self.capacity = 1.0
        self.wpd_failures = 0
        self.feat_report = {}

    def _safe_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear').bfill().ffill()
        return df_clean

    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        target = df[self.target_col].values
        feats = [target.reshape(-1, 1)]
        cols = [self.target_col]
        
        for ex in self.exog_cols:
            if ex in df.columns and ex != self.target_col:
                feats.append(df[ex].values.reshape(-1, 1))
                cols.append(ex)
                
        if self.use_ramp:
            ramp = np.zeros_like(target)
            ramp[1:] = target[1:] - target[:-1]
            feats.append(ramp.reshape(-1, 1))
            cols.append('feat_ramp')

        if self.use_adv:
            months = df.index.month.values
            feats.append(np.isin(months, [12, 1, 2]).astype(float).reshape(-1, 1)); cols.append('feat_winter')
            feats.append(np.isin(months, [6, 7, 8]).astype(float).reshape(-1, 1)); cols.append('feat_summer')
            if self.lat is not None and self.lon is not None:
                feats.append(estimate_solar_elevation(df.index, self.lat, self.lon).reshape(-1, 1))
                cols.append('feat_elevation')

        if self.use_wpd:
            n = len(target)
            try:
                level = 3
                pad_len = int(np.ceil(n / (2**level)) * (2**level)) - n
                padded_target = np.pad(target, (pad_len, 0), mode='reflect')
                swt_coeffs = pywt.swt(padded_target, 'haar', level=level, trim_approx=True) 
                for j, c_array in enumerate(swt_coeffs):
                    feats.append(c_array[pad_len:].reshape(-1, 1))
                    cols.append(f'feat_swt_level_{j}')
            except Exception:
                self.wpd_failures += n
                for j in range(3):
                    feats.append(np.zeros((n, 1)))
                    cols.append(f'feat_swt_level_{j}')
                    
        return np.hstack(feats), cols

    def fit(self, df_train: pd.DataFrame, provided_cap: float=None):
        self.capacity = provided_cap if provided_cap and provided_cap > 0 else df_train[self.target_col].max()
        target_data = df_train[self.target_col].fillna(0).values.reshape(-1, 1)
        self.iso_forest.fit(target_data)
        
        df_clean = self._safe_impute(df_train)
        X_raw, self.feature_cols = self._extract_features(df_clean)
        
        # ACADEMIC FIX: Mathematically anchor the target scaler to physical absolute limits
        target_bounds = np.array([[0.0], [self.capacity]])
        self.scaler_y.fit(target_bounds)
        
        # Fit exogenous scaler strictly on weather/engineered features (ignoring the target column at index 0)
        if X_raw.shape[1] > 1:
            self.scaler_X_exog.fit(X_raw[:, 1:])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self._safe_impute(df)
        X_raw, current_cols = self._extract_features(df_clean)
        
        if list(current_cols) != list(self.feature_cols):
            df_raw = pd.DataFrame(X_raw, columns=current_cols, index=df.index)
            for col in self.feature_cols:
                if col not in df_raw.columns: df_raw[col] = 0.0
            X_raw = df_raw[self.feature_cols].values
            
        X_scaled = np.zeros_like(X_raw)
        
        # ACADEMIC FIX: Explicitly route the target through its dedicated MinMaxScaler
        X_scaled[:, 0] = self.scaler_y.transform(X_raw[:, 0].reshape(-1, 1)).flatten()
        
        # Route all exogenous features through the StandardScaler
        if X_raw.shape[1] > 1:
            X_scaled[:, 1:] = self.scaler_X_exog.transform(X_raw[:, 1:])
            
        return pd.DataFrame(X_scaled, columns=self.feature_cols, index=df.index)