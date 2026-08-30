# data/sequences.py
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view

from data.features import FoldPreprocessor
from core.datatypes import CausalityLeakError

class PVSequenceGenerator(tf.keras.utils.Sequence):
    """Memory-efficient sequence generator for Keras fit loops."""
    def __init__(self, X, Y, seq_len, horizon, batch_size=64, drop_remainder=False):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.valid_indices = np.arange(len(X) - seq_len - horizon + 1)
        
    def __len__(self):
        if self.drop_remainder: return len(self.valid_indices) // self.batch_size
        return int(np.ceil(len(self.valid_indices) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_inds = self.valid_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = np.empty((len(batch_inds), self.seq_len, self.X.shape[1]), dtype=np.float32)
        batch_y = np.empty((len(batch_inds), self.horizon), dtype=np.float32)
        for i, start_idx in enumerate(batch_inds):
            batch_x[i] = self.X[start_idx : start_idx + self.seq_len]
            batch_y[i] = self.Y[start_idx + self.seq_len : start_idx + self.seq_len + self.horizon]
        return batch_x, batch_y

class SequenceBuilder:
    """Builder methods for 3D tensors using NumPy stride tricks."""
    @staticmethod
    def build_tf_dataset(df: pd.DataFrame, feature_cols: list, target_col: str, 
                         seq_len: int, horizon: int, batch_size: int = 64, drop_remainder: bool = False):
        features = df[feature_cols].values.astype(np.float32)
        targets = df[target_col].values.astype(np.float32)
        if len(df) < seq_len + horizon: return None, 0
        gen = PVSequenceGenerator(features, targets, seq_len, horizon, batch_size, drop_remainder=drop_remainder)
        return gen, len(gen.valid_indices)

    @staticmethod
    def build_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, seq_len: int, horizon: int):
        features = df[feature_cols].values.astype(np.float32)
        targets = df[target_col].values.astype(np.float32)
        times = df.index
        N = len(df)
        
        if N < seq_len + horizon: 
            return np.array([]), np.array([]), pd.DatetimeIndex([]), np.array([])
        
        swv_feat = sliding_window_view(features, window_shape=(seq_len, features.shape[1]))
        X = swv_feat[:N - seq_len - horizon + 1, 0, :, :].copy() 
        swv_targ = sliding_window_view(targets, window_shape=(horizon,))
        Y = swv_targ[seq_len : N - horizon + 1, :].copy()
        iss_idx = times[seq_len - 1 : N - horizon]
        swv_times = sliding_window_view(times.values, window_shape=(horizon,))
        val_matrix = swv_times[seq_len : N - horizon + 1, :].copy()
        
        return X, Y, iss_idx, val_matrix

    @staticmethod
    def assert_causality_guard(df: pd.DataFrame, prep: FoldPreprocessor, seq_len: int, horizon: int, target_col: str):
        """Injects forward-looking noise to strictly prove data leakage does not exist."""
        if len(df) < seq_len + horizon + 5: return
        test_indices = np.random.choice(range(seq_len, len(df) - horizon - 1), size=min(5, len(df)//10), replace=False)
        for issue_idx in test_indices:
            df_copy = df.copy()
            corrupt_start = issue_idx + 1
            df_copy.iloc[corrupt_start:, df_copy.columns.get_loc(target_col)] = np.random.normal(9999, 100, len(df_copy)-corrupt_start)
            
            X1, _, _, _ = SequenceBuilder.build_sequences(prep.transform(df), prep.feature_cols, target_col, seq_len, horizon)
            X2, _, _, _ = SequenceBuilder.build_sequences(prep.transform(df_copy), prep.feature_cols, target_col, seq_len, horizon)
            
            loc = issue_idx - seq_len + 1
            if len(X1) > loc and len(X2) > loc:
                diff = np.nanmax(np.abs(X1[loc] - X2[loc]))
                if diff > 1e-12: 
                    raise CausalityLeakError(f"Causality Violation! Future corruption altered historical states. Max Diff: {diff}")