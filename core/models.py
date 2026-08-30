# core/models.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Bidirectional, Dropout, Reshape, 
    Conv1D, SeparableConv1D, BatchNormalization, LayerNormalization, 
    MultiHeadAttention, Flatten, Layer, Add, Lambda, Concatenate, Softmax
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

class CancelCallback(Callback):
    def __init__(self, is_cancelled_func):
        super().__init__()
        self.is_cancelled = is_cancelled_func
    def on_epoch_end(self, epoch, logs=None):
        if self.is_cancelled(): self.model.stop_training = True

class BaselineEngine:
    @staticmethod
    def make_persistence(df: pd.DataFrame, col: str, iss_t: pd.DatetimeIndex) -> np.ndarray:
        try: 
            return df.loc[iss_t, col].values
        except Exception: 
            try:
                # Hardened Fallback: Strip timezones and force nearest-neighbor alignment
                df_clean = df.copy()
                df_clean.index = df_clean.index.tz_localize(None)
                iss_clean = iss_t.tz_localize(None)
                aligned = df_clean[[col]].reindex(iss_clean, method='nearest', tolerance=pd.Timedelta('15min'))
                return aligned[col].fillna(0.0).values
            except Exception as e:
                print(f"CRITICAL: Persistence alignment failed. {e}")
                return np.zeros(len(iss_t))

    @staticmethod
    def make_empirical_intervals(resid: np.ndarray, y_base: np.ndarray, capacity: float, q_labels: list) -> dict:
        return {q: np.clip(y_base + (np.quantile(resid, q) if len(resid) > 0 else 0.0), 0.0, capacity) for q in q_labels}

class PositionalEncoding(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        seq_len, d_model = input_shape[1], input_shape[2]
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = pos * (1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model)))
        pos_emb = np.zeros((1, seq_len, d_model), dtype=np.float32)
        pos_emb[0, :, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_emb[0, :, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_emb = tf.constant(pos_emb, dtype=tf.float32)
        super().build(input_shape)
    def call(self, inputs): return inputs + self.pos_emb

class ChebyshevKANLinear(Layer):
    def __init__(self, out_features, degree=4, **kwargs):
        super().__init__(**kwargs)
        self.out_features, self.degree = out_features, degree
    def build(self, input_shape):
        in_features = input_shape[-1]
        self.base_weight = self.add_weight(shape=(in_features, self.out_features), initializer='glorot_uniform', name='base_w')
        self.poly_weight = self.add_weight(shape=(in_features, self.degree + 1, self.out_features), initializer='glorot_uniform', name='poly_w')
        super().build(input_shape)
    def call(self, x):
        base_output = tf.matmul(x, self.base_weight)
        x_norm = tf.math.tanh(x) 
        t0, t1 = tf.ones_like(x_norm), x_norm
        t2 = 2.0 * x_norm * t1 - t0
        t3 = 2.0 * x_norm * t2 - t1
        t4 = 2.0 * x_norm * t3 - t2
        poly_output = tf.einsum('...id,ido->...o', tf.stack([t0, t1, t2, t3, t4], axis=-1), self.poly_weight)
        return tf.nn.silu(base_output) + poly_output

class DynamicWeatherGater(Layer):
    def __init__(self, num_experts, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.w_hidden = Dense(32, activation='gelu', name="gater_context")
        self.w_logits = Dense(num_experts, activation='linear', name="gater_routing")
        self.softmax = Softmax(axis=-1)

    def call(self, expert_outputs, gater_features):
        weather_context = tf.reduce_mean(gater_features, axis=1) 
        weights = self.softmax(self.w_logits(self.w_hidden(weather_context)))
        stacked_experts = tf.stack(expert_outputs, axis=-1)
        broadcast_weights = tf.reshape(weights, (-1, 1, 1, self.num_experts))
        return tf.reduce_sum(stacked_experts * broadcast_weights, axis=-1)

def _build_rnn_family(inputs, model_type, units, dropout):
    r_drop = 0.05 
    l2_reg = regularizers.l2(1e-5)
    if "CNN-BiLSTM" in model_type:
        x = Conv1D(filters=units, kernel_size=3, padding='causal', activation='gelu')(inputs)
        x = Bidirectional(LSTM(units, recurrent_dropout=r_drop, recurrent_regularizer=l2_reg, return_sequences=False))(x)
    elif "BiLSTM" in model_type:
        x = Bidirectional(LSTM(units, recurrent_dropout=r_drop, recurrent_regularizer=l2_reg, return_sequences=False))(inputs)
    elif "LSTM" in model_type:
        x = LSTM(units, recurrent_dropout=r_drop, recurrent_regularizer=l2_reg, return_sequences=False)(inputs)
    else:
        x = GRU(units, recurrent_dropout=r_drop, recurrent_regularizer=l2_reg, return_sequences=False)(inputs)
    return Dense(units, activation='selu')(Dropout(dropout)(BatchNormalization()(x)))

def _build_transformer_family(inputs, model_type, units, dropout, num_heads, seq_len, n_feats):
    if "iTransformer" in model_type:
        x_pe = Dense(units, activation='linear')(PositionalEncoding()(inputs))
        c3 = SeparableConv1D(units//3, kernel_size=3, padding='causal', activation='gelu')(x_pe)
        c5 = SeparableConv1D(units//3, kernel_size=5, padding='causal', activation='gelu')(x_pe)
        c7 = SeparableConv1D(units - 2*(units//3), kernel_size=7, padding='causal', activation='gelu')(x_pe)
        x_ms = LayerNormalization(epsilon=1e-6)(tf.concat([c3, c5, c7], axis=-1) + x_pe)
        x_emb = LayerNormalization(epsilon=1e-6)(Dense(units, activation='gelu')(tf.transpose(x_ms, perm=[0, 2, 1])))
        x_attn = LayerNormalization(epsilon=1e-6)(x_emb + MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)(x_emb, x_emb))
        ffn = ChebyshevKANLinear(units, degree=4)(LayerNormalization(epsilon=1e-6)(Dropout(dropout)(ChebyshevKANLinear(units * 2, degree=4)(LayerNormalization(epsilon=1e-6)(x_attn)))))
        return Flatten()(LayerNormalization(epsilon=1e-6)(x_attn + ffn))
    else:
        x_ci = tf.reshape(tf.transpose(PositionalEncoding()(inputs), perm=[0, 2, 1]), [-1, seq_len, 1]) 
        x_patch = LayerNormalization(epsilon=1e-6)(Conv1D(filters=units, kernel_size=max(2, seq_len//4), strides=max(2, seq_len//4)//2, padding="valid", activation="gelu")(x_ci))
        x_attn = LayerNormalization(epsilon=1e-6)(x_patch + MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)(x_patch, x_patch))
        return tf.reshape(Flatten()(x_attn), [-1, n_feats * x_attn.shape[-1]])

def _build_apex_hybrid(inputs, units, dropout, kernel_size, num_heads):
    x_tcn = inputs
    for dilation_rate in [1, 2, 4, 8, 16, 32]:
        x_tcn = LayerNormalization(epsilon=1e-6)(Conv1D(filters=units, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate, activation='gelu')(x_tcn))
    x_emb = LayerNormalization(epsilon=1e-6)(Dense(units, activation='gelu')(tf.transpose(x_tcn, perm=[0, 2, 1])))
    x_attn = LayerNormalization(epsilon=1e-6)(x_emb + MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)(x_emb, x_emb))
    ffn = ChebyshevKANLinear(units, degree=4)(LayerNormalization(epsilon=1e-6)(Dropout(dropout)(ChebyshevKANLinear(units * 2, degree=4)(x_attn))))
    return Flatten()(LayerNormalization(epsilon=1e-6)(x_attn + ffn))

def build_agfa_model(model_type: str, seq_len: int, n_feats: int, wpd_indices: list, context_indices: list, gater_indices: list, units: int, dropout: float, lr: float, quantiles: list, horizon: int):
    inputs = Input(shape=(seq_len, n_feats))
    
    if wpd_indices:
        wpd_in = Lambda(lambda x: tf.gather(x, wpd_indices, axis=-1))(inputs)
        context_in = Lambda(lambda x: tf.gather(x, context_indices, axis=-1))(inputs)
        gater_in = Lambda(lambda x: tf.gather(x, gater_indices, axis=-1))(inputs)
        
        expert_preds = []
        for i in range(len(wpd_indices)):
            mode_slice = Lambda(lambda x, idx=i: tf.expand_dims(x[:, :, idx], -1))(wpd_in)
            expert_input = Concatenate(axis=-1)([mode_slice, context_in])
            
            if "TCN" in model_type: x_feat = _build_apex_hybrid(expert_input, units, dropout, 3, 4)
            elif "Transformer" in model_type: x_feat = _build_transformer_family(expert_input, model_type, units, dropout, 4, seq_len, 1 + len(context_indices))
            else: x_feat = _build_rnn_family(expert_input, model_type, units, dropout)
            
            dl_raw = Dense(horizon * len(quantiles), activation='linear')(x_feat)
            expert_reshaped = Reshape((horizon, len(quantiles)))(dl_raw)
            expert_preds.append(expert_reshaped)
            
        outputs = DynamicWeatherGater(num_experts=len(wpd_indices))(expert_preds, gater_in)
    else:
        if "TCN" in model_type: x_feat = _build_apex_hybrid(inputs, units, dropout, 3, 4)
        elif "Transformer" in model_type: x_feat = _build_transformer_family(inputs, model_type, units, dropout, 4, seq_len, n_feats)
        else: x_feat = _build_rnn_family(inputs, model_type, units, dropout)
        
        dl_out = Dense(horizon * len(quantiles), activation='linear')(x_feat)
        outputs = Reshape((horizon, len(quantiles)))(dl_out)

    model = Model(inputs, outputs)
    
    # --- ACADEMIC FIX: delta reduced from 0.1 to 0.01 ---
    def smoothed_pinball_loss(y_true, y_pred, delta=0.01):
        err = tf.expand_dims(y_true, -1) - y_pred
        abs_err = tf.abs(err)
        huber_err = tf.where(abs_err <= delta, 0.5 * tf.square(err) / delta, abs_err - 0.5 * delta)
        q_tensor = tf.constant(quantiles, dtype=tf.float32)
        return tf.reduce_mean(tf.where(err >= 0, q_tensor * huber_err, (1.0 - q_tensor) * huber_err))
        
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=0.5), loss=smoothed_pinball_loss)
    return model

def apply_transfer_learning(model: Model, weights_path: str, freeze_base: bool = True, lr: float = 1e-4) -> Model:
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    if freeze_base:
        for layer in model.layers:
            if isinstance(layer, (MultiHeadAttention, SeparableConv1D, Conv1D, PositionalEncoding, DynamicWeatherGater)): layer.trainable = False
            elif 'kan' not in layer.name.lower() and 'dense' not in layer.name.lower(): layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss=model.loss)
    return model

class MultiHorizonNGBoost:
    def __init__(self, horizon):
        self.horizon = horizon
        self.models = {}

    def fit(self, X, y_seq):
        for h in range(self.horizon):
            self.models[h] = {}
            for q in [0.1, 0.5, 0.9]:
                model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=50, max_depth=4, random_state=42)
                model.fit(X, y_seq[:, h])
                self.models[h][q] = model

    def predict_quantiles(self, X, q_labels):
        N = len(X)
        results = {q: np.zeros((N, self.horizon)) for q in q_labels}
        preds_10, preds_50, preds_90 = np.zeros((N, self.horizon)), np.zeros((N, self.horizon)), np.zeros((N, self.horizon))
        for h in range(self.horizon):
            preds_10[:, h] = self.models[h][0.1].predict(X)
            preds_50[:, h] = self.models[h][0.5].predict(X)
            preds_90[:, h] = self.models[h][0.9].predict(X)
        x_q = [0.1, 0.5, 0.9]
        for i in range(N):
            for h in range(self.horizon):
                y_q = np.sort([preds_10[i, h], preds_50[i, h], preds_90[i, h]])
                f = interp1d(x_q, y_q, kind='linear', fill_value="extrapolate")
                for q in q_labels: results[q][i, h] = f(q)
        return results

class MultiHorizonCopula:
    def __init__(self, horizon):
        self.horizon = horizon
        self.copula_corrs = []
        self.ecdf_base = []
        self.err_vals = []

    def fit(self, y_true_seq, y_base_seq, z_covariate=None):
        for h in range(self.horizon):
            yb, yt = y_base_seq[:, h], y_true_seq[:, h]
            err = yt - yb
            eb, ee = ECDF(yb), ECDF(err)
            self.ecdf_base.append(eb); self.err_vals.append(np.sort(err))
            z_b, z_e = norm.ppf(np.clip(eb(yb), 1e-6, 1 - 1e-6)), norm.ppf(np.clip(ee(err), 1e-6, 1 - 1e-6))
            self.copula_corrs.append(np.corrcoef(z_b, z_e)[0, 1] if len(z_b) > 1 and np.var(z_b) > 0 else 0.0)

    def predict_quantiles(self, y_base_seq, z_covariate, q_labels):
        N = len(y_base_seq)
        results = {q: np.zeros((N, self.horizon)) for q in q_labels}
        for h in range(self.horizon):
            yb, rho, eb = y_base_seq[:, h], self.copula_corrs[h], self.ecdf_base[h]
            z_b = norm.ppf(np.clip(eb(yb), 1e-6, 1 - 1e-6))
            cond_mean, cond_std = rho * z_b, np.sqrt(max(0.0, 1.0 - rho**2))
            for q in q_labels:
                u_q = norm.cdf(norm.ppf(q, loc=cond_mean, scale=cond_std))
                results[q][:, h] = yb + np.percentile(self.err_vals[h], u_q * 100)
        return results