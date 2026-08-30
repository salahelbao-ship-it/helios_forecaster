# ui/plots.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from core.datatypes import ExperimentResult

# --- ACADEMIC Q1 PUBLICATION THEME (PRINT-READY) ---
# Strips out UI dark-modes for high-contrast, PDF-safe paper plots
THEME = {
    'bg': '#ffffff',          # Pure white for paper background
    'panel': '#fcfcfc',       # Off-white for axes panel
    'grid': '#e5e7eb',        # Subtle gray for gridlines
    'text': '#000000',        # Strict black for primary text
    'text_muted': '#374151',  # Slate for secondary labels
    'primary': '#0284c7',     # Deep Academic Blue
    'secondary': '#b91c1c',   # Deep Crimson
    'accent': '#047857',      # Forest Green
    'danger': '#b91c1c',      # Crimson
    'success': '#047857'      # Forest Green
}

class BasePlotter(QWidget):
    """Base class for all academic plotters implementing safe data extraction and Q1 theming."""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Enforce Academic Typography
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        
        self.fig = Figure(facecolor=THEME['bg'], dpi=120) 
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

    def clear(self):
        self.fig.clear()
        
    def _apply_theme(self, ax, title="", xlabel="", ylabel="", is_3d=False):
        """Applies strict academic formatting to axes."""
        if not is_3d:
            ax.set_facecolor(THEME['panel'])
            ax.grid(True, linestyle=':', alpha=0.7, color=THEME['grid'])
            for spine in ax.spines.values():
                spine.set_color('#9ca3af')
                spine.set_linewidth(0.8)
                
        ax.tick_params(colors=THEME['text_muted'], direction='in', length=4)
        ax.set_title(title, color=THEME['text'], fontweight='bold', pad=12)
        ax.set_xlabel(xlabel, color=THEME['text'], fontweight='bold', labelpad=8)
        ax.set_ylabel(ylabel, color=THEME['text'], fontweight='bold', labelpad=8)
        
        if is_3d:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor(THEME['grid'])
            ax.yaxis.pane.set_edgecolor(THEME['grid'])
            ax.zaxis.pane.set_edgecolor(THEME['grid'])

    def _get_q_np(self, f, q) -> np.ndarray:
        if not hasattr(f, '_q_np_cache'): f._q_np_cache = {}
        str_q = str(q)
        if str_q not in f._q_np_cache:
            raw_val = f.y_pred_quantiles.get(str_q)
            if raw_val is None: raw_val = f.y_pred_quantiles.get(float(q))
            if raw_val is None:
                legacy_y_pred = getattr(f, 'y_pred', None) 
                if legacy_y_pred is not None: raw_val = legacy_y_pred
                else: raw_val = f.y_pred_quantiles.get('0.5', np.zeros_like(f.y_true))
            f._q_np_cache[str_q] = np.array(raw_val, dtype=np.float32)
        return f._q_np_cache[str_q]

    def _get_horizon_index(self, exp: ExperimentResult, target_lead_min: int) -> Tuple[int, int]:
        H = exp.horizon
        try:
            dt_minutes = 15 
            k = max(0, min(int(round(target_lead_min / dt_minutes)) - 1, H - 1))
        except Exception: k = 0
        return k, H

    def draw_canvas(self):
        self.fig.tight_layout()
        self.canvas.draw()

class InteractivePlotter(BasePlotter):
    def plot_ts(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111)
        colors = [THEME['primary'], THEME['secondary'], THEME['accent'], '#d97706']
        e_base = exps[0]
        k, H = self._get_horizon_index(e_base, target_lead_min)
        
        try:
            # ACADEMIC FIX: Map sequence steps to true validation timestamps
            y_true = np.concatenate([np.array(f.y_true)[k::H] for f in e_base.folds])
            valid_times_str = np.concatenate([np.array(f.valid_times)[k::H] for f in e_base.folds])
            dates = pd.to_datetime(valid_times_str)
            
            limit = min(400, len(y_true))
            y_true_slice = y_true[-limit:]
            dates_slice = dates[-limit:]
            
            ax.plot(dates_slice, y_true_slice, color='black', label='Observed Reference', linewidth=1.5, zorder=3)
            
            for i, exp in enumerate(exps):
                c = colors[i % len(colors)]
                y_pred = np.concatenate([self._get_q_np(f, '0.5')[k::H] for f in exp.folds])[-limit:]
                ax.plot(dates_slice, y_pred, color=c, label=f'{exp.model_name}', linewidth=1.5, alpha=0.85, zorder=2)
                
                y_10 = np.concatenate([self._get_q_np(f, '0.1')[k::H] for f in exp.folds])[-limit:]
                y_90 = np.concatenate([self._get_q_np(f, '0.9')[k::H] for f in exp.folds])[-limit:]
                if not np.array_equal(y_10, y_pred) and not np.array_equal(y_90, y_pred):
                    ax.fill_between(dates_slice, y_10, y_90, color=c, alpha=0.15, label=f'{exp.model_name} (80% CI)', zorder=1)

            # Format X-axis to chronological time
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            self.fig.autofmt_xdate(rotation=0, ha='center')
            
            self._apply_theme(ax, f"Temporal Diurnal Generation Trajectory (+{target_lead_min}m Horizon)", "Chronological Time", "Power Output (W)")
            ax.legend(loc='upper right', facecolor=THEME['bg'], edgecolor=THEME['grid'], fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Data Alignment Error: {e}", color=THEME['danger'], ha='center')
        self.draw_canvas()

    def plot_ts_horizons(self, exps: List[ExperimentResult], target_model: str):
        self.clear()
        exp = next((e for e in exps if e.model_name == target_model), None)
        if not exp: return
        ax = self.fig.add_subplot(111)
        H = exp.horizon
        try:
            f = exp.folds[-1] 
            limit = min(200, len(f.y_true) // H)
            
            # ACADEMIC FIX: Align predictions to their TARGET Valid Time
            for k in range(min(H, 4)): 
                y_pred_k = self._get_q_np(f, '0.5')[k::H][:limit]
                dates_k = pd.to_datetime(np.array(f.valid_times)[k::H][:limit])
                ax.plot(dates_k, y_pred_k, label=f'Lead +{k+1} Step', linewidth=1.5, alpha=1.0 - (k*0.15))
                
            y_true_0 = np.array(f.y_true)[0::H][:limit]
            dates_0 = pd.to_datetime(np.array(f.valid_times)[0::H][:limit])
            ax.plot(dates_0, y_true_0, color='black', label='Actual Observation', linewidth=1.5, linestyle='--')
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            self.fig.autofmt_xdate(rotation=0, ha='center')
            
            self._apply_theme(ax, f"Forecast Horizon Decay Analysis ({target_model})", "Target Validation Time", "Power Output (W)")
            ax.legend(loc='upper right', facecolor=THEME['bg'], edgecolor=THEME['grid'], fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Insufficient Horizon Data: {e}", color=THEME['danger'], ha='center')
        self.draw_canvas()

class ScientificPlotter(BasePlotter):
    def plot_correlation_matrix(self, df: pd.DataFrame, target_col: str):
        self.clear()
        ax = self.fig.add_subplot(111)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty: return
        corr = num_df.corr()
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        self.fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns, rotation=45, ha='left', color=THEME['text'], fontsize=8)
        ax.set_yticklabels(corr.columns, color=THEME['text'], fontsize=8)
        self._apply_theme(ax, "Pearson Covariance Matrix (Multicollinearity)", "", "")
        self.draw_canvas()

    def plot_scatter_parity(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        n_exps = len(exps)
        cols = min(3, n_exps)
        rows = int(np.ceil(n_exps / cols))
        
        for i, exp in enumerate(exps):
            ax = self.fig.add_subplot(rows, cols, i+1)
            k, H = self._get_horizon_index(exp, target_lead_min)
            try:
                y_true = np.concatenate([np.array(f.y_true)[k::H] for f in exp.folds])
                y_pred = np.concatenate([self._get_q_np(f, '0.5')[k::H] for f in exp.folds])
                if len(y_true) > 3000: 
                    idx = np.random.choice(len(y_true), 3000, replace=False)
                    y_true, y_pred = y_true[idx], y_pred[idx]
                    
                ax.scatter(y_true, y_pred, alpha=0.4, s=6, color=THEME['primary'], edgecolors='none')
                lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
                ax.plot(lims, lims, 'k--', alpha=0.8, zorder=3, color=THEME['danger'])
                
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true, y_pred) if len(y_true) > 0 else 0
                ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=ax.transAxes, color=THEME['text'], fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                self._apply_theme(ax, f"{exp.model_name}", "Observed Power (W)", "Predicted Power (W)")
            except Exception: pass
            
        self.fig.suptitle(f"Regression Parity Scatter Plot (+{target_lead_min}m)", color=THEME['text'], fontweight='bold', fontsize=14)
        self.draw_canvas()

    def plot_error_box(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111)
        residuals, labels = [], []
        
        for exp in exps:
            k, H = self._get_horizon_index(exp, target_lead_min)
            try:
                y_t = np.concatenate([np.array(f.y_true)[k::H] for f in exp.folds])
                y_p = np.concatenate([self._get_q_np(f, '0.5')[k::H] for f in exp.folds])
                residuals.append(y_p - y_t)
                labels.append(exp.model_name)
            except Exception: pass
                
        if residuals:
            box = ax.boxplot(residuals, patch_artist=True, labels=labels, vert=False, notch=True, flierprops=dict(marker='.', markersize=2, alpha=0.1))
            for patch in box['boxes']:
                patch.set_facecolor(THEME['grid']); patch.set_edgecolor(THEME['text'])
            for median in box['medians']: 
                median.set_color(THEME['danger']); median.set_linewidth(2)
            ax.axvline(0, color='black', linestyle='--', alpha=0.7)
            self._apply_theme(ax, f"Distribution of Residual Bias (+{target_lead_min}m)", "Error Magnitude: (Predicted - Actual)", "Forecasting Architecture")
        self.draw_canvas()

    def plot_radar_metrics(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111, polar=True)
        metrics_keys = ['RMSE', 'MAE', 'MBE', 'sMAPE', 'Skill_Ramp_%']
        angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False).tolist()
        angles += angles[:1]
        colors = [THEME['primary'], THEME['secondary'], THEME['accent'], '#d97706']
        
        for i, exp in enumerate(exps):
            vals = []
            for mk in metrics_keys:
                fold_vals = [f.metrics.get(f"{mk}_L{target_lead_min}", 0) for f in exp.folds]
                vals.append(abs(np.mean(fold_vals) if fold_vals else 0))
            if max(vals) > 0: vals = [v / max(vals) for v in vals]
            vals += vals[:1]
            c = colors[i % len(colors)]
            ax.plot(angles, vals, color=c, linewidth=2, label=exp.model_name)
            ax.fill(angles, vals, color=c, alpha=0.1)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ') for m in metrics_keys], color=THEME['text'], fontsize=10, fontweight='bold')
        ax.set_yticks([])
        ax.set_facecolor(THEME['panel'])
        ax.spines['polar'].set_color('#d1d5db')
        ax.grid(color='#d1d5db', linestyle='--')
        self._apply_theme(ax, f"Multivariate Metric Signature Radar (+{target_lead_min}m)", "", "")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=THEME['bg'], edgecolor=THEME['grid'], fontsize=8)
        self.draw_canvas()

    def plot_taylor(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111, polar=True)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        colors = [THEME['primary'], THEME['secondary'], THEME['accent'], '#d97706']
        has_ref = False
        
        for i, exp in enumerate(exps):
            k, H = self._get_horizon_index(exp, target_lead_min)
            try:
                y_t = np.concatenate([np.array(f.y_true)[k::H] for f in exp.folds])
                y_p = np.concatenate([self._get_q_np(f, '0.5')[k::H] for f in exp.folds])
                if len(y_t) > 0 and len(y_p) > 0:
                    if not has_ref:
                        ref_std = np.std(y_t)
                        ax.plot(0, ref_std, marker='*', color='black', markersize=14, linestyle='None', label='Observed Reference')
                        has_ref = True
                    std_p = np.std(y_p)
                    corr = np.corrcoef(y_t, y_p)[0, 1] if np.std(y_p) > 0 else 0
                    theta = np.arccos(np.clip(corr, 0, 1))
                    c = colors[i % len(colors)]
                    ax.plot(theta, std_p, marker='o', color=c, markersize=8, linestyle='None', label=exp.model_name)
            except Exception: pass
                
        ax.set_facecolor(THEME['panel'])
        ax.grid(color=THEME['grid'], linestyle='--', alpha=0.7)
        ax.spines['polar'].set_color('#9ca3af')
        corrs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
        angles = np.arccos(corrs)
        ax.set_xticks(angles)
        ax.set_xticklabels([str(c) for c in corrs], color=THEME['text'], fontsize=9)
        
        self._apply_theme(ax, f"Taylor Diagram Analysis (+{target_lead_min}m)\n(Radial = Standard Deviation, Angular = Pearson Correlation)", "", "")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=THEME['bg'], edgecolor=THEME['grid'], fontsize=8)
        self.draw_canvas()

    def plot_bars(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111)
        models, rmses = [], []
        
        for exp in exps:
            fold_rmses = [f.metrics.get(f"RMSE_L{target_lead_min}") for f in exp.folds if f.metrics.get(f"RMSE_L{target_lead_min}") is not None]
            if fold_rmses:
                models.append(exp.model_name)
                rmses.append(np.mean(fold_rmses))
                
        if models:
            bars = ax.bar(models, rmses, color=THEME['primary'], edgecolor='black', linewidth=1)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + (max(rmses)*0.02), f'{yval:.1f}', ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)
            self._apply_theme(ax, f"Benchmark Analysis: Root Mean Square Error (+{target_lead_min}m)", "", "RMSE Magnitude (W)")
            ax.set_xticklabels(models, rotation=15, ha='center')
        self.draw_canvas()

    def plot_wpd_diag(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111)
        colors = [THEME['primary'], THEME['secondary'], THEME['accent']]
        
        for i, exp in enumerate(exps):
            k, H = self._get_horizon_index(exp, target_lead_min)
            try:
                y_t = np.concatenate([np.array(f.y_true)[k::H] for f in exp.folds])
                y_p = np.concatenate([self._get_q_np(f, '0.5')[k::H] for f in exp.folds])
                
                valid_times_str = np.concatenate([np.array(f.valid_times)[k::H] for f in exp.folds])
                dates = pd.to_datetime(valid_times_str)
                
                limit = min(250, len(y_t))
                err = (y_p - y_t)[-limit:]
                dates_slice = dates[-limit:]
                
                ax.plot(dates_slice, err, label=exp.model_name, color=colors[i % len(colors)], linewidth=1.2, alpha=0.8)
            except Exception: pass
                
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        self.fig.autofmt_xdate(rotation=0, ha='center')
            
        self._apply_theme(ax, f"High-Frequency Transient Residuals (+{target_lead_min}m)", "Chronological Time", "Delta Error (W)")
        ax.legend(loc='upper right', facecolor=THEME['bg'], edgecolor=THEME['grid'], fontsize=8)
        self.draw_canvas()

    def plot_reliability(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        ax = self.fig.add_subplot(111)
        q_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        colors = [THEME['primary'], THEME['secondary'], THEME['accent']]
        ax.plot([0, 1], [0, 1], 'k--', color='black', label='Perfect Calibration', zorder=1)
        
        for i, exp in enumerate(exps):
            k, H = self._get_horizon_index(exp, target_lead_min)
            obs_freqs = []
            try:
                y_t = np.concatenate([np.array(f.y_true)[k::H] for f in exp.folds])
                if len(y_t) > 0:
                    for q in q_targets:
                        q_preds = np.concatenate([self._get_q_np(f, str(q))[k::H] for f in exp.folds])
                        hits = np.sum(y_t <= q_preds)
                        obs_freqs.append(hits / len(y_t))
                    ax.plot(q_targets, obs_freqs, marker='o', color=colors[i % len(colors)], label=exp.model_name, linewidth=1.5, zorder=2)
            except Exception: pass
                
        self._apply_theme(ax, f"Probabilistic Reliability & Calibration (+{target_lead_min}m)", "Target Quantile Interval", "Empirical Observation Frequency")
        ax.legend(loc='upper left', facecolor=THEME['bg'], edgecolor=THEME['grid'], fontsize=8)
        self.draw_canvas()

    def plot_xai_importance(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        exp = next((e for e in exps if any(f.feature_selection_report.get('xai_importance') for f in e.folds)), None)
        if not exp:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Permutation Importance (XAI) data available.", color=THEME['danger'], ha='center', va='center')
            self._apply_theme(ax, "Explainable AI (Feature Importance)", "", "")
            self.draw_canvas()
            return
            
        f = next(f for f in exp.folds if 'xai_importance' in f.feature_selection_report)
        importance_dict = f.feature_selection_report['xai_importance']
        if not importance_dict: return
        
        sorted_items = sorted(importance_dict.items(), key=lambda item: item[1], reverse=False)
        features = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        ax = self.fig.add_subplot(111)
        bars = ax.barh(features, scores, color=THEME['primary'], edgecolor='black', linewidth=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max(scores)*0.01), bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center', color='black', fontsize=9, fontweight='bold')
            
        self._apply_theme(ax, f"Permutation Feature Importance Thresholds ({exp.model_name})", "Relative Impact on Loss (%)", "")
        self.draw_canvas()

    def plot_3d_uncertainty_manifold(self, exps: List[ExperimentResult], target_lead_min: int):
        self.clear()
        if not exps: return
        exp = exps[0]
        k, H = self._get_horizon_index(exp, target_lead_min)
        
        try:
            f = exp.folds[-1]
            limit = min(150, len(f.y_true) // H)
            if limit < 2: return
            
            q_targets = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            # ACADEMIC FIX: Convert Dates to numeric format for accurate 3D Meshgrid plotting
            dates_str = np.array(f.valid_times)[k::H][:limit]
            dates = pd.to_datetime(dates_str)
            x_vals = mdates.date2num(dates)
            
            Z = []
            for q in q_targets:
                Z.append(self._get_q_np(f, q)[k::H][:limit])
            Z = np.array(Z)
            
            X, Y = np.meshgrid(x_vals, np.array(q_targets))
            
            ax = self.fig.add_subplot(111, projection='3d')
            # 'viridis' is the standard colorblind-friendly scientific map
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', linewidth=0.2, alpha=0.95)
            
            self._apply_theme(ax, f"3D Probabilistic Uncertainty Manifold ({exp.model_name})", "Time (t)", "Confidence Interval (Q)", is_3d=True)
            ax.set_zlabel("Predicted Power (W)", color=THEME['text'], fontsize=10, fontweight='bold', labelpad=10)
            
            # Format 3D X-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            for tick in ax.get_xticklabels(): tick.set_rotation(30)
            
            # Professional Viewing Angle for Academic Journals
            ax.view_init(elev=25, azim=-55)
            
            ax.xaxis.pane.set_facecolor(THEME['panel'])
            ax.yaxis.pane.set_facecolor(THEME['panel'])
            ax.zaxis.pane.set_facecolor(THEME['panel'])
            
            cbar = self.fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, pad=0.08)
            cbar.set_label('Power Magnitude', rotation=270, labelpad=15, fontweight='bold')
            
        except Exception as e:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"3D Render Failed: {e}", color=THEME['danger'], ha='center')
            self._apply_theme(ax, "3D Uncertainty Manifold", "", "")
            
        self.draw_canvas()