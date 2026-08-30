# utils/reporter.py
import os
import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime

from core.datatypes import ExperimentResult

class Reporter:
    """Generates industrial-grade Excel reports with embedded statistical visualizations."""
    
    @staticmethod
    def _save_fig_for_excel(fig: Figure, prefix: str) -> str:
        path = f"exports/figures/{prefix}_{int(time.time()*1000)}.png"
        fig.savefig(path, facecolor='#ffffff', bbox_inches='tight', dpi=100)
        fig.clear()
        plt.close(fig) 
        return path

    @staticmethod
    def export_excel_sync(exps: List[ExperimentResult], path: str, per_model_mode: bool = True):
        temp_path = path + ".tmp.xlsx"
        raw_export_path = path.replace(".xlsx", "_RAW_PREDICTIONS.csv.gz")
        
        Reporter._export_raw_predictions_csv(exps, raw_export_path)
        
        try:
            with pd.ExcelWriter(temp_path, engine='xlsxwriter') as writer:
                workbook = writer.book
                header_format = workbook.add_format({
                    'bold': True, 'text_wrap': True, 'valign': 'top', 
                    'fg_color': '#1e293b', 'font_color': '#f8fafc', 'border': 1
                })
                
                Reporter._write_readme_sheets(writer, exps, header_format)
                Reporter._write_leaderboard(writer, exps, header_format)
                Reporter._write_hyperparameters(writer, exps, header_format)
                
                if per_model_mode:
                    for e in exps: 
                        Reporter._write_model_visuals(writer, e, exps)
                        
                for sn in writer.book.sheetnames:
                    ws = writer.sheets[sn]
                    ws.set_column(0, 20, 18)
                    ws.freeze_panes(1, 0)
                    
            os.replace(temp_path, path)
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            raise Exception(f"Excel Export Engine Failure: {str(e)}")

    @staticmethod
    def _export_raw_predictions_csv(exps: List[ExperimentResult], path: str):
        all_rows = []
        for e in exps:
            for f in e.folds:
                df_fold = pd.DataFrame({
                    'model_id': e.model_name, 
                    'fold_id': f.fold_id, 
                    'timestamp_issue_utc': f.issue_times, 
                    'timestamp_valid_utc': f.valid_times, 
                    'target_actual': f.y_true, 
                    'pred_q50': f.y_pred,
                    'pred_baseline_pers': f.baseline_pers
                })
                for q_k, q_vals in f.y_pred_quantiles.items(): 
                    df_fold[f'pred_q{int(q_k*100)}'] = q_vals
                all_rows.append(df_fold)
        if all_rows: 
            pd.concat(all_rows, ignore_index=True).to_csv(path, index=False, compression='gzip')

    @staticmethod
    def _write_readme_sheets(writer, exps, header_format):
        if not exps: return
        meta_df = pd.DataFrame({
            "Metadata_Key": ["Platform", "Methodology", "Run_Fingerprint", "Timestamp_UTC", "Data_Source", "Total_Observations"], 
            "Metadata_Value": ["Helios-Grid Probabilistic PV Forecasting", exps[0].methodology, exps[0].run_fingerprint, datetime.utcnow().isoformat(), exps[0].site_name, sum([len(f.valid_times) for e in exps for f in e.folds])]
        })
        meta_df.to_excel(writer, sheet_name='00_Experiment_Topology', index=False)
        
        qa_flat = []
        for k, v in exps[0].qa_report.get('issues', {}).items(): qa_flat.append({"Entity_Type": "Data_Issue", "Metric_Name": k, "Metric_Value": str(v)})
        for k, v in exps[0].qa_report.get('actions', {}).items(): qa_flat.append({"Entity_Type": "Pipeline_Action", "Metric_Name": k, "Metric_Value": str(v)})
        if qa_flat:
            pd.DataFrame(qa_flat).to_excel(writer, sheet_name='01_Data_Integrity_Audit', index=False)

    @staticmethod
    def _write_leaderboard(writer, exps, header_format):
        summ = []
        for e in exps:
            if not e.folds: continue
            base_row = {
                "Model_ID": e.model_name,
                "Site_ID": e.site_name,
                "Training_Time_Sec": round(e.training_time_total, 2),
                "Folds_Evaluated": len(e.folds)
            }
            metrics_keys = e.folds[0].metrics.keys()
            for k in metrics_keys:
                base_row[f"Mean_{k}"] = round(np.mean([f.metrics.get(k, 0) for f in e.folds]), 4)
            summ.append(base_row)
        
        df_lead = pd.DataFrame(summ)
        sort_col = next((c for c in df_lead.columns if 'Mean_RMSE_L' in c), None)
        if not sort_col: sort_col = next((c for c in df_lead.columns if 'Mean_RMSE' in c), None)
        if sort_col: df_lead = df_lead.sort_values(by=sort_col, ascending=True)
        
        df_lead.to_excel(writer, sheet_name='02_Empirical_Evaluation', index=False)

    @staticmethod
    def _write_hyperparameters(writer, exps, header_format):
        params_list = []
        for e in exps:
            if 'Persistence' in e.model_name or 'ARIMA' in e.model_name: continue
            if e.folds:
                p = {"Model_ID": e.model_name}
                p.update(e.folds[0].params)
                params_list.append(p)
        if params_list:
            pd.DataFrame(params_list).to_excel(writer, sheet_name='03_Hyperparameter_Manifold', index=False)

    @staticmethod
    def _write_model_visuals(writer, e, all_exps):
        safe_m = e.model_name[:25].replace('-','_').replace(':','')
        ws_name = f"Vis_{safe_m}"
        pd.DataFrame([{"Architecture": e.model_name, "Visual_Artifacts": "Rendered Below"}]).to_excel(writer, sheet_name=ws_name, index=False)
        
        H = e.horizon
        k_indices = sorted(list(set([0, max(0, H//2), H-1])))
        ws_sum = writer.sheets[ws_name]
        current_excel_row = 4
        
        for k in k_indices:
            t_vals_utc = pd.to_datetime(np.concatenate([f.valid_times for f in e.folds]), utc=True)[k::H]
            y_slice = np.concatenate([f.y_true for f in e.folds])[k::H]
            p_slice = np.concatenate([f.y_pred for f in e.folds])[k::H]
            
            df_all = pd.DataFrame({'t': t_vals_utc, 'y': y_slice, 'p': p_slice})
            if df_all.empty: continue
            
            df_all['date'] = df_all['t'].dt.date
            df_all['err'] = np.abs(df_all['y'] - df_all['p'])
            df_all['ramp'] = np.abs(np.diff(df_all['y'].values, prepend=df_all['y'].iloc[0]))
            
            day_agg = df_all.groupby('date')['err'].mean()
            med_day = day_agg.index[np.argsort(day_agg.values)[len(day_agg)//2]] if len(day_agg) > 0 else df_all['date'].iloc[0]
            max_day = day_agg.idxmax() if len(day_agg) > 0 else df_all['date'].iloc[0]
            ramp_day = df_all.groupby('date')['ramp'].max().idxmax() if len(day_agg) > 0 else df_all['date'].iloc[0]

            def render_rep_day(t_date, suffix, title):
                d_sub = df_all[df_all['date'] == t_date]
                if len(d_sub) == 0: return None, None
                t_plot = d_sub['t'].dt.tz_localize(None) 
                
                f1 = Figure(figsize=(6, 3)); FigureCanvasAgg(f1); ax1 = f1.add_subplot(111)
                ax1.plot(t_plot, d_sub['y'], label='Actual Obs', color='#1e293b', linewidth=1.5)
                ax1.plot(t_plot, d_sub['p'], label='Prediction', color='#0284c7', linewidth=1.5, linestyle='--')
                ax1.set_title(f"{title} (Step {k+1})", fontsize=10, fontweight='bold')
                ax1.legend(loc='upper right', fontsize=8); ax1.set_ylabel("Power")
                p1 = Reporter._save_fig_for_excel(f1, f"{safe_m}_k{k}_{suffix}_ts")
                
                f2 = Figure(figsize=(6, 1.5)); FigureCanvasAgg(f2); ax2 = f2.add_subplot(111)
                ax2.bar(t_plot, d_sub['p'] - d_sub['y'], color='#e11d48', width=0.01)
                ax2.set_title("Absolute Residuals", fontsize=8); ax2.axhline(0, color='black', linewidth=1)
                p2 = Reporter._save_fig_for_excel(f2, f"{safe_m}_k{k}_{suffix}_err")
                return p1, p2

            p1_m, p2_m = render_rep_day(med_day, "med", "Median Operational Day")
            p1_w, p2_w = render_rep_day(max_day, "max", "Worst-Case Error Day")
            p1_r, p2_r = render_rep_day(ramp_day, "rmp", "High-Volatility Ramp Event")

            try:
                ws_sum.write_string(f"A{current_excel_row-1}", f"=== FORECAST HORIZON STEP {k+1} ===")
                if p1_m: ws_sum.insert_image(f"A{current_excel_row}", p1_m); ws_sum.insert_image(f"A{current_excel_row+16}", p2_m)
                if p1_w: ws_sum.insert_image(f"I{current_excel_row}", p1_w); ws_sum.insert_image(f"I{current_excel_row+16}", p2_w)
                if p1_r: ws_sum.insert_image(f"Q{current_excel_row}", p1_r); ws_sum.insert_image(f"Q{current_excel_row+16}", p2_r)
            except Exception: pass
            
            current_excel_row += 28