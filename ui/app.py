# ui/app.py
import os
import traceback
from typing import List
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QCheckBox, 
    QTabWidget, QFileDialog, QMessageBox, QGroupBox, QHeaderView, 
    QTextEdit, QFormLayout, QProgressBar, QListWidget, 
    QAbstractItemView, QLabel, QScrollArea, QApplication, QLineEdit, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QThreadPool
from PySide6.QtGui import QColor, QFont

from core.datatypes import ExperimentConfig, SiteContext, ExperimentResult
from core.engine import WorkerThread
from data.database import ExperimentRegistry
from data.loader import DataLoader
from ui.dialogs import AddSiteDialog
from ui.plots import InteractivePlotter, ScientificPlotter
from utils.reporter import Reporter

from workers.io_workers import CorrelationWorker, DbLoadWorker


class ExportWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, exps: List[ExperimentResult], path: str): 
        super().__init__()
        self.exps = exps
        self.path = path
        
    def run(self):
        try: 
            Reporter.export_excel_sync(self.exps, self.path, per_model_mode=True)
            self.finished.emit(self.path)
        except Exception as e: 
            self.error.emit(str(e))


class DashboardApp(QMainWindow):
    
    # Constants for Tab Routing
    TAB_METRICS_INDEX = 12
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helios-Grid: Advanced PV Forecasting Suite")
        self.showMaximized()
        self._load_stylesheet()
        
        self.db = ExperimentRegistry()
        self.sites = self.db.load_sites()
        self.exps: List[ExperimentResult] = []
        self.model_checkboxes = []
        
        self._init_ui()
        self._refresh_db_list()
        
        for site in self.sites: 
            self.list_sites.addItem(f"{site.name} | Target: {site.target_col}")
        
        if self.combo_db.count() > 0: 
            self.combo_db.setCurrentIndex(0)
            self.load_db()

    def closeEvent(self, event):
        self.lbl_status.setText("SYSTEM SHUTDOWN...")
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        event.accept()

    def _load_stylesheet(self):
        """Compact Professional Dark Academic Theme with Gradient Progress"""
        self.setStyleSheet("""
            QMainWindow { background-color: #09090b; color: #fafafa; font-family: 'Segoe UI', system-ui, sans-serif; }
            
            QGroupBox {
                background-color: #18181b; border: 1px solid #27272a; border-radius: 4px;
                margin-top: 14px; padding-top: 10px; font-weight: 600; font-size: 12px; color: #e4e4e7;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: -2px;
                padding: 2px 8px; background-color: #06b6d4; color: #09090b; border-radius: 3px; font-weight: bold; font-size: 11px;
            }
            QPushButton {
                background-color: #27272a; color: #fafafa; border: 1px solid #3f3f46;
                border-radius: 4px; padding: 6px 12px; font-weight: 600; font-size: 11px;
            }
            QPushButton:hover { background-color: #3f3f46; border: 1px solid #06b6d4; color: #22d3ee; }
            QPushButton:pressed { background-color: #18181b; border: 1px solid #0891b2; }
            
            QPushButton#RunBtn { background-color: #0284c7; border: 1px solid #0369a1; font-size: 13px; padding: 10px; font-weight: bold; }
            QPushButton#RunBtn:hover { background-color: #0ea5e9; }
            QPushButton#TermBtn { background-color: #be123c; border: 1px solid #9f1239; font-size: 13px; padding: 10px; font-weight: bold;}
            QPushButton#TermBtn:hover { background-color: #e11d48; }
            
            QComboBox, QLineEdit, QSpinBox {
                background-color: #09090b; color: #e4e4e7; border: 1px solid #3f3f46; 
                border-radius: 4px; padding: 4px 8px; font-size: 11px; min-height: 20px;
            }
            QComboBox:hover, QLineEdit:focus, QSpinBox:hover { border: 1px solid #06b6d4; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView { background-color: #18181b; color: #fafafa; border: 1px solid #27272a; selection-background-color: #06b6d4; selection-color: #09090b; }
            
            QListWidget, QTextEdit { background-color: #09090b; border: 1px solid #27272a; border-radius: 4px; padding: 6px; color: #d4d4d8; font-size: 11px; }
            QListWidget::item { padding: 4px; border-bottom: 1px solid #18181b; }
            QListWidget::item:selected { background-color: #0891b2; color: #ffffff; border-radius: 3px; }
            
            QTableWidget { background-color: #18181b; alternate-background-color: #09090b; color: #e4e4e7; gridline-color: #27272a; border: 1px solid #3f3f46; border-radius: 4px; font-size: 11px; }
            QHeaderView::section { background-color: #09090b; color: #22d3ee; padding: 6px; border: 1px solid #27272a; border-bottom: 2px solid #06b6d4; font-weight: bold; font-size: 11px; }
            QTableWidget::item:selected { background-color: #0891b2; color: white; }
            
            QTabWidget::pane { border: 1px solid #3f3f46; border-radius: 4px; background: #18181b; top: -1px; }
            QTabBar::tab { background: #09090b; border: 1px solid #27272a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; padding: 8px 16px; color: #a1a1aa; font-weight: 600; margin-right: 2px; font-size: 12px; }
            QTabBar::tab:selected { background: #18181b; color: #22d3ee; border-top: 3px solid #06b6d4; border-left: 1px solid #3f3f46; border-right: 1px solid #3f3f46; }
            QTabBar::tab:hover:!selected { background: #27272a; color: #fafafa; }
            
            QCheckBox { color: #e4e4e7; font-weight: 500; font-size: 11px; spacing: 6px; }
            QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #3f3f46; border-radius: 3px; background: #09090b; }
            QCheckBox::indicator:checked { background: #06b6d4; border-color: #0891b2; }
            
            /* ACADEMIC UPGRADE: Sleek Gradient Progress Bar */
            QProgressBar { 
                border: 1px solid #27272a; border-radius: 5px; background-color: #09090b; 
                text-align: center; color: #ffffff; font-weight: bold; height: 18px; font-size: 11px; 
            }
            QProgressBar::chunk { 
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #0891b2, stop:1 #22d3ee); 
                border-radius: 4px; width: 10px; 
            }
        """)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.thread_pool = QThreadPool.globalInstance()
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        
        sidebar_widget = self._build_sidebar()
        main_panel = self._build_main_panel()
        
        self.splitter.addWidget(sidebar_widget)
        self.splitter.addWidget(main_panel)
        self.splitter.setSizes([380, 1500]) 
        
        main_layout.addWidget(self.splitter)

    def _build_sidebar(self) -> QWidget:
        sidebar_widget = QWidget()
        sidebar_widget.setMinimumWidth(360)
        sidebar_widget.setMaximumWidth(420)
        
        layout = QVBoxLayout(sidebar_widget)
        layout.setContentsMargins(0, 0, 5, 0) 
        layout.setSpacing(6)
        
        layout.addWidget(self._build_site_context_group())
        layout.addWidget(self._build_transfer_learning_group())
        layout.addWidget(self._build_hyperparameter_group())
        layout.addWidget(self._build_model_topology_group())
        layout.addLayout(self._build_execution_controls())
        layout.addWidget(self._build_status_monitor())
        
        return sidebar_widget

    def _build_site_context_group(self) -> QGroupBox:
        grp = QGroupBox("Spatial & Telemetry Configuration") # Updated Title
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(8, 16, 8, 8)
        layout.setSpacing(6)
        
        self.list_sites = QListWidget()
        self.list_sites.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_sites.setMinimumHeight(100) 
        
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Site")
        btn_add.clicked.connect(self.add_site)
        btn_delete = QPushButton("Delete Site")
        btn_delete.clicked.connect(self.delete_site)
        
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_delete)
        
        btn_corr = QPushButton("Plot Correlation Matrix")
        btn_corr.clicked.connect(self.plot_site_correlation)
        
        layout.addWidget(self.list_sites)
        layout.addLayout(btn_layout)
        layout.addWidget(btn_corr)
        return grp

    def _build_transfer_learning_group(self) -> QGroupBox:
        grp = QGroupBox("Cross-Domain Transfer Learning") # Updated Title
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(8, 16, 8, 8)
        
        h_layout = QHBoxLayout()
        self.e_transfer_weights = QLineEdit()
        self.e_transfer_weights.setPlaceholderText("Path to .keras or .h5 weights...")
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_weights_file)
        
        h_layout.addWidget(self.e_transfer_weights)
        h_layout.addWidget(btn_browse)
        layout.addLayout(h_layout)
        
        h_layout2 = QHBoxLayout()
        self.spin_ft_epochs = QSpinBox()
        self.spin_ft_epochs.setRange(5, 200)
        self.spin_ft_epochs.setValue(30)
        h_layout2.addWidget(QLabel("Fine-Tune Epochs:", styleSheet="color: #a1a1aa; font-weight: bold;"))
        h_layout2.addWidget(self.spin_ft_epochs)
        layout.addLayout(h_layout2)
        
        return grp

    def browse_weights_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SOTA Neural Weights", "cache", "Keras Models (*.keras);;HDF5 Weights (*.h5 *.weights.h5);;All Files (*)"
        )
        if file_path: 
            self.e_transfer_weights.setText(file_path)

    def _build_hyperparameter_group(self) -> QGroupBox:
        grp = QGroupBox("Hyperparameter & NAS Topology") # Updated Title
        layout = QFormLayout(grp)
        layout.setContentsMargins(8, 16, 8, 8)
        layout.setSpacing(6)
        
        self.e_horizons = QLineEdit("15, 30, 60")
        
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1, 200)
        self.spin_trials.setValue(25)
        
        self.combo_feat_sel = QComboBox()
        self.combo_feat_sel.addItems(["Auto (Mutual Info)", "Manual"])
        
        self.spin_folds = QSpinBox()
        self.spin_folds.setValue(5)
        
        self.chk_ramp = QCheckBox("Ramp Dynamics")
        self.chk_ramp.setChecked(True)
        
        self.chk_adv = QCheckBox("Harmonics")
        self.chk_adv.setChecked(True)
        
        self.chk_cache = QCheckBox("Enforce Caching")
        self.chk_cache.setChecked(True)
        
        self.chk_strict = QCheckBox("Strict QA")
        self.chk_strict.setChecked(False)
        
        layout.addRow("Horizons (m):", self.e_horizons)
        layout.addRow("NAS Trials:", self.spin_trials)
        layout.addRow("Protocol:", self.combo_feat_sel)
        layout.addRow("K-Folds:", self.spin_folds)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.chk_ramp)
        h_layout.addWidget(self.chk_adv)
        layout.addRow(h_layout)
        
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.chk_cache)
        h_layout2.addWidget(self.chk_strict)
        layout.addRow(h_layout2)
        
        return grp

    def _build_model_topology_group(self) -> QGroupBox:
        grp = QGroupBox("Architectural Ensembles") # Updated Title
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(8, 16, 8, 8)
        
        model_list = [
            "Persistence", "ARIMA", "LSTM", "GRU", "BiLSTM", "PatchTST", 
            "Transformer", "iTransformer", "TCN-iTransformer", "Dynamic Ensemble", 
            "NGBoostPSPF", "CopulaBayes"
        ]
        
        self.table_models = QTableWidget(len(model_list), 4)
        self.table_models.setHorizontalHeaderLabels(["Model", "ON", "WPD", "NAS"])
        self.table_models.setAlternatingRowColors(True)
        self.table_models.verticalHeader().setVisible(False)
        self.table_models.verticalHeader().setDefaultSectionSize(22) 
        self.table_models.setMinimumHeight(280)
        
        self.table_models.horizontalHeader().setStretchLastSection(True)
        self.table_models.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4): 
            self.table_models.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        for i, model_name in enumerate(model_list):
            self.table_models.setItem(i, 0, QTableWidgetItem(f" {model_name}"))
            
            widget_run = QWidget()
            layout_run = QHBoxLayout(widget_run)
            layout_run.setContentsMargins(0, 0, 0, 0)
            checkbox_run = QCheckBox()
            checkbox_run.setChecked(i in [0, 2]) 
            layout_run.addWidget(checkbox_run)
            layout_run.setAlignment(Qt.AlignCenter)
            widget_run.chk = checkbox_run
            self.table_models.setCellWidget(i, 1, widget_run)
            
            deep_learning_models = ["LSTM", "GRU", "BiLSTM", "PatchTST", "Transformer", "iTransformer", "TCN-iTransformer"]
            
            if model_name in deep_learning_models:
                widget_wpd = QWidget()
                layout_wpd = QHBoxLayout(widget_wpd)
                layout_wpd.setContentsMargins(0, 0, 0, 0)
                checkbox_wpd = QCheckBox()
                checkbox_wpd.setChecked(True)
                layout_wpd.addWidget(checkbox_wpd)
                layout_wpd.setAlignment(Qt.AlignCenter)
                widget_wpd.chk = checkbox_wpd
                self.table_models.setCellWidget(i, 2, widget_wpd)
                
                widget_opt = QWidget()
                layout_opt = QHBoxLayout(widget_opt)
                layout_opt.setContentsMargins(0, 0, 0, 0)
                checkbox_opt = QCheckBox()
                checkbox_opt.setChecked(True)
                layout_opt.addWidget(checkbox_opt)
                layout_opt.setAlignment(Qt.AlignCenter)
                widget_opt.chk = checkbox_opt
                self.table_models.setCellWidget(i, 3, widget_opt)
            else:
                for col_idx in (2, 3): 
                    empty_item = QTableWidgetItem("-")
                    empty_item.setFlags(Qt.ItemIsSelectable)
                    empty_item.setTextAlignment(Qt.AlignCenter)
                    self.table_models.setItem(i, col_idx, empty_item)
                    
        layout.addWidget(self.table_models)
        return grp

    def _build_execution_controls(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(6)
        
        btn_hl = QHBoxLayout()
        self.btn_run = QPushButton("Start Training")
        self.btn_run.setObjectName("RunBtn")
        self.btn_run.clicked.connect(self.run_experiment)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("TermBtn")
        self.btn_cancel.clicked.connect(self.cancel_experiment)
        self.btn_cancel.setEnabled(False)
        
        btn_hl.addWidget(self.btn_run)
        btn_hl.addWidget(self.btn_cancel)
        layout.addLayout(btn_hl)
        
        util_hl = QHBoxLayout()
        self.combo_db = QComboBox()
        
        btn_load = QPushButton("Load DB Run")
        btn_load.clicked.connect(self.load_db)
        
        btn_clear_c = QPushButton("Clear Cache")
        btn_clear_c.clicked.connect(self._flush_cache)
        
        util_hl.addWidget(self.combo_db)
        util_hl.addWidget(btn_load)
        util_hl.addWidget(btn_clear_c)
        layout.addLayout(util_hl)
        return layout

    def _build_status_monitor(self) -> QGroupBox:
        grp = QGroupBox("System Telemetry & Logs") # Updated Title
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(8, 16, 8, 8)
        layout.setSpacing(6)
        
        self.lbl_status = QLabel("SYSTEM IDLE")
        self.lbl_status.setStyleSheet("color: #06b6d4; font-weight: bold; font-family: 'Consolas', monospace; font-size: 11px; background: #09090b; padding: 4px; border-radius: 4px; border: 1px solid #27272a;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        
        self.prog = QProgressBar()
        self.prog.setAlignment(Qt.AlignCenter)
        self.prog.setFixedHeight(18)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 9))
        self.log_area.setMinimumHeight(60) 
        self.log_area.setStyleSheet("background: #000000; color: #22c55e; border: 1px solid #27272a;")
        
        btn_ex_xl = QPushButton("Export Academic Report (Excel)")
        btn_ex_xl.clicked.connect(self.export_xl)
        
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.prog)
        layout.addWidget(self.log_area)
        layout.addWidget(btn_ex_xl)
        return grp

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addLayout(self._build_chart_controls())
        
        self.tabs = QTabWidget()
        
        self.p_corr = ScientificPlotter()
        self.p_ts = InteractivePlotter()
        self.p_ts_horizons = InteractivePlotter()
        self.p_parity = ScientificPlotter()
        self.p_box = ScientificPlotter()
        self.p_radar = ScientificPlotter()
        self.p_tay = ScientificPlotter()
        self.p_bar = ScientificPlotter()
        self.p_wpd = ScientificPlotter()
        self.p_rel = ScientificPlotter()
        self.p_xai = ScientificPlotter()
        self.p_3d = ScientificPlotter()
        
        # Professional Tabs matching the scientific workflow
        tab_setup = [
            (self.p_corr, "Covariance"), 
            (self.p_ts, "Temporal Trajectory"),
            (self.p_ts_horizons, "Horizon Decay"), 
            (self.p_parity, "Parity Scatter"),
            (self.p_box, "Bias Distribution"), 
            (self.p_radar, "Metric Radar"),
            (self.p_tay, "Taylor Diagram"), 
            (self.p_bar, "Benchmarks"),
            (self.p_wpd, "Probabilistic"), 
            (self.p_rel, "Calibration"),
            (self.p_xai, "XAI Importance"), 
            (self.p_3d, "3D Manifold") 
        ]
        
        for plotter_widget, tab_name in tab_setup: 
            self.tabs.addTab(plotter_widget, tab_name)
            
        self.tabs.addTab(self._build_table_tab(), "Metric Matrix")
        layout.addWidget(self.tabs)
        
        self._dirty_tabs = {i: False for i in range(self.tabs.count())}
        self.tabs.currentChanged.connect(self.render_active_tab)
        return panel

    def _build_chart_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(15)
        
        lbl_chart_lead = QLabel("Target Horizon:")
        lbl_chart_lead.setStyleSheet("color: #a1a1aa; font-weight: bold; font-size: 12px; text-transform: uppercase;")
        
        self.combo_chart_lead = QComboBox()
        self.combo_chart_lead.addItems(["5", "15", "30", "60", "120", "180"])
        self.combo_chart_lead.currentTextChanged.connect(self.update_plots)
        self.combo_chart_lead.setFixedWidth(80)
        
        lbl_chart_model = QLabel("Primary Model:")
        lbl_chart_model.setStyleSheet("color: #a1a1aa; font-weight: bold; font-size: 12px; text-transform: uppercase;")
        
        self.combo_chart_target_model = QComboBox()
        self.combo_chart_target_model.currentTextChanged.connect(self.update_plots)
        self.combo_chart_target_model.setMinimumWidth(180)
        
        lbl_filter = QLabel("Visibility Toggles:")
        lbl_filter.setStyleSheet("color: #a1a1aa; font-weight: bold; font-size: 12px; text-transform: uppercase;")
        
        self.scroll_models = QScrollArea()
        self.scroll_models.setWidgetResizable(True)
        self.scroll_models.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_models.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_models.setFixedHeight(30)
        self.scroll_models.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        widget_models = QWidget()
        widget_models.setStyleSheet("background: transparent;")
        
        self.hl_models = QHBoxLayout(widget_models)
        self.hl_models.setContentsMargins(0, 0, 0, 0)
        self.hl_models.setSpacing(12)
        self.hl_models.setAlignment(Qt.AlignLeft)
        
        self.scroll_models.setWidget(widget_models)
        
        layout.addWidget(lbl_chart_lead)
        layout.addWidget(self.combo_chart_lead)
        layout.addWidget(QLabel("m", styleSheet="color:#71717a; font-weight:bold;"))
        layout.addSpacing(20)
        
        layout.addWidget(lbl_chart_model)
        layout.addWidget(self.combo_chart_target_model)
        layout.addSpacing(20)
        
        layout.addWidget(lbl_filter)
        layout.addWidget(self.scroll_models)
        
        return layout

    def _build_table_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        ctrl_layout = QHBoxLayout()
        lbl_lead = QLabel("Evaluation Boundary:")
        lbl_lead.setStyleSheet("color: #22d3ee; font-weight: bold; font-size: 13px; text-transform: uppercase;")
        
        self.combo_table_lead = QComboBox()
        self.combo_table_lead.addItems(["5", "15", "30", "60", "120", "180"])
        self.combo_table_lead.currentTextChanged.connect(lambda _: self.update_table())
        self.combo_table_lead.setFixedWidth(80)
        
        ctrl_layout.addWidget(lbl_lead)
        ctrl_layout.addWidget(self.combo_table_lead)
        ctrl_layout.addWidget(QLabel(" minutes", styleSheet="color:#71717a; font-weight:bold;"))
        ctrl_layout.addStretch()
        
        self.table_res = QTableWidget()
        self.table_res.setAlternatingRowColors(True)
        self.table_res.verticalHeader().setVisible(False)
        self.table_res.setFocusPolicy(Qt.NoFocus)
        self.table_res.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table_res.horizontalHeader().setStretchLastSection(True)
        
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.table_res)
        return widget

    def log(self, msg: str):
        from datetime import datetime
        clean_msg = str(msg).replace('<', '&lt;').replace('>', '&gt;')
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_area.append(f"<span style='color:#71717a;'>[{timestamp}]</span> {clean_msg}")
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _flush_cache(self):
        confirm = QMessageBox.question(
            self, "Clear Cache", 
            "Execute strict cache purge of all compiled neural weights?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                protected_path = self.e_transfer_weights.text().strip()
                protected_filename = os.path.basename(protected_path) if protected_path else ""

                deleted_count = 0
                for filename in os.listdir("cache"):
                    file_path = os.path.join("cache", filename)
                    
                    if protected_filename and filename == protected_filename:
                        self.log(f"Shielded transfer weights from purge: {filename}")
                        continue
                        
                    if os.path.isfile(file_path): 
                        os.unlink(file_path)
                        deleted_count += 1
                        
                self.log(f"Cache purge complete. {deleted_count} temporary files removed.")
            except Exception as e: 
                self.log(f"Failed to clear cache: {e}")

    def add_site(self):
        try:
            dlg = AddSiteDialog(self)
            if dlg.exec():
                file_path = dlg.e_file.text()
                target_col = dlg.e_target.text().strip()
                
                if not target_col:
                    try: 
                        headers = pd.read_csv(file_path, nrows=0).columns
                        target_col = [c for c in headers if 'pv' in str(c).lower() or 'power' in str(c).lower()][-1]
                    except Exception: 
                        target_col = "target"
                
                capacity_val = float(dlg.e_cap.text()) if dlg.e_cap.text().strip() else None
                exog_features = [x.strip() for x in dlg.e_exog.text().split(',')] if dlg.e_exog.text().strip() else []
                
                site = SiteContext(
                    name=dlg.e_name.text().strip(), 
                    file_path=file_path, 
                    target_col=target_col, 
                    lat=float(dlg.e_lat.text()), 
                    lon=float(dlg.e_lon.text()), 
                    timezone=dlg.c_tz.currentText(), 
                    capacity=capacity_val, 
                    technology_type=dlg.c_tech.currentText(), 
                    location_label=dlg.e_loc.text().strip(), 
                    exog_cols=exog_features, 
                    inferred_flags={}
                )
                self.sites.append(site)
                self.db.save_site(site)
                self.list_sites.addItem(f"{site.name} | Target: {site.target_col}")
                self.log(f"Site Added: {site.name}")
        except Exception as e: 
            self.log(f"ERROR adding node: {str(e)}")

    def delete_site(self):
        selected_indices = [i.row() for i in self.list_sites.selectedIndexes()]
        if not selected_indices: return
        
        confirm = QMessageBox.question(
            self, "Remove Site", 
            "Purge selected site?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            for idx in sorted(selected_indices, reverse=True): 
                self.db.delete_site(self.sites[idx].name)
                del self.sites[idx]
                self.list_sites.takeItem(idx)

    def plot_site_correlation(self):
        selected_indices = [i.row() for i in self.list_sites.selectedIndexes()]
        if not selected_indices: 
            return QMessageBox.warning(self, "Selection Required", "Please highlight a site first.")
            
        self.lbl_status.setText("COMPUTING MATRIX...")
        self.lbl_status.setStyleSheet("color: #fbbf24; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")
        
        site = self.sites[selected_indices[0]]
        worker = CorrelationWorker(site)
        worker.signals.matrix_ready.connect(self._on_matrix_ready)
        worker.signals.error.connect(lambda e: self.log(f"Matrix error: {e}"))
        self.thread_pool.start(worker)

    def _on_matrix_ready(self, df_corr):
        selected_indices = [i.row() for i in self.list_sites.selectedIndexes()]
        if selected_indices:
            site = self.sites[selected_indices[0]]
            self.p_corr.plot_correlation_matrix(df_corr, site.target_col)
            self.tabs.setCurrentWidget(self.p_corr)
        self.lbl_status.setText("SYSTEM IDLE")
        self.lbl_status.setStyleSheet("color: #06b6d4; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")

    def run_experiment(self):
        selected_indices = [i.row() for i in self.list_sites.selectedIndexes()]
        if not selected_indices: 
            return QMessageBox.warning(self, "Execution Halted", "You must click and highlight a Dataset/Site before running.")
            
        run_sites = [self.sites[i] for i in selected_indices]
        
        models_to_run = []
        optuna_flags = {}
        
        for r in range(self.table_models.rowCount()):
            item = self.table_models.item(r, 0)
            if not item: continue
            
            model_name = item.text().strip()
            widget_run = self.table_models.cellWidget(r, 1)
            
            if not (widget_run and widget_run.chk.isChecked()): 
                continue
                
            deep_learning_models = ["LSTM", "GRU", "BiLSTM", "PatchTST", "Transformer", "iTransformer", "TCN-iTransformer"]
            
            if model_name in deep_learning_models:
                widget_wpd = self.table_models.cellWidget(r, 2)
                widget_opt = self.table_models.cellWidget(r, 3)
                
                is_wpd = widget_wpd.chk.isChecked() if widget_wpd else False
                is_opt = widget_opt.chk.isChecked() if widget_opt else False
                
                final_model_name = f"WPD-{model_name}" if is_wpd else model_name
                models_to_run.append(final_model_name)
                optuna_flags[final_model_name] = is_opt
            else: 
                models_to_run.append(model_name)

        if not models_to_run: 
            return QMessageBox.warning(self, "Requirement", "Select at least one model.")
        
        try:
            dynamic_horizons = [int(h.strip()) for h in self.e_horizons.text().split(',')]
            
            self.combo_chart_lead.blockSignals(True)
            self.combo_chart_lead.clear()
            self.combo_chart_lead.addItems([str(h) for h in dynamic_horizons])
            self.combo_chart_lead.blockSignals(False)
            
            self.combo_table_lead.blockSignals(True)
            self.combo_table_lead.clear()
            self.combo_table_lead.addItems([str(h) for h in dynamic_horizons])
            self.combo_table_lead.blockSignals(False)
            
        except ValueError: 
            return QMessageBox.warning(self, "Horizon Error", "Horizons must be comma-separated integers (e.g., 15, 30).")

        transfer_path = self.e_transfer_weights.text().strip()
        if transfer_path and not os.path.exists(transfer_path): 
            return QMessageBox.warning(self, "Transfer Error", "The weights file does not exist.")
        
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.prog.setValue(0)
        self.exps.clear()
        
        self.lbl_status.setText("TRAINING ACTIVE")
        self.lbl_status.setStyleSheet("color: #3b82f6; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")
        
        self.log(f"Initializing Engine for {len(run_sites)} site(s)...")
        if transfer_path: 
            self.log(f"Transfer Learning enabled. Loading base weights from: {transfer_path}")
        
        cfgs = []
        for site in run_sites:
            # --- ACADEMIC FIX: AUTO-REGULATOR FOR AMORPHOUS SILICON ---
            is_amorphous = 'amor' in getattr(site, 'technology_type', '').lower()
            
            if is_amorphous:
                self.log(f"[{site.name}] AUTO-DETECTED AMORPHOUS SILICON: Overriding parameters to prevent Clear-Sky Collapse.")
                final_use_adv = False                # Turns off astronomical "clock"
                final_feat_sel = "Manual"            # Forces physical GHI inclusion
                final_ft_epochs = 80                 # Longer fine-tune runway for Amorphous
            else:
                final_use_adv = self.chk_adv.isChecked()
                final_feat_sel = self.combo_feat_sel.currentText()
                final_ft_epochs = self.spin_ft_epochs.value()

            cfg = ExperimentConfig(
                site_name=site.name, 
                data_path=site.file_path, 
                target_col=site.target_col, 
                n_folds=self.spin_folds.value(), 
                use_ramp_features=self.chk_ramp.isChecked(), 
                use_advanced_features=final_use_adv,         # DYNAMIC INJECTION
                use_cache=self.chk_cache.isChecked(), 
                strict_qa_mode=self.chk_strict.isChecked(), 
                modify_target=True, 
                selected_models=models_to_run, 
                model_optuna=optuna_flags, 
                feature_selection_mode=final_feat_sel,       # DYNAMIC INJECTION
                target_leads_minutes=dynamic_horizons, 
                n_trials=self.spin_trials.value(), 
                transfer_weights_path=transfer_path if transfer_path else None,
                freeze_base_layers=True, 
                fine_tune_epochs=final_ft_epochs             # DYNAMIC INJECTION
            )
            cfgs.append(cfg)
        
        self.worker = WorkerThread(cfgs, run_sites)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.prog.setValue)
        
        def safe_status_update(txt):
            self.lbl_status.setText(f"{txt}")
            self.lbl_status.setStyleSheet("color: #fbbf24; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")
            
        self.worker.status.connect(safe_status_update)
        
        def handle_model_done(result_obj):
            try:
                self.exps.append(result_obj)
                self._populate_model_dropdown()
                self.db.save_exp(result_obj)
                self._refresh_db_list()
                self.update_plots()
                self.tabs.setCurrentIndex(self.TAB_METRICS_INDEX) 
            except Exception as e: 
                self.log(f"UI Error processing result: {str(e)}")
            
        self.worker.model_finished.connect(handle_model_done)
        self.worker.finished_run.connect(self.on_done)
        self.worker.error.connect(self.on_worker_error)
        
        self.worker.start()

    def cancel_experiment(self):
        if hasattr(self, 'worker'): 
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("CANCELLING...")
            self.lbl_status.setStyleSheet("color: #ef4444; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")

    def on_worker_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("CRITICAL ERROR")
        self.lbl_status.setStyleSheet("color: #ef4444; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")
        QMessageBox.critical(self, "Error", f"Engine failed:\n\n{err_msg}")

    def on_done(self, results):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("TRAINING COMPLETE")
        self.lbl_status.setStyleSheet("color: #22c55e; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")
        self._populate_model_dropdown()

    def _refresh_db_list(self):
        self.combo_db.clear()
        
        groups = {}
        for record in self.db.list_all():
            fingerprint = record.get('run_fingerprint') or record['timestamp'][:16]
            if fingerprint not in groups: 
                groups[fingerprint] = []
            groups[fingerprint].append(record)
            
        for fingerprint, group in groups.items(): 
            display_text = f"{group[0]['timestamp'][:16].replace('T', ' ')} | {group[0]['site']} | {len(group)} Models"
            self.combo_db.addItem(display_text, fingerprint)

    def load_db(self):
        fingerprint = self.combo_db.currentData()
        if not fingerprint:
            return
            
        self.lbl_status.setText("LOADING DATABASE ARTIFACTS...")
        self.lbl_status.setStyleSheet("color: #fbbf24; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")
        
        worker = DbLoadWorker(self.db, fingerprint)
        worker.signals.experiments_ready.connect(self._on_db_loaded)
        worker.signals.error.connect(lambda e: self.log(f"DB Load Error: {e}"))
        self.thread_pool.start(worker)

    def _on_db_loaded(self, loaded_exps):
        self.exps = loaded_exps
        self._populate_model_dropdown()
        self.update_plots()
        self.tabs.setCurrentIndex(self.TAB_METRICS_INDEX)
        self.lbl_status.setText("SYSTEM IDLE")
        self.lbl_status.setStyleSheet("color: #06b6d4; font-weight: bold; background: #09090b; padding: 4px; border: 1px solid #27272a;")

    def _populate_model_dropdown(self):
        if not self.exps: return
        
        current_target = self.combo_chart_target_model.currentText()
        models = sorted(list(set([e.model_name for e in self.exps])))
        
        self.combo_chart_target_model.blockSignals(True)
        self.combo_chart_target_model.clear()
        self.combo_chart_target_model.addItems(models)
        if current_target in models: 
            self.combo_chart_target_model.setCurrentText(current_target)
        elif models: 
            self.combo_chart_target_model.setCurrentText(models[0])
        self.combo_chart_target_model.blockSignals(False)

        existing_states = {cb.text(): cb.isChecked() for cb in self.model_checkboxes}
        
        for i in reversed(range(self.hl_models.count())): 
            widget = self.hl_models.itemAt(i).widget()
            if widget: 
                widget.deleteLater()
        
        self.model_checkboxes = []
        for model_name in models:
            cb = QCheckBox(model_name)
            cb.setChecked(existing_states.get(model_name, True))
            cb.setStyleSheet("color: #e4e4e7; font-weight: bold; font-size: 11px; padding-right: 12px;")
            cb.stateChanged.connect(self.update_plots)
            
            self.hl_models.addWidget(cb)
            self.model_checkboxes.append(cb)
            
        self.hl_models.addStretch()

    def update_table(self, filtered_exps=None):
        exps_to_show = filtered_exps if filtered_exps is not None else self.exps
        if isinstance(exps_to_show, str): 
            exps_to_show = self.exps
        
        if not exps_to_show: 
            self.table_res.clearContents()
            self.table_res.setRowCount(0)
            return
            
        lead_min = int(self.combo_table_lead.currentText())
        
        headers = [
            "Target Area", "Model", "Cache"
        ] + [
            f"{m} (L{lead_min})" for m in [
                "RMSE", "RMSE_Day", "RMSE_Ramp", "Skill_Day%", "Skill_Ramp%", 
                "nRMSE_c%", "nRMSE_m%", "MAE", "MAPE", "sMAPE", "MBE", 
                "R2", "CRPS", "PICP", "MPIW", "DM_p"
            ]
        ]
        
        self.table_res.clearContents()
        self.table_res.setColumnCount(len(headers))
        self.table_res.setHorizontalHeaderLabels(headers)
        self.table_res.setRowCount(len(exps_to_show))
        
        for row_idx, exp in enumerate(exps_to_show):
            
            def get_mean_metric(metric_key):
                vals = [f.metrics.get(f"{metric_key}_L{lead_min}") for f in exp.folds if f.metrics.get(f"{metric_key}_L{lead_min}") is not None]
                return np.mean(vals) if vals else None
            
            is_cached = any(f.loaded_from_cache for f in exp.folds)
            base_info = [exp.site_name, exp.model_name, str(is_cached)]
            
            for col_idx, text in enumerate(base_info): 
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_res.setItem(row_idx, col_idx, item)
            
            current_col = 3
            metric_keys = [
                "RMSE", "RMSE_Day", "RMSE_Ramp", "Skill_Day_%", "Skill_Ramp_%", 
                "nRMSE_c%", "nRMSE_m%", "MAE", "MAPE", "sMAPE", "MBE", "R2", 
                "CRPS_Approx", "PICP", "MPIW"
            ]
            
            for m_key in metric_keys:
                val = get_mean_metric(m_key)
                item = QTableWidgetItem(f"{val:.2f}" if val is not None else "N/A")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_res.setItem(row_idx, current_col, item)
                current_col += 1
                
            pval = get_mean_metric('DM_p')
            item_p = QTableWidgetItem(f"{pval:.4f}" if pval is not None else "1.0000")
            item_p.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if pval is not None and pval < 0.05: 
                item_p.setBackground(QColor("#0891b2"))
                item_p.setForeground(QColor("white"))
                
            self.table_res.setItem(row_idx, current_col, item_p)
            
        self.table_res.resizeColumnsToContents()

    def update_plots(self):
        if not self.exps: return
        for i in range(self.tabs.count()): 
            self._dirty_tabs[i] = True
        self.render_active_tab()

    def render_active_tab(self):
        if not self.exps: return
        
        idx = self.tabs.currentIndex()
        if not getattr(self, '_dirty_tabs', {}).get(idx, True): 
            return
        
        target_lead_min = int(self.combo_chart_lead.currentText())
        target_model = self.combo_chart_target_model.currentText()
        
        selected_models = [cb.text() for cb in self.model_checkboxes if cb.isChecked()]
        filtered_exps = [e for e in self.exps if e.model_name in selected_models]

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            if idx == 0: pass 
            elif idx == 1: self.p_ts.plot_ts(filtered_exps, target_lead_min)
            elif idx == 2: self.p_ts_horizons.plot_ts_horizons(self.exps, target_model)
            elif idx == 3: self.p_parity.plot_scatter_parity(filtered_exps, target_lead_min)
            elif idx == 4: self.p_box.plot_error_box(filtered_exps, target_lead_min)
            elif idx == 5: self.p_radar.plot_radar_metrics(filtered_exps, target_lead_min)
            elif idx == 6: self.p_tay.plot_taylor(filtered_exps, target_lead_min)
            elif idx == 7: self.p_bar.plot_bars(filtered_exps, target_lead_min)
            elif idx == 8: self.p_wpd.plot_wpd_diag(filtered_exps, target_lead_min)
            elif idx == 9: self.p_rel.plot_reliability(filtered_exps, target_lead_min)
            elif idx == 10: self.p_xai.plot_xai_importance(filtered_exps, target_lead_min)
            elif idx == 11: self.p_3d.plot_3d_uncertainty_manifold(filtered_exps, target_lead_min)
            elif idx == self.TAB_METRICS_INDEX: self.update_table(filtered_exps)
            
            self._dirty_tabs[idx] = False
            
        except Exception as e: 
            self.log(f"Warning: Plot rendering error on tab {idx}: {e}")
            traceback.print_exc()
        finally: 
            QApplication.restoreOverrideCursor()

    def export_xl(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "exports/academic_results.xlsx", "Excel Spreadsheet (*.xlsx)"
        )
        if file_path: 
            self.export_thread = ExportWorker(self.exps, file_path)
            self.export_thread.error.connect(
                lambda e: QMessageBox.critical(self, "Export Error", f"Failed:\n{e}")
            )
            self.export_thread.finished.connect(
                lambda p: QMessageBox.information(self, "Success", f"Data exported successfully to:\n{p}")
            )
            self.export_thread.start()