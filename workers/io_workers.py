# workers/io_workers.py
import pandas as pd
from typing import List
from PySide6.QtCore import QObject, QRunnable, Signal

from core.datatypes import SiteContext, ExperimentResult
from data.database import ExperimentRegistry
from data.loader import DataLoader

class IOWorkerSignals(QObject):
    headers_ready = Signal(list)
    matrix_ready = Signal(pd.DataFrame)
    experiments_ready = Signal(list)
    error = Signal(str)

class CsvHeaderWorker(QRunnable):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.signals = IOWorkerSignals()

    def run(self):
        try:
            df = pd.read_csv(self.file_path, nrows=0)
            self.signals.headers_ready.emit(list(df.columns))
        except Exception as e:
            self.signals.error.emit(f"Failed to read CSV headers: {str(e)}")

class CorrelationWorker(QRunnable):
    def __init__(self, site: SiteContext):
        super().__init__()
        self.site = site
        self.signals = IOWorkerSignals()

    def run(self):
        try:
            df, _, _ = DataLoader.load_df(
                self.site.file_path, self.site.timezone, 
                modify_target=False, strict_mode=False, target_col=self.site.target_col
            )
            num_df = df.select_dtypes(include=['number'])
            if num_df.empty:
                raise ValueError(f"No numeric columns found in {self.site.file_path}")
            self.signals.matrix_ready.emit(num_df.corr())
        except Exception as e:
            self.signals.error.emit(f"Failed to compute correlation matrix: {str(e)}")

class DbLoadWorker(QRunnable):
    def __init__(self, db: ExperimentRegistry, fingerprint: str):
        super().__init__()
        self.db = db
        self.fingerprint = fingerprint
        self.signals = IOWorkerSignals()

    def run(self):
        try:
            exps: List[ExperimentResult] = []
            for record in self.db.list_all():
                if record.get('run_fingerprint') == self.fingerprint or record['timestamp'][:16] == self.fingerprint:
                    loaded_exp = self.db.load(record['id'])
                    exps.append(loaded_exp)
            if not exps:
                raise ValueError(f"No records found for fingerprint: {self.fingerprint}")
            self.signals.experiments_ready.emit(exps)
        except Exception as e:
            self.signals.error.emit(f"Database load error: {str(e)}")