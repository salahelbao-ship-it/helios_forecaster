# core/datatypes.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

class CancelledError(Exception): pass
class DataQualityError(Exception): pass
class AlignmentError(Exception): pass
class CausalityLeakError(Exception): pass

@dataclass
class SiteContext:
    name: str
    file_path: str
    target_col: str
    lat: float
    lon: float
    timezone: str
    capacity: float
    technology_type: str
    location_label: str
    exog_cols: List[str]
    inferred_flags: Dict = field(default_factory=dict)

@dataclass
class FoldDefinition:
    fold_id: int
    train_start: str
    train_end: str
    val_start: str = ""
    val_end: str = ""
    train_size: int = 0
    val_size: int = 0
    gap_size: int = 0
    inferred_freq: str = ""
    test_start: str = ""
    test_end: str = ""
    test_size: int = 0
    embargo_steps: int = 0
    
    def __post_init__(self):
        if self.test_start and not self.val_start: self.val_start = self.test_start
        if self.test_end and not self.val_end: self.val_end = self.test_end
        if self.test_size and not self.val_size: self.val_size = self.test_size
        if self.embargo_steps and not self.gap_size: self.gap_size = self.embargo_steps

@dataclass
class FoldResult:
    fold_id: int
    issue_times: List[str]
    valid_times: List[str]
    y_true: List[float]
    y_pred_quantiles: Dict[str, List[float]]
    baseline_pers: List[float]
    metrics: Dict[str, float]
    pinball_by_q: Dict[str, float]
    crossing_rate_pre_repair: float
    params: Dict
    fold_def: FoldDefinition
    training_duration_sec: float = 0.0
    loaded_from_cache: bool = False
    feature_selection_report: dict = field(default_factory=dict)
    degraded_mode: bool = False 
    data_file_path: str = "" 
    artifact_hash: str = "" 

    # --- EXPORT SCHEMA FIX ---
    @property
    def y_pred(self) -> List[float]:
        """Intercepts legacy requests and safely handles both float and integer schemas."""
        if '50' in self.y_pred_quantiles: return self.y_pred_quantiles['50']
        elif '0.5' in self.y_pred_quantiles: return self.y_pred_quantiles['0.5']
        elif 0.5 in self.y_pred_quantiles: return self.y_pred_quantiles[0.5]
        return [0.0] * len(self.y_true)

@dataclass
class ExperimentResult:
    model_name: str
    site_name: str
    config_id: str
    seq_len: int
    timestamp_created: str
    capacity: float
    horizon: int
    folds: List[FoldResult]
    training_time_total: float = 0.0
    run_fingerprint: str = ""
    env_meta: dict = field(default_factory=dict)
    site_meta: dict = field(default_factory=dict)
    qa_report: dict = field(default_factory=dict)
    config_dump: dict = field(default_factory=dict)
    methodology: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class ExperimentConfig:
    site_name: str
    data_path: str
    target_col: str
    n_folds: int
    use_ramp_features: bool
    use_advanced_features: bool
    use_cache: bool
    strict_qa_mode: bool
    modify_target: bool
    selected_models: List[str]
    model_optuna: Dict[str, bool]
    feature_selection_mode: str
    target_leads_minutes: List[int]
    n_trials: int
    dataset_hash: str = ""
    forecast_horizon: int = 0
    embargo_gap: int = 24
    sequence_length: int = 96 
    exog_availability: str = "Perfect"
    evaluation_mode: str = "Strict"
    
    transfer_weights_path: Optional[str] = None
    freeze_base_layers: bool = True
    
    # --- TL STABILITY FIX: Longer runway, EarlyStopping will catch the peak ---
    fine_tune_epochs: int = 50 

    use_optuna: bool = True 
    
    def get_hash(self):
        import hashlib, json
        cfg_dict = self.__dict__.copy()
        cfg_dict.pop('dataset_hash', None)
        return hashlib.md5(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()