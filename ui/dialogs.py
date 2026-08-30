# ui/dialogs.py
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QPushButton, 
    QFileDialog, QHBoxLayout, QComboBox, QDialogButtonBox, QMessageBox
)

class AddSiteDialog(QDialog):
    """Configuration modal for deploying a new PV array telemetry node."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure PV Array Telemetry Node")
        self.setMinimumWidth(550)
        
        layout = QFormLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        self.e_name = QLineEdit()
        self.e_name.setPlaceholderText("e.g., Experimental_Node_01")
        
        self.e_file = QLineEdit()
        self.btn_file = QPushButton("Select Telemetry Matrix...")
        self.btn_file.clicked.connect(self._select_file)
        
        flay = QHBoxLayout()
        flay.addWidget(self.e_file)
        flay.addWidget(self.btn_file)
        
        self.e_target = QLineEdit()
        self.e_target.setPlaceholderText("Leave blank for automatic detection")
        
        self.e_cap = QLineEdit()
        self.e_cap.setPlaceholderText("e.g., 2000 (Watts for Amorphous)") 
        
        self.e_lat = QLineEdit()
        self.e_lat.setPlaceholderText("e.g., 32.299")
        
        self.e_lon = QLineEdit()
        self.e_lon.setPlaceholderText("e.g., -9.237")
        
        self.e_loc = QLineEdit()
        self.e_loc.setPlaceholderText("e.g., Safi, Morocco")
        
        self.c_tz = QComboBox()
        self.c_tz.addItems(['UTC', 'Africa/Casablanca', 'Europe/Paris', 'America/New_York'])
        
        self.c_tech = QComboBox()
        self.c_tech.addItems(['Monocrystalline', 'Polycrystalline', 'Amorphous', 'Thin-Film', 'Bifacial', 'Unknown'])
        
        self.e_exog = QLineEdit()
        self.e_exog.setPlaceholderText("Comma separated (e.g., temp, ghi, w_spd)")
        
        form_fields = [
            ("Node ID *:", self.e_name), 
            ("Data Vector *:", flay), 
            ("Latitude *:", self.e_lat), 
            ("Longitude *:", self.e_lon), 
            ("Timezone:", self.c_tz), 
            ("Technology:", self.c_tech), 
            ("Geospatial:", self.e_loc), 
            ("Target Col:", self.e_target), 
            ("Theoretical Capacity (W):", self.e_cap), 
            ("Exogenous Covariates:", self.e_exog)
        ]
        
        for label, widget in form_fields: 
            layout.addRow(label, widget)
            
        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        layout.addRow(self.btns)
        
    def _select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Empirical Dataset", "", "Data Matrix (*.csv *.xlsx)")
        if file_path:
            self.e_file.setText(file_path)
            
    def accept(self):
        if not all([self.e_name.text(), self.e_file.text(), self.e_lat.text(), self.e_lon.text()]): 
            return QMessageBox.warning(self, "Audit Validation", "Node ID, Telemetry File, Latitude, and Longitude are strictly required.")
        try: 
            float(self.e_lat.text())
            float(self.e_lon.text())
        except ValueError: 
            return QMessageBox.warning(self, "Audit Validation", "Latitude and Longitude parameters must be floating point decimals.")
        super().accept()