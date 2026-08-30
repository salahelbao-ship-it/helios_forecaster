import sys
import os
from pathlib import Path

# ==============================================================================
# NUCLEAR PATH RESOLUTION
# This forces Python to recognize the exact absolute path of this project.
# ==============================================================================
# Get the absolute path to the folder containing main.py
PROJECT_ROOT = str(Path(__file__).resolve().parent)

# Force the operating system to change its working directory to this folder
os.chdir(PROJECT_ROOT)

# Force Python's import system to look in this folder first
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Diagnostic Check ---
try:
    from utils.helpers import set_global_seed, enforce_gpu_memory_growth, setup_directories
    from ui.app import DashboardApp
    from PySide6.QtWidgets import QApplication, QStyleFactory
    from PySide6.QtGui import QColor, QPalette
except ModuleNotFoundError as e:
    print("\n" + "="*60)
    print(f"❌ CRITICAL IMPORT FAILURE: {e}")
    print("="*60)
    print(f"Python thinks the project root is: {PROJECT_ROOT}")
    print("\nHere is what Python actually sees inside that folder right now:")
    for item in os.listdir(PROJECT_ROOT):
        print(f"  - {item}")
    print("\nIf you do not see a folder named 'utils' in the list above,")
    print("then the folder is missing, misnamed, or you are running the wrong script.")
    print("="*60 + "\n")
    sys.exit(1)


def main():
    """Bootstraps the Helios-Grid PV Forecasting Dashboard."""
    # 1. Initialize Environment
    set_global_seed(42)
    enforce_gpu_memory_growth()
    setup_directories(PROJECT_ROOT)

    # 2. Initialize Qt Application
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # 3. Apply Global Color Palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(11, 17, 32))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(248, 250, 252))
    app.setPalette(palette)

    # 4. Launch Main Window
    window = DashboardApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()