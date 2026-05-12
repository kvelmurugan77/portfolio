#!/usr/bin/env python3
"""
WindFarm Designer Pro - Main Entry Point
=========================================
A comprehensive wind farm design and energy assessment desktop application
similar to WindPRO. Features include:
  - SRTM1 terrain data download (30m resolution, 20km buffer)
  - Land cover / roughness data download
  - Global Wind Atlas mesoscale wind data integration
  - Wind farm layout optimization (Grid, Greedy, PSO, GA algorithms)
  - Wind flow modeling over complex terrain
  - AEP calculation with wake effects (Jensen, Frandsen, Ainslie models)
  - Mast data import for wind resource assessment
  - Professional results export (CSV, PDF reports)

Usage:
    python main.py              # Launch the GUI application
    python main.py --console    # Launch in console mode (headless)
    python main.py --version    # Print version and exit
"""

import sys
import os
import argparse
import logging
import traceback

# Ensure the project root is on the Python path
# This allows both direct execution and frozen (.exe) execution
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    APPLICATION_PATH = os.path.dirname(sys.executable)
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))

# Add project root to path for src imports
sys.path.insert(0, APPLICATION_PATH)


def setup_logging():
    """Configure application-wide logging."""
    log_dir = os.path.join(APPLICATION_PATH, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'windfarm_designer.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)


def check_dependencies():
    """Verify that all required dependencies are installed."""
    missing = []

    try:
        import PyQt5
    except ImportError:
        missing.append('PyQt5')

    try:
        import numpy
    except ImportError:
        missing.append('numpy')

    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')

    try:
        import scipy
    except ImportError:
        missing.append('scipy')

    try:
        import requests
    except ImportError:
        missing.append('requests')

    try:
        import shapely
    except ImportError:
        missing.append('shapely')

    if missing:
        print("=" * 60)
        print("ERROR: Missing required Python packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("Install all dependencies with:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        sys.exit(1)

    # Optional dependencies (warn but don't fail)
    optional = []
    try:
        import rasterio
    except ImportError:
        optional.append('rasterio (for GeoTIFF handling - strongly recommended)')

    try:
        import pyproj
    except ImportError:
        optional.append('pyproj (for accurate coordinate transformations)')

    try:
        import cdsapi
    except ImportError:
        optional.append('cdsapi (for Copernicus CDS data download)')

    if optional:
        print("WARNING: Optional packages not installed (some features limited):")
        for pkg in optional:
            print(f"  - {pkg}")
        print()


def launch_gui():
    """Launch the PyQt5 GUI application."""
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt

    from src.gui.main_window import MainWindow

    # High DPI support — MUST be set BEFORE creating QApplication instance
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("WindFarm Designer Pro")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("WindFarm Designer")

    # Set application-wide style
    app.setStyle("Fusion")

    # Create main window
    window = MainWindow()
    window.show()

    # Handle uncaught exceptions
    def exception_hook(exctype, value, tb):
        logger = logging.getLogger('main')
        logger.error(f"Uncaught exception: {exctype.__name__}: {value}")
        logger.error("".join(traceback.format_tb(tb)))

        # Show error dialog
        msg = QMessageBox(window)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(f"An unexpected error occurred:\n\n{value}")
        msg.setDetailedText("".join(traceback.format_tb(tb)))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    sys.excepthook = exception_hook

    # Run event loop
    sys.exit(app.exec_())


def console_mode():
    """Run in headless/console mode (for batch processing)."""
    print("WindFarm Designer Pro - Console Mode")
    print("=====================================")
    print("Console mode is not yet implemented.")
    print("Use the GUI mode for full functionality: python main.py")
    print()
    print("Available options:")
    print("  python main.py              # Launch GUI")
    print("  python main.py --version    # Show version")
    print("  python main.py --help       # Show help")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='WindFarm Designer Pro - Wind Farm Design & Energy Assessment Tool'
    )
    parser.add_argument('--version', action='version', version='WindFarm Designer Pro v1.0.0')
    parser.add_argument('--console', action='store_true',
                        help='Run in console mode (no GUI)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger('main')

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("WindFarm Designer Pro v1.0.0 starting...")
    logger.info(f"Application path: {APPLICATION_PATH}")
    logger.info(f"Python version: {sys.version}")

    # Check dependencies
    check_dependencies()

    if args.console:
        console_mode()
    else:
        try:
            launch_gui()
        except Exception as e:
            logger.critical(f"Failed to start application: {e}")
            logger.critical(traceback.format_exc())
            print(f"CRITICAL ERROR: Failed to start application: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
