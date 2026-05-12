"""
WindFarm Designer Pro - Main Application Window.
Provides the primary GUI shell with tabbed interface, menus, toolbar,
status bar, and a shared project data store passed between tabs.
"""

import logging
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QStatusBar, QMenuBar,
    QToolBar, QAction, QMessageBox, QFileDialog, QProgressBar, QLabel,
    QStyle, QDesktopWidget, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor

from src.gui.project_tab import ProjectTab
from src.gui.terrain_tab import TerrainTab
from src.gui.roughness_tab import RoughnessTab
from src.gui.wind_tab import WindTab
from src.gui.layout_tab import LayoutTab
from src.gui.flow_tab import FlowTab
from src.gui.results_tab import ResultsTab
from src.utils.data_utils import save_project, load_project

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for WindFarm Designer Pro.

    Hosts 7 workflow tabs, menu bar, toolbar, and status bar.
    Manages a central project data dictionary that is shared across all tabs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WindFarm Designer Pro")
        self.setMinimumSize(1280, 800)
        self._apply_dark_theme()

        # Central project data store
        self._project_data = {
            'project_name': '',
            'center_lat': 0.0,
            'center_lon': 0.0,
            'boundary': None,          # list of (lat, lon)
            'bbox': None,              # (min_lon, min_lat, max_lon, max_lat)
            'capacity_mw': 0.0,
            'turbine_model': '',
            'turbine_spec': None,      # dict from get_turbine_spec
            'power_curve': None,       # dict from generate_default_power_curve
            'buffer_km': 20.0,
            'wtg_layout': None,        # list of WTGPosition dicts
            'mast_data': None,         # dict from load_mast_data
            'terrain': {},             # terrain mosaic paths, tiles, etc.
            'roughness': {},           # roughness data paths
            'wind_data': {},           # GWA wind resource data
            'flow_results': None,      # flow model output
            'aep_results': None,       # AEP calculation output
            'layout_results': None,    # layout optimizer output
        }

        self._current_project_path = None

        self._build_ui()
        self._build_menubar()
        self._build_toolbar()
        self._build_statusbar()
        self.update_status("Ready. Create a new project or open an existing one.")
        logger.info("WindFarm Designer Pro initialized.")

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self):
        """Apply a professional dark colour palette."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(35, 35, 38))
        palette.setColor(QPalette.AlternateBase, QColor(50, 50, 55))
        palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(55, 55, 58))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
        QApplication.instance().setPalette(palette)
        QApplication.instance().setStyleSheet("""
            QToolTip { color: #ffffff; background: #2a2a2a; border: 1px solid #555; }
            QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 4px;
                        margin-top: 8px; padding-top: 16px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QTabWidget::pane { border: 1px solid #555; }
            QTabBar::tab { padding: 6px 12px; margin-right: 2px; }
            QTabBar::tab:selected { background: #2a82da; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background: #2a82da; }
        """)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the central tab widget."""
        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)

        self._project_tab = ProjectTab(self)
        self._terrain_tab = TerrainTab(self)
        self._roughness_tab = RoughnessTab(self)
        self._wind_tab = WindTab(self)
        self._layout_tab = LayoutTab(self)
        self._flow_tab = FlowTab(self)
        self._results_tab = ResultsTab(self)

        self._tab_widget.addTab(self._project_tab, "1 \u2022 Project Setup")
        self._tab_widget.addTab(self._terrain_tab, "2 \u2022 Terrain Data")
        self._tab_widget.addTab(self._roughness_tab, "3 \u2022 Roughness Data")
        self._tab_widget.addTab(self._wind_tab, "4 \u2022 Wind Resource")
        self._tab_widget.addTab(self._layout_tab, "5 \u2022 Layout Optimizer")
        self._tab_widget.addTab(self._flow_tab, "6 \u2022 Wind Flow Model")
        self._tab_widget.addTab(self._results_tab, "7 \u2022 AEP Results")

        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._tab_widget)
        self.setCentralWidget(central)

    def _build_menubar(self):
        """Build application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        act_new = QAction("&New Project", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._action_new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project\u2026", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._action_open_project)
        file_menu.addAction(act_open)

        act_save = QAction("&Save Project", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._action_save_project)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_export = QAction("&Export Results\u2026", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._action_export_results)
        file_menu.addAction(act_export)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        act_settings = QAction("&Settings", self)
        act_settings.triggered.connect(self._action_settings)
        tools_menu.addAction(act_settings)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        act_about = QAction("&About", self)
        act_about.triggered.connect(self._action_about)
        help_menu.addAction(act_about)

    def _build_toolbar(self):
        """Build the main toolbar with common actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        act_new = QAction(self.style().standardIcon(QStyle.SP_FileIcon), "New", self)
        act_new.triggered.connect(self._action_new_project)
        toolbar.addAction(act_new)

        act_open = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Open", self)
        act_open.triggered.connect(self._action_open_project)
        toolbar.addAction(act_open)

        act_save = QAction(self.style().standardIcon(QStyle.SP_DriveHDIcon), "Save", self)
        act_save.triggered.connect(self._action_save_project)
        toolbar.addAction(act_save)

        toolbar.addSeparator()

        act_prev = QAction(self.style().standardIcon(QStyle.SP_ArrowBack), "Previous Tab", self)
        act_prev.triggered.connect(lambda: self._tab_widget.setCurrentIndex(
            max(0, self._tab_widget.currentIndex() - 1)))
        toolbar.addAction(act_prev)

        act_next = QAction(self.style().standardIcon(QStyle.SP_ArrowForward), "Next Tab", self)
        act_next.triggered.connect(lambda: self._tab_widget.setCurrentIndex(
            min(self._tab_widget.count() - 1, self._tab_widget.currentIndex() + 1)))
        toolbar.addAction(act_next)

    def _build_statusbar(self):
        """Build status bar with permanent progress bar and status label."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._status_label.setMinimumWidth(400)
        self._status_bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(260)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

    # ------------------------------------------------------------------
    # Public helpers for tabs
    # ------------------------------------------------------------------

    def update_status(self, message: str):
        """Update the status bar label."""
        self._status_label.setText(message)
        self._status_bar.repaint()
        logger.info(message)

    def update_progress(self, value: int, maximum: int = 100):
        """Update the progress bar.  *value* in [0, maximum]."""
        self._progress_bar.setMaximum(maximum)
        self._progress_bar.setValue(value)
        self._progress_bar.setVisible(maximum > 0 and value < maximum)
        QApplication.processEvents()

    def show_progress(self, visible: bool):
        """Show or hide the progress bar."""
        self._progress_bar.setVisible(visible)
        if not visible:
            self._progress_bar.setValue(0)

    def get_project_data(self) -> dict:
        """Return a reference to the central project data dict."""
        return self._project_data

    def set_project_data(self, data: dict):
        """Replace the central project data dict entirely."""
        self._project_data = data
        self._refresh_all_tabs()

    def _refresh_all_tabs(self):
        """Notify every tab to reload from project data."""
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if hasattr(widget, 'refresh_from_project'):
                widget.refresh_from_project()

    # ------------------------------------------------------------------
    # Tab switching
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int):
        """Called when the user switches tabs; refresh the target tab."""
        widget = self._tab_widget.widget(index)
        if widget is not None and hasattr(widget, 'refresh_from_project'):
            widget.refresh_from_project()
        tab_name = self._tab_widget.tabText(index)
        self.update_status(f"Switched to: {tab_name}")

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _action_new_project(self):
        if self._project_data.get('project_name'):
            reply = QMessageBox.question(
                self, "New Project",
                "Current project will be discarded. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        self._project_data = {
            'project_name': '', 'center_lat': 0.0, 'center_lon': 0.0,
            'boundary': None, 'bbox': None, 'capacity_mw': 0.0,
            'turbine_model': '', 'turbine_spec': None, 'power_curve': None,
            'buffer_km': 20.0, 'wtg_layout': None, 'mast_data': None,
            'terrain': {}, 'roughness': {}, 'wind_data': {},
            'flow_results': None, 'aep_results': None, 'layout_results': None,
        }
        self._current_project_path = None
        self._refresh_all_tabs()
        self._tab_widget.setCurrentIndex(0)
        self.update_status("New project created.")
        self.update_progress(0, 0)
        self.show_progress(False)

    def _action_open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "WindFarm Project (*.json);;All Files (*)")
        if not path:
            return
        try:
            data = load_project(path)
            self.set_project_data(data)
            self._current_project_path = path
            self.update_status(f"Project loaded from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")
            logger.exception("Failed to open project")

    def _action_save_project(self):
        if self._current_project_path:
            path = self._current_project_path
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Project", "windfarm_project.json",
                "WindFarm Project (*.json);;All Files (*)")
        if not path:
            return
        try:
            save_project(self._project_data, path)
            self._current_project_path = path
            self.update_status(f"Project saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")
            logger.exception("Failed to save project")

    def _action_export_results(self):
        self._tab_widget.setCurrentIndex(6)
        results_tab = self._results_tab
        if hasattr(results_tab, 'export_to_csv'):
            results_tab.export_to_csv()
        else:
            QMessageBox.information(self, "Export",
                                    "Switch to the AEP Results tab to export.")

    def _action_settings(self):
        QMessageBox.information(self, "Settings",
                                "Settings dialog is not yet implemented.\n\n"
                                "Configure options are available within each tab.")

    def _action_about(self):
        QMessageBox.about(self, "About WindFarm Designer Pro",
                          "<h3>WindFarm Designer Pro</h3>"
                          "<p>Version 1.0.0</p>"
                          "<p>A comprehensive wind energy assessment tool "
                          "for wind farm design, layout optimization, and "
                          "energy yield analysis.</p>"
                          "<p>Built with PyQt5, matplotlib, and scientific "
                          "Python libraries.</p>")

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._project_data.get('project_name') and self._current_project_path is None:
            reply = QMessageBox.question(
                self, "Exit",
                "You have unsaved changes. Exit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                return
        logger.info("WindFarm Designer Pro closed.")
        event.accept()


# ------------------------------------------------------------------
# Entry-point convenience
# ------------------------------------------------------------------

def run_app():
    """Create and launch the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("WindFarm Designer Pro")
    app.setOrganizationName("WindFarm")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
