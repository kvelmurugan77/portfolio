"""
WindFarm Designer Pro - Wind Resource Tab.
Downloads wind resource data from the Global Wind Atlas or imports mast data,
and shows wind speed map and wind rose previews.
"""

import logging
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

class _WindWorker(QThread):
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, bbox, source, height, grid_spacing, output_dir, parent=None):
        super().__init__(parent)
        self._bbox = bbox
        self._source = source
        self._height = height
        self._grid_spacing = grid_spacing
        self._output_dir = output_dir

    def run(self):
        try:
            if self._source == 'gwa':
                from src.core.gwa_downloader import GWADownloader
                dl = GWADownloader(
                    output_dir=self._output_dir,
                    grid_spacing_km=self._grid_spacing)
                dl.set_progress_callback(lambda c, t, m: self.progress.emit(c, t, m))
                dl.set_status_callback(lambda m: self.status.emit(m))
                result = dl.download_and_process(self._bbox, height=self._height)
                if not result:
                    self.finished_err.emit("GWA download produced no data.")
                else:
                    self.finished_ok.emit(result)
            elif self._source == 'mast':
                self.finished_err.emit(
                    "Mast import is handled directly in the GUI thread.")
            else:
                self.finished_err.emit(f"Unknown source: {self._source}")
        except Exception as e:
            self.finished_err.emit(str(e))


# ------------------------------------------------------------------
# Tab widget
# ------------------------------------------------------------------

class WindTab(QWidget):
    """Fourth workflow tab: download or import wind resource data."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Info ---
        self._bbox_label = QLabel("No boundary defined.")
        self._bbox_label.setWordWrap(True)
        layout.addWidget(self._bbox_label)

        # --- Settings ---
        settings_group = QGroupBox("Wind Data Settings")
        settings_form = QFormLayout()

        self._source_combo = QComboBox()
        self._source_combo.addItems(["Global Wind Atlas", "Mast Data"])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        settings_form.addRow("Data Source:", self._source_combo)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(50, 200)
        self._height_spin.setSuffix(" m")
        self._height_spin.setValue(100)
        settings_form.addRow("Hub Height:", self._height_spin)

        self._grid_spin = QDoubleSpinBox()
        self._grid_spin.setRange(0.5, 5.0)
        self._grid_spin.setDecimals(1)
        self._grid_spin.setSuffix(" km")
        self._grid_spin.setValue(1.0)
        self._grid_spin.setSingleStep(0.5)
        settings_form.addRow("Grid Spacing:", self._grid_spin)

        self._mast_btn = QPushButton("Import Mast Data\u2026")
        self._mast_btn.clicked.connect(self._import_mast)
        settings_form.addRow("Mast File:", self._mast_btn)
        self._mast_file_label = QLabel("No file selected")
        settings_form.addRow("", self._mast_file_label)

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # --- Actions ---
        action_row = QHBoxLayout()
        self._download_btn = QPushButton("\U0001F4E5  Download GWA Wind Data")
        self._download_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        self._download_btn.clicked.connect(self._start_download)
        action_row.addWidget(self._download_btn)
        layout.addLayout(action_row)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        layout.addWidget(self._log)

        # --- Summary stats ---
        stats_group = QGroupBox("Wind Resource Summary")
        stats_form = QFormLayout()
        self._mean_speed_label = QLabel("\u2014")
        self._mean_pd_label = QLabel("\u2014")
        self._n_points_label = QLabel("\u2014")
        stats_form.addRow("Mean Wind Speed:", self._mean_speed_label)
        stats_form.addRow("Mean Power Density:", self._mean_pd_label)
        stats_form.addRow("Data Points:", self._n_points_label)
        stats_group.setLayout(stats_form)
        layout.addWidget(stats_group)

        # --- Matplotlib: wind speed map + wind rose ---
        self._figure = Figure(figsize=(10, 4), dpi=100)
        self._figure.patch.set_facecolor('#2d2d30')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(250)
        layout.addWidget(self._canvas)

        layout.addStretch()

        # Init UI state
        self._on_source_changed(self._source_combo.currentIndex())

    def _on_source_changed(self, index):
        is_gwa = (index == 0)
        self._height_spin.setEnabled(is_gwa)
        self._grid_spin.setEnabled(is_gwa)
        self._download_btn.setEnabled(is_gwa)
        self._mast_btn.setEnabled(not is_gwa)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _start_download(self):
        data = self._main_window.get_project_data()
        bbox = data.get('bbox')
        if not bbox:
            QMessageBox.warning(self, "No Boundary",
                                "Please create a project with a boundary first.")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "A download is already in progress.")
            return

        height = self._height_spin.value()
        grid_spacing = self._grid_spin.value()
        source = 'gwa'

        self._log.clear()
        self._progress.setValue(0)
        self._download_btn.setEnabled(False)
        self._main_window.update_status("Downloading GWA wind data\u2026")
        self._main_window.show_progress(True)

        self._worker = _WindWorker(
            bbox, source, height, grid_spacing, './gwa_data', parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.start()

    def _import_mast(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Mast Data", "",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            from src.utils.data_utils import load_mast_data
            mast_data = load_mast_data(path)
            data = self._main_window.get_project_data()
            data['mast_data'] = mast_data
            data['wind_data'] = {}
            self._mast_file_label.setText(path)
            self._main_window.update_status(f"Mast data imported from {path}")
            self._log.append(f"Mast data imported: format={mast_data.get('format', '?')}")
            QMessageBox.information(self, "Imported",
                                    f"Mast data loaded successfully.\nFormat: {mast_data.get('format')}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import mast data:\n{e}")

    def _on_progress(self, current, total, msg):
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._main_window.update_progress(current, total)

    def _on_status(self, msg):
        self._log.append(msg)

    def _on_finished_ok(self, result):
        data = self._main_window.get_project_data()
        data['wind_data'] = result
        self._mean_speed_label.setText(f"{result.get('mean_speed', 0):.1f} m/s")
        self._mean_pd_label.setText(f"{result.get('mean_power_density', 0):.0f} W/m\u00b2")
        self._n_points_label.setText(str(result.get('num_points', 0)))
        self._download_btn.setEnabled(True)
        self._main_window.update_status("GWA wind data download complete.")
        self._main_window.show_progress(False)
        self._preview_wind(result)

    def _on_finished_err(self, msg):
        self._log.append(f"ERROR: {msg}")
        self._download_btn.setEnabled(True)
        self._main_window.update_status("Wind data download failed.")
        self._main_window.show_progress(False)
        QMessageBox.critical(self, "Error", f"Wind data download failed:\n{msg}")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _preview_wind(self, result):
        points = result.get('points', [])
        if not points:
            return

        self._figure.clear()

        # --- Left: wind speed map ---
        ax1 = self._figure.add_subplot(121)
        lats = [p['lat'] for p in points]
        lons = [p['lon'] for p in points]
        speeds = [p['mean_wind_speed'] for p in points]

        sc = ax1.scatter(lons, lats, c=speeds, cmap='RdYlBu_r', s=10, edgecolors='none')
        ax1.set_title("Mean Wind Speed Map", color='white', fontsize=10)
        ax1.set_xlabel("Longitude", color='white')
        ax1.set_ylabel("Latitude", color='white')
        ax1.tick_params(colors='white')
        cb = self._figure.colorbar(sc, ax=ax1, shrink=0.8, label='m/s')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')

        # --- Right: wind rose ---
        ax2 = self._figure.add_subplot(122, projection='polar')
        first_point = points[0]
        sectors = first_point.get('sectors', [])
        if sectors:
            directions = [np.radians(s['direction'] + 180) for s in sectors]  # meteorological -> math
            frequencies = [s.get('frequency', 0) for s in sectors]
            speeds_sec = [s.get('mean_speed', 0) for s in sectors]

            # Normalise frequency for display
            total_f = sum(frequencies) if sum(frequencies) > 0 else 1
            freq_norm = [f / total_f for f in frequencies]

            bar_width = 2 * np.pi / len(sectors)
            bars = ax2.bar(directions, freq_norm, width=bar_width, bottom=0,
                           color=speeds_sec, cmap='RdYlBu_r', edgecolor='#333', linewidth=0.5)
            ax2.set_title("Wind Rose", color='white', fontsize=10, pad=15)
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            ax2.tick_params(colors='white')
            ax2.set_facecolor('#2d2d30')
        else:
            ax2.set_title("Wind Rose (no sector data)", color='white', fontsize=10)
            ax2.set_facecolor('#2d2d30')

        self._figure.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_from_project(self):
        data = self._main_window.get_project_data()
        bbox = data.get('bbox')
        if bbox:
            self._bbox_label.setText(
                f"Bounding box: [{bbox[0]:.4f}, {bbox[1]:.4f}, "
                f"{bbox[2]:.4f}, {bbox[3]:.4f}]")
        else:
            self._bbox_label.setText("No boundary defined.")

        wind_data = data.get('wind_data', {})
        if wind_data:
            self._mean_speed_label.setText(f"{wind_data.get('mean_speed', 0):.1f} m/s")
            self._mean_pd_label.setText(f"{wind_data.get('mean_power_density', 0):.0f} W/m\u00b2")
            self._n_points_label.setText(str(wind_data.get('num_points', 0)))
            self._preview_wind(wind_data)
