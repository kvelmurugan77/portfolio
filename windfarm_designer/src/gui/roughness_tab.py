"""
WindFarm Designer Pro - Roughness Data Tab.
Downloads land-cover-based roughness data and shows a preview map
with optional OpenStreetMap background overlay (via contextily).
Includes land cover class statistics and a roughness colorbar legend.

Improvements over previous version:
  - "Export to .map" button (WAsP map format)
  - Boundary polygon always shown on roughness preview
  - Better roughness class color legend with percentages
  - Improved alpha blending (0.4 for OSM basemap under roughness)
  - Roughness statistics by land cover class (mean z0 per class)
  - Better colorbar with z0 class labels and tick formatting
  - Consistent dark theme styling with white text
  - Graceful degradation when optional packages are missing
"""

import logging
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Roughness class definitions for the legend and colorbar
# (name, z0_range_low, z0_range_high, color)
_ROUGHNESS_CLASSES = [
    ("Water",            0.0000, 0.0001, "#2166AC"),
    ("Bare Soil",        0.0001, 0.0010, "#D1B48C"),
    ("Grassland",        0.0010, 0.0300, "#7FBF7F"),
    ("Cropland",         0.0300, 0.1000, "#FEE08B"),
    ("Shrubland/Forest", 0.1000, 0.5000, "#4DAF4A"),
    ("Dense Forest",     0.5000, 1.5000, "#1B7837"),
    ("Urban",            1.5000, 5.0000, "#762A83"),
]


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

class _RoughnessWorker(QThread):
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, bbox, source_key, output_dir, parent=None):
        super().__init__(parent)
        self._bbox = bbox
        self._source_key = source_key
        self._output_dir = output_dir

    def run(self):
        try:
            from src.core.roughness_downloader import RoughnessDownloader
            dl = RoughnessDownloader(output_dir=self._output_dir)
            dl.set_progress_callback(lambda c, t, m: self.progress.emit(c, t, m))
            dl.set_status_callback(lambda m: self.status.emit(m))
            result = dl.download_and_process(
                self._bbox, preferred_source=self._source_key)
            if not result:
                self.finished_err.emit(
                    "Roughness download produced no results.")
            else:
                self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


# ------------------------------------------------------------------
# Tab widget
# ------------------------------------------------------------------

class RoughnessTab(QWidget):
    """Third workflow tab: download and preview roughness data."""

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
        settings_group = QGroupBox("Download Settings")
        settings_form = QFormLayout()

        self._source_combo = QComboBox()
        self._source_combo.addItems([
            "Global Land Cover (CCI 300m)",
            "Copernicus CCI",
            "ESA WorldCover",
            "OpenStreetMap",
        ])
        self._source_combo.setCurrentIndex(0)
        settings_form.addRow("Data Source:", self._source_combo)

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # --- Action buttons row ---
        action_row = QHBoxLayout()

        self._download_btn = QPushButton(
            "\U0001F4E5  Download Roughness Data")
        self._download_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        self._download_btn.clicked.connect(self._start_download)
        action_row.addWidget(self._download_btn)

        self._export_map_btn = QPushButton(
            "\U0001F4C4  Export to .map (WAsP)")
        self._export_map_btn.clicked.connect(self._export_to_map)
        self._export_map_btn.setEnabled(False)
        self._export_map_btn.setToolTip(
            "Convert roughness GeoTIFF to WAsP .map format.")
        action_row.addWidget(self._export_map_btn)

        layout.addLayout(action_row)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        layout.addWidget(self._log)

        # --- Statistics (basic z0 stats + land cover class breakdown) ---
        stats_group = QGroupBox("Roughness Statistics")
        stats_layout = QVBoxLayout()

        z0_form = QFormLayout()
        self._z0_min_label = QLabel("\u2014")
        self._z0_max_label = QLabel("\u2014")
        self._z0_mean_label = QLabel("\u2014")
        self._z0_std_label = QLabel("\u2014")
        self._source_used_label = QLabel("\u2014")
        z0_form.addRow("Min z\u2080:", self._z0_min_label)
        z0_form.addRow("Max z\u2080:", self._z0_max_label)
        z0_form.addRow("Mean z\u2080:", self._z0_mean_label)
        z0_form.addRow("Std Dev z\u2080:", self._z0_std_label)
        z0_form.addRow("Source Used:", self._source_used_label)
        stats_layout.addLayout(z0_form)

        # Land cover class breakdown table
        class_group = QGroupBox("Land Cover Class Distribution")
        class_layout = QFormLayout()
        self._class_labels = {}
        for class_name, _, _, color in _ROUGHNESS_CLASSES:
            lbl = QLabel("\u2014")
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            class_layout.addRow(f"{class_name}:", lbl)
            self._class_labels[class_name] = lbl
        class_group.setLayout(class_layout)
        stats_layout.addWidget(class_group)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # --- Preview ---
        self._figure = Figure(figsize=(7, 4), dpi=100)
        self._figure.patch.set_facecolor('#2d2d30')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(220)
        layout.addWidget(self._canvas)

        layout.addStretch()

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
            QMessageBox.information(self, "Busy",
                                    "A download is already in progress.")
            return

        # Map combo index to source key
        source_map = {
            0: 'cci',       # Global Land Cover (CCI 300m) - primary
            1: 'cci',       # Copernicus CCI
            2: 'worldcover',
            3: 'osm',
        }
        source_key = source_map.get(
            self._source_combo.currentIndex(), 'auto')

        self._log.clear()
        self._progress.setValue(0)
        self._download_btn.setEnabled(False)
        self._main_window.update_status("Downloading roughness data\u2026")
        self._main_window.show_progress(True)

        self._worker = _RoughnessWorker(
            bbox, source_key, './roughness_data', parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.start()

    def _on_progress(self, current, total, msg):
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._main_window.update_progress(current, total)

    def _on_status(self, msg):
        self._log.append(msg)

    def _on_finished_ok(self, result):
        data = self._main_window.get_project_data()
        data['roughness'] = result
        self._source_used_label.setText(result.get('source', '\u2014'))
        self._download_btn.setEnabled(True)
        self._main_window.update_status("Roughness data download complete.")
        self._main_window.show_progress(False)
        self._preview_roughness(result)
        self._compute_statistics(result)
        self._compute_class_distribution(result)

        # Enable export button if file exists
        roughness_path = result.get('roughness_path')
        self._export_map_btn.setEnabled(
            bool(roughness_path and os.path.isfile(str(roughness_path))))

    def _on_finished_err(self, msg):
        self._log.append(f"ERROR: {msg}")
        self._download_btn.setEnabled(True)
        self._main_window.update_status("Roughness download failed.")
        self._main_window.show_progress(False)
        QMessageBox.critical(self, "Error",
                             f"Roughness download failed:\n{msg}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_to_map(self):
        """Export roughness data to WAsP .map format."""
        data = self._main_window.get_project_data()
        roughness = data.get('roughness', {})
        roughness_path = roughness.get('roughness_path')

        if not roughness_path or not os.path.isfile(str(roughness_path)):
            QMessageBox.warning(
                self, "No Roughness Data",
                "No roughness data available for export.\n"
                "Please download roughness data first.")
            return

        default_name = data.get('project_name', 'windfarm')
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to WAsP .map Format",
            f"{default_name}_roughness.map",
            "WAsP Map Files (*.map);;All Files (*)")
        if not path:
            return

        self._main_window.update_status("Exporting to WAsP .map format\u2026")
        self._log.append("Converting roughness data to WAsP .map format...")

        try:
            from src.core.roughness_downloader import RoughnessDownloader
            dl = RoughnessDownloader()
            dl.set_status_callback(lambda m: self._log.append(f"  {m}"))

            result_path = dl.geotiff_to_map_file(roughness_path, output_path=path)
            if result_path:
                self._main_window.update_status(f"WAsP .map file exported to {result_path}")
                self._log.append(f"WAsP .map export complete: {result_path}")
                QMessageBox.information(
                    self, "Export Complete",
                    f"WAsP .map file saved to:\n{result_path}\n\n"
                    f"A companion text-format .map file may also have been generated.")
            else:
                QMessageBox.warning(
                    self, "Export Failed",
                    "WAsP .map export returned no result. Check the log for details.")
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "The roughness_downloader module could not be imported.\n"
                "Make sure all project modules are available.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export .map file:\n{e}")
            logger.exception(".map export failed")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self, result):
        roughness_path = result.get('roughness_path')
        if not roughness_path:
            self._z0_min_label.setText("\u2014")
            self._z0_max_label.setText("\u2014")
            self._z0_mean_label.setText("\u2014")
            self._z0_std_label.setText("\u2014")
            return
        try:
            import rasterio
            with rasterio.open(roughness_path) as src:
                z0 = src.read(1)
                nodata = src.nodata if src.nodata else -9999
            valid = z0[z0 != nodata]
            valid = valid[valid > 0]
            if len(valid) > 0:
                self._z0_min_label.setText(f"{float(np.min(valid)):.4f} m")
                self._z0_max_label.setText(f"{float(np.max(valid)):.4f} m")
                self._z0_mean_label.setText(f"{float(np.mean(valid)):.4f} m")
                self._z0_std_label.setText(f"{float(np.std(valid)):.4f} m")
            else:
                self._z0_min_label.setText("\u2014")
                self._z0_max_label.setText("\u2014")
                self._z0_mean_label.setText("\u2014")
                self._z0_std_label.setText("\u2014")
        except ImportError:
            self._log.append("Warning: rasterio not available for statistics.")
        except Exception as e:
            self._log.append(f"Statistics error: {e}")

    def _compute_class_distribution(self, result):
        """Compute percentage of each roughness class in the area."""
        roughness_path = result.get('roughness_path')
        if not roughness_path:
            for lbl in self._class_labels.values():
                lbl.setText("\u2014")
            return
        try:
            import rasterio
            with rasterio.open(roughness_path) as src:
                z0 = src.read(1)
                nodata = src.nodata if src.nodata else -9999
            valid = z0[z0 != nodata]
            valid = valid[valid >= 0]
            total = len(valid)
            if total == 0:
                for lbl in self._class_labels.values():
                    lbl.setText("\u2014")
                return

            for class_name, z0_lo, z0_hi, _ in _ROUGHNESS_CLASSES:
                if z0_hi >= 5.0:
                    # Urban and above: catch all higher values
                    count = np.sum((valid >= z0_lo))
                else:
                    count = np.sum((valid >= z0_lo) & (valid < z0_hi))
                pct = count / total * 100.0
                # Also compute mean z0 for pixels in this class
                if z0_hi >= 5.0:
                    class_pixels = valid[valid >= z0_lo]
                else:
                    class_pixels = valid[(valid >= z0_lo) & (valid < z0_hi)]
                mean_z0 = float(np.mean(class_pixels)) if len(class_pixels) > 0 else 0
                self._class_labels[class_name].setText(
                    f"{pct:.1f}% ({count} px, \u0305z\u2080={mean_z0:.3f})")
        except ImportError:
            self._log.append("Warning: rasterio not available for class stats.")
        except Exception as e:
            self._log.append(f"Class distribution error: {e}")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _preview_roughness(self, result):
        roughness_path = result.get('roughness_path')
        if not roughness_path:
            return

        # Try enhanced preview with contextily OSM basemap
        try:
            self._preview_roughness_with_osm(roughness_path)
            return
        except ImportError:
            self._log.append(
                "Info: contextily/rasterio not fully available. "
                "Using basic preview.")
        except Exception as e:
            self._log.append(f"Info: OSM basemap preview failed ({e}). "
                             "Using basic preview.")

        # Fallback: basic preview
        self._preview_roughness_basic(roughness_path)

    def _preview_roughness_with_osm(self, roughness_path):
        """Preview roughness with OSM basemap, class colorbar, and boundary."""
        import rasterio
        import contextily as ctx

        with rasterio.open(roughness_path) as src:
            z0 = src.read(1)
            nodata = src.nodata if src.nodata else -9999
            bounds = src.bounds
            data_crs = src.crs

        z0_masked = np.ma.masked_where(z0 <= 0, z0)

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # Custom discrete colormap based on roughness classes
        colors = [c[3] for c in _ROUGHNESS_CLASSES]
        bounds_vals = [c[1] for c in _ROUGHNESS_CLASSES] + [_ROUGHNESS_CLASSES[-1][2]]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds_vals, cmap.N)

        im = ax.imshow(
            z0_masked, cmap=cmap, norm=norm, aspect='auto',
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin='upper')

        # Add OSM basemap (alpha=0.4 for good blending)
        try:
            ctx.add_basemap(
                ax, crs=data_crs,
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom='auto', alpha=0.4)
        except Exception:
            pass  # If contextily basemap fails, just show roughness without it

        # Show boundary polygon
        boundary = self._main_window.get_project_data().get('boundary')
        if boundary and len(boundary) >= 3:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.plot(bnd_lons, bnd_lats, color='#00FFFF', linewidth=2.0,
                    linestyle='--', label='Boundary', zorder=10)

        # Colorbar with class labels
        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='z\u2080 (m)')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        # Set tick labels to class names at midpoints
        tick_positions = []
        tick_labels = []
        for i, (name, lo, hi, _) in enumerate(_ROUGHNESS_CLASSES):
            tick_positions.append((lo + hi) / 2.0)
            tick_labels.append(f"{name}\n({lo:.4f}\u2013{hi:.2f})")
        cb.set_ticks(tick_positions)
        cb.set_ticklabels(tick_labels)
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
            label.set_fontsize(6.5)

        ax.set_title(
            "Surface Roughness z\u2080 (with OSM base & class legend)",
            color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8, facecolor='#3d3d40',
                  edgecolor='#666', labelcolor='white')
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    def _preview_roughness_basic(self, roughness_path):
        """Basic roughness preview without contextily (fallback)."""
        import rasterio

        with rasterio.open(roughness_path) as src:
            z0 = src.read(1)
            nodata = src.nodata if src.nodata else -9999
            transform = src.transform
        z0_masked = np.ma.masked_where(z0 <= 0, z0)

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # Use discrete colormap even in basic mode for consistency
        colors = [c[3] for c in _ROUGHNESS_CLASSES]
        bounds_vals = [c[1] for c in _ROUGHNESS_CLASSES] + [_ROUGHNESS_CLASSES[-1][2]]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds_vals, cmap.N)

        im = ax.imshow(
            z0_masked, cmap=cmap, norm=norm, aspect='auto',
            extent=[transform[2], transform[2] + transform[0] * z0.shape[1],
                    transform[5] + transform[4] * z0.shape[0], transform[5]])

        # Show boundary on basic preview too
        boundary = self._main_window.get_project_data().get('boundary')
        if boundary and len(boundary) >= 3:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.plot(bnd_lons, bnd_lats, color='#00FFFF', linewidth=2.0,
                    linestyle='--', label='Boundary', zorder=10)

        ax.set_title("Surface Roughness z\u2080 Preview",
                      color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')
        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='z\u2080 (m)')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
        ax.legend(loc='upper right', fontsize=8, facecolor='#3d3d40',
                  edgecolor='#666', labelcolor='white')
        ax.set_facecolor('#2d2d30')
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

        roughness = data.get('roughness', {})
        if roughness:
            self._source_used_label.setText(roughness.get('source', '\u2014'))
            self._compute_statistics(roughness)
            self._compute_class_distribution(roughness)
            if roughness.get('roughness_path'):
                self._preview_roughness(roughness)

            roughness_path = roughness.get('roughness_path')
            self._export_map_btn.setEnabled(
                bool(roughness_path and os.path.isfile(str(roughness_path))))
