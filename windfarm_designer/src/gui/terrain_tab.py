"""
WindFarm Designer Pro - Terrain Data Tab.
Downloads SRTM1 terrain data, creates mosaics, and shows a preview
with optional OpenStreetMap background overlay (via contextily).

Improvements over previous version:
  - Elevation statistics (min, max, mean, std) displayed in Results group
  - Boundary polygon always shown on terrain preview
  - Wind farm area extent rectangle shown on map
  - Better alpha blending (0.3 for OSM basemap under DEM)
  - Export terrain visualization as PNG button
  - Export to .map button (WAsP map format via roughness_downloader)
  - Consistent dark theme styling with white text
  - Better DEM colorbar with proper tick formatting
  - Graceful degradation when optional packages are missing
"""

import logging
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QProgressBar,
    QTextEdit, QCheckBox, QGroupBox, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

class _TerrainWorker(QThread):
    """Runs SRTM download in a background thread."""
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, bbox, output_dir, source, n_threads, compute_slope, parent=None):
        super().__init__(parent)
        self._bbox = bbox
        self._output_dir = output_dir
        self._source = source
        self._n_threads = n_threads
        self._compute_slope = compute_slope

    def run(self):
        try:
            from src.core.srtm_downloader import SRTMDownloader
            dl = SRTMDownloader(output_dir=self._output_dir, buffer_km=0)
            dl.set_progress_callback(lambda c, t, m: self.progress.emit(c, t, m))
            dl.set_status_callback(lambda m: self.status.emit(m))
            result = dl.download_and_process(
                self._bbox, source=self._source, max_threads=self._n_threads)
            if not result:
                self.finished_err.emit("Download produced no results.")
            else:
                self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


# ------------------------------------------------------------------
# Tab widget
# ------------------------------------------------------------------

class TerrainTab(QWidget):
    """Second workflow tab: download SRTM1 terrain data."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Info ---
        self._bbox_label = QLabel("No boundary defined. Create a project first.")
        self._bbox_label.setWordWrap(True)
        layout.addWidget(self._bbox_label)

        # --- Settings Group ---
        settings_group = QGroupBox("Download Settings")
        settings_form = QFormLayout()

        self._source_combo = QComboBox()
        self._source_combo.addItems(["AWS OpenTopo", "NASA EarthData"])
        settings_form.addRow("Data Source:", self._source_combo)

        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(1, 8)
        self._threads_spin.setValue(4)
        settings_form.addRow("Download Threads:", self._threads_spin)

        self._slope_check = QCheckBox("Compute slope & aspect")
        self._slope_check.setChecked(True)
        settings_form.addRow(self._slope_check)

        self._buffer_check = QCheckBox("Download with buffer")
        self._buffer_check.setChecked(True)
        self._buffer_display = QLabel("20 km (from project settings)")
        settings_form.addRow(self._buffer_check)
        settings_form.addRow("", self._buffer_display)

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # --- Action buttons row ---
        action_row = QHBoxLayout()

        self._download_btn = QPushButton("\U0001F4E5  Download SRTM1 Terrain Data")
        self._download_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        self._download_btn.clicked.connect(self._start_download)
        action_row.addWidget(self._download_btn)

        self._load_dem_btn = QPushButton("\U0001F4C2  Load Existing DEM\u2026")
        self._load_dem_btn.clicked.connect(self._load_existing_dem)
        action_row.addWidget(self._load_dem_btn)

        layout.addLayout(action_row)

        # --- Export buttons row ---
        export_row = QHBoxLayout()

        self._export_png_btn = QPushButton("\U0001F5BC  Export Preview as PNG")
        self._export_png_btn.clicked.connect(self._export_png)
        self._export_png_btn.setEnabled(False)
        export_row.addWidget(self._export_png_btn)

        self._export_map_btn = QPushButton("\U0001F4C4  Export to .map (WAsP)")
        self._export_map_btn.clicked.connect(self._export_to_map)
        self._export_map_btn.setEnabled(False)
        self._export_map_btn.setToolTip(
            "Convert terrain and roughness data to WAsP .map format.\n"
            "Requires roughness data to be downloaded first.")
        export_row.addWidget(self._export_map_btn)

        layout.addLayout(export_row)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        layout.addWidget(self._log)

        # --- Results labels ---
        info_group = QGroupBox("Results")
        info_form = QFormLayout()
        self._tiles_label = QLabel("\u2014")
        self._mosaic_label = QLabel("\u2014")
        self._slope_label = QLabel("\u2014")
        info_form.addRow("Tiles Downloaded:", self._tiles_label)
        info_form.addRow("Mosaic File:", self._mosaic_label)
        info_form.addRow("Slope/Aspect:", self._slope_label)
        info_group.setLayout(info_form)
        layout.addWidget(info_group)

        # --- Elevation Statistics ---
        stats_group = QGroupBox("Elevation Statistics")
        stats_form = QFormLayout()
        self._elev_min_label = QLabel("\u2014")
        self._elev_max_label = QLabel("\u2014")
        self._elev_mean_label = QLabel("\u2014")
        self._elev_std_label = QLabel("\u2014")
        stats_form.addRow("Min Elevation:", self._elev_min_label)
        stats_form.addRow("Max Elevation:", self._elev_max_label)
        stats_form.addRow("Mean Elevation:", self._elev_mean_label)
        stats_form.addRow("Std Deviation:", self._elev_std_label)
        stats_group.setLayout(stats_form)
        layout.addWidget(stats_group)

        # --- Matplotlib preview ---
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
            QMessageBox.information(self, "Busy", "A download is already in progress.")
            return

        source = 'aws_opentopo' if self._source_combo.currentIndex() == 0 else 'earthdata'
        n_threads = self._threads_spin.value()
        compute_slope = self._slope_check.isChecked()
        buffer_km = data.get('buffer_km', 20.0)
        output_dir = './srtm_data'

        self._log.clear()
        self._progress.setValue(0)
        self._download_btn.setEnabled(False)
        self._main_window.update_status("Downloading terrain data\u2026")
        self._main_window.show_progress(True)

        if not self._buffer_check.isChecked():
            buffer_km = 0.0

        self._worker = _TerrainWorker(
            bbox, output_dir, source, n_threads, compute_slope, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.start()

    def _load_existing_dem(self):
        """Let the user load a previously downloaded GeoTIFF DEM file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Existing DEM (GeoTIFF)", "",
            "GeoTIFF Files (*.tif *.tiff);;All Files (*)")
        if not path:
            return

        self._log.clear()
        self._log.append(f"Loading DEM from: {path}")
        self._main_window.update_status(f"Loading DEM from {path}")

        # Store as terrain result in project data
        result = {
            'mosaic_path': path,
            'tiles_downloaded': 0,
            'tiles_failed': 0,
            'slope_path': None,
            'aspect_path': None,
        }

        data = self._main_window.get_project_data()
        data['terrain'] = result

        self._tiles_label.setText("Loaded from existing file")
        self._mosaic_label.setText(path)
        self._slope_label.setText("N/A (no slope computed)")

        self._main_window.update_status(f"DEM loaded from {path}")
        self._preview_terrain(result)
        self._compute_elevation_stats(result)
        self._export_png_btn.setEnabled(True)
        self._update_export_map_btn()

    def _on_progress(self, current, total, msg):
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._main_window.update_progress(current, total)

    def _on_status(self, msg):
        self._log.append(msg)
        self._main_window.update_status(msg)

    def _on_finished_ok(self, result):
        data = self._main_window.get_project_data()
        data['terrain'] = result
        self._tiles_label.setText(
            f"{result.get('tiles_downloaded', 0)} downloaded, "
            f"{result.get('tiles_failed', 0)} failed")
        self._mosaic_label.setText(result.get('mosaic_path', '\u2014') or '\u2014')
        self._slope_label.setText(
            f"Slope: {result.get('slope_path', 'N/A')}\n"
            f"Aspect: {result.get('aspect_path', 'N/A')}")

        self._download_btn.setEnabled(True)
        self._main_window.update_status("Terrain download complete.")
        self._main_window.show_progress(False)
        self._preview_terrain(result)
        self._compute_elevation_stats(result)
        self._export_png_btn.setEnabled(True)
        self._update_export_map_btn()

    def _on_finished_err(self, msg):
        self._log.append(f"ERROR: {msg}")
        self._download_btn.setEnabled(True)
        self._main_window.update_status("Terrain download failed.")
        self._main_window.show_progress(False)
        QMessageBox.critical(self, "Error", f"Terrain download failed:\n{msg}")

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _export_png(self):
        """Export the current terrain preview as a PNG image."""
        if not self._figure.axes:
            QMessageBox.information(self, "No Preview",
                                    "No terrain preview to export. Download or load terrain data first.")
            return
        default_name = self._main_window.get_project_data().get('project_name', 'terrain')
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Terrain Preview as PNG",
            f"{default_name}_terrain.png",
            "PNG Files (*.png);;All Files (*)")
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=150, bbox_inches='tight',
                                 facecolor=self._figure.get_facecolor(), edgecolor='none')
            self._main_window.update_status(f"Terrain preview exported to {path}")
            self._log.append(f"Preview exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export PNG:\n{e}")
            logger.exception("PNG export failed")

    def _export_to_map(self):
        """Export terrain and roughness data to WAsP .map format."""
        data = self._main_window.get_project_data()
        roughness = data.get('roughness', {})
        roughness_path = roughness.get('roughness_path')

        if not roughness_path or not os.path.isfile(roughness_path):
            QMessageBox.warning(
                self, "No Roughness Data",
                "Roughness data is required for WAsP .map export.\n\n"
                "Please download roughness data first in the Roughness Data tab.")
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

    def _update_export_map_btn(self):
        """Enable/disable the .map export button based on data availability."""
        data = self._main_window.get_project_data()
        roughness_path = data.get('roughness', {}).get('roughness_path')
        terrain_path = data.get('terrain', {}).get('mosaic_path')
        self._export_map_btn.setEnabled(
            bool(roughness_path and terrain_path and
                 os.path.isfile(str(roughness_path))))

    # ------------------------------------------------------------------
    # Elevation statistics
    # ------------------------------------------------------------------

    def _compute_elevation_stats(self, result):
        """Compute and display elevation statistics from the DEM."""
        mosaic_path = result.get('mosaic_path')
        if not mosaic_path or not os.path.isfile(str(mosaic_path)):
            self._elev_min_label.setText("\u2014")
            self._elev_max_label.setText("\u2014")
            self._elev_mean_label.setText("\u2014")
            self._elev_std_label.setText("\u2014")
            return

        try:
            import rasterio
            with rasterio.open(mosaic_path) as src:
                dem = src.read(1)
                nodata = src.nodata if src.nodata else -32768

            valid = dem[dem != nodata]
            valid = valid[~np.isnan(valid.astype(float))]
            if len(valid) > 0:
                self._elev_min_label.setText(f"{float(np.min(valid)):.1f} m")
                self._elev_max_label.setText(f"{float(np.max(valid)):.1f} m")
                self._elev_mean_label.setText(f"{float(np.mean(valid)):.1f} m")
                self._elev_std_label.setText(f"{float(np.std(valid)):.1f} m")
            else:
                self._elev_min_label.setText("N/A")
                self._elev_max_label.setText("N/A")
                self._elev_mean_label.setText("N/A")
                self._elev_std_label.setText("N/A")
        except ImportError:
            self._log.append("Warning: rasterio not available for elevation statistics.")
            self._elev_min_label.setText("N/A")
            self._elev_max_label.setText("N/A")
            self._elev_mean_label.setText("N/A")
            self._elev_std_label.setText("N/A")
        except Exception as e:
            self._log.append(f"Elevation statistics error: {e}")
            self._elev_min_label.setText("Error")
            self._elev_max_label.setText("Error")
            self._elev_mean_label.setText("Error")
            self._elev_std_label.setText("Error")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _get_boundary_coords(self):
        """Return the boundary polygon from project data, or None."""
        boundary = self._main_window.get_project_data().get('boundary')
        if boundary and len(boundary) >= 3:
            return boundary
        return None

    def _preview_terrain(self, result):
        mosaic_path = result.get('mosaic_path')
        if not mosaic_path or not os.path.isfile(str(mosaic_path)):
            self._log.append("Warning: mosaic file not found for preview.")
            return

        # Try the enhanced preview with contextily OSM basemap
        try:
            self._preview_terrain_with_osm(mosaic_path)
            return
        except ImportError:
            self._log.append(
                "Info: contextily/rasterio not fully available. "
                "Using basic preview.")
        except Exception as e:
            self._log.append(f"Info: OSM basemap preview failed ({e}). "
                             "Using basic preview.")

        # Fallback: basic preview without contextily
        self._preview_terrain_basic(mosaic_path)

    def _preview_terrain_with_osm(self, mosaic_path):
        """Preview with DEM + OSM basemap overlay, boundary polygon, and area extent."""
        import rasterio
        import contextily as ctx

        with rasterio.open(mosaic_path) as src:
            dem = src.read(1)
            nodata = src.nodata if src.nodata else -32768
            bounds = src.bounds  # (left, bottom, right, top) in CRS coordinates
            data_crs = src.crs

        dem_masked = np.ma.masked_where(dem == nodata, dem)

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # Plot DEM with terrain colormap
        im = ax.imshow(
            dem_masked, cmap='terrain', aspect='auto',
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin='upper')

        # Add OSM basemap under DEM (alpha=0.3 for subtle background)
        try:
            ctx.add_basemap(
                ax, crs=data_crs,
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom='auto', alpha=0.3)
        except Exception:
            pass  # If contextily basemap fails, just show DEM without it

        # Show boundary polygon
        boundary = self._get_boundary_coords()
        if boundary:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.plot(bnd_lons, bnd_lats, color='#00FFFF', linewidth=2.0,
                    linestyle='--', label='Boundary', zorder=10)

        # Show wind farm area extent (bbox)
        bbox = self._main_window.get_project_data().get('bbox')
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            ax.plot([min_lon, max_lon, max_lon, min_lon, min_lon],
                    [min_lat, min_lat, max_lat, max_lat, min_lat],
                    color='#FFD700', linewidth=1.5, linestyle=':',
                    label='Farm Extent', zorder=9)

        # Colorbar for elevation with proper tick formatting
        ax.set_title("Terrain Elevation Map (with OSM base)", color='white',
                      fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')
        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='Elevation (m)')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
        # Format tick labels to integers for cleaner display
        cb.ax.yaxis.set_major_formatter(
            __import__('matplotlib.ticker', fromlist=['FormatStrFormatter']).FormatStrFormatter('%d'))

        ax.legend(loc='upper right', fontsize=8, facecolor='#3d3d40',
                  edgecolor='#666', labelcolor='white')
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    def _preview_terrain_basic(self, mosaic_path):
        """Basic preview without contextily (fallback)."""
        import rasterio

        with rasterio.open(mosaic_path) as src:
            dem = src.read(1)
            nodata = src.nodata if src.nodata else -32768
            transform = src.transform
        dem_masked = np.ma.masked_where(dem == nodata, dem)

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        im = ax.imshow(
            dem_masked, cmap='terrain', aspect='auto',
            extent=[transform[2], transform[2] + transform[0] * dem.shape[1],
                    transform[5] + transform[4] * dem.shape[0], transform[5]])

        # Show boundary polygon on basic preview too
        boundary = self._get_boundary_coords()
        if boundary:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.plot(bnd_lons, bnd_lats, color='#00FFFF', linewidth=2.0,
                    linestyle='--', label='Boundary', zorder=10)

        ax.set_title("Terrain Mosaic Preview", color='white', fontsize=11,
                      fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')
        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='Elevation (m)')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
        cb.ax.yaxis.set_major_formatter(
            __import__('matplotlib.ticker', fromlist=['FormatStrFormatter']).FormatStrFormatter('%d'))

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
            self._bbox_label.setText("No boundary defined. Create a project first.")

        buffer_km = data.get('buffer_km', 20.0)
        self._buffer_display.setText(f"{buffer_km} km (from project settings)")

        terrain = data.get('terrain', {})
        if terrain:
            self._tiles_label.setText(
                f"{terrain.get('tiles_downloaded', 0)} downloaded, "
                f"{terrain.get('tiles_failed', 0)} failed")
            self._mosaic_label.setText(
                terrain.get('mosaic_path', '\u2014') or '\u2014')
            if terrain.get('mosaic_path'):
                self._preview_terrain(terrain)
                self._compute_elevation_stats(terrain)
                self._export_png_btn.setEnabled(True)

        self._update_export_map_btn()
