"""
WindFarm Designer Pro - Wind Flow Model Tab.
Runs the simplified wind flow model to compute terrain and roughness
corrections at each turbine position, and displays speed-up contours.

Improvements over previous version:
  - Speed-up preview shows geographic coordinates (lat/lon) instead of grid indices
  - OSM background on speed-up contour plot
  - Turbine positions overlaid on speed-up map
  - Boundary polygon shown on speed-up map
  - Proper georeferencing using DEM transform
  - Better colorbar with speed-up ratio labels
  - Sector selection dropdown to view specific sector speed-up
  - Consistent dark theme styling with white text
  - Graceful degradation when optional packages are missing
"""

import logging
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QLabel, QSpinBox, QCheckBox, QPushButton, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

class _FlowWorker(QThread):
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, positions, wind_data, terrain, roughness,
                 use_terrain, use_roughness, n_sectors, parent=None):
        super().__init__(parent)
        self._positions = positions
        self._wind_data = wind_data
        self._terrain = terrain
        self._roughness = roughness
        self._use_terrain = use_terrain
        self._use_roughness = use_roughness
        self._n_sectors = n_sectors

    def run(self):
        try:
            from src.core.wind_flow_model import WindFlowModel, FlowModelConfig
            from src.core.layout_optimizer import WTGPosition

            config = FlowModelConfig(
                n_sectors=self._n_sectors,
                use_terrain_correction=self._use_terrain,
                use_roughness_correction=self._use_roughness,
            )

            model = WindFlowModel(config)
            model.set_progress_callback(lambda c, t, m: self.progress.emit(c, t, m))
            model.set_status_callback(lambda m: self.status.emit(m))

            # Load DEM
            mosaic_path = self._terrain.get('mosaic_path')
            if mosaic_path and self._use_terrain:
                model.load_dem(mosaic_path)
                slope_path = self._terrain.get('slope_path')
                aspect_path = self._terrain.get('aspect_path')
                if slope_path:
                    model.load_slope_aspect(slope_path=slope_path, aspect_path=aspect_path)

            # Load roughness
            roughness_path = self._roughness.get('roughness_path')
            if roughness_path and self._use_roughness:
                model.load_roughness(roughness_path)

            result = model.run_flow_model(self._positions, self._wind_data)

            # Store the DEM transform for georeferencing the preview
            if model.dem_transform is not None:
                result['_dem_transform'] = model.dem_transform
            if model.dem is not None:
                result['_dem_bounds'] = {
                    'left': float(model.dem_transform[2]),
                    'bottom': float(model.dem_transform[5] + model.dem_transform[4] * model.dem.shape[0]),
                    'right': float(model.dem_transform[2] + model.dem_transform[0] * model.dem.shape[1]),
                    'top': float(model.dem_transform[5]),
                }

            self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


# ------------------------------------------------------------------
# Tab widget
# ------------------------------------------------------------------

class FlowTab(QWidget):
    """Sixth workflow tab: wind flow modelling."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._worker = None
        self._cached_result = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Settings ---
        settings_group = QGroupBox("Flow Model Settings")
        settings_form = QFormLayout()

        self._terrain_check = QCheckBox("Enable terrain correction")
        self._terrain_check.setChecked(True)
        settings_form.addRow(self._terrain_check)

        self._roughness_check = QCheckBox("Enable roughness correction")
        self._roughness_check.setChecked(True)
        settings_form.addRow(self._roughness_check)

        self._sectors_spin = QSpinBox()
        self._sectors_spin.setRange(12, 36)
        self._sectors_spin.setSingleStep(12)
        self._sectors_spin.setValue(12)
        settings_form.addRow("Number of Sectors:", self._sectors_spin)

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # --- Action ---
        self._run_btn = QPushButton("\U0001F30A  Run Wind Flow Model")
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        self._run_btn.clicked.connect(self._start_flow)
        layout.addWidget(self._run_btn)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        layout.addWidget(self._log)

        # --- Sector selection for speed-up preview ---
        sector_row_layout = QFormLayout()
        self._sector_combo = QComboBox()
        self._sector_combo.setEnabled(False)
        self._sector_combo.setToolTip(
            "Select a specific wind sector to view its terrain speed-up map.")
        sector_row_layout.addRow("View Sector Speed-up:", self._sector_combo)
        self._sector_combo.currentIndexChanged.connect(self._on_sector_changed)
        layout.addLayout(sector_row_layout)

        # --- Speed-up contour preview ---
        self._figure = Figure(figsize=(7, 4), dpi=100)
        self._figure.patch.set_facecolor('#2d2d30')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(220)
        layout.addWidget(self._canvas)

        # --- Per-turbine corrections table ---
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Turbine", "Terrain Speed-up", "Roughness Ratio", "Combined Factor"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _start_flow(self):
        data = self._main_window.get_project_data()
        positions = self._get_wtg_positions(data)
        if not positions:
            QMessageBox.warning(self, "No Layout",
                                "Please run layout optimization first.")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Flow model is already running.")
            return

        wind_data = data.get('wind_data', {})
        terrain = data.get('terrain', {})
        roughness = data.get('roughness', {})

        if self._terrain_check.isChecked() and not terrain.get('mosaic_path'):
            QMessageBox.warning(self, "No Terrain",
                                "Please download terrain data first.")
            return

        self._log.clear()
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._main_window.update_status("Running wind flow model\u2026")
        self._main_window.show_progress(True)

        self._worker = _FlowWorker(
            positions, wind_data, terrain, roughness,
            self._terrain_check.isChecked(), self._roughness_check.isChecked(),
            self._sectors_spin.value(), parent=self)
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
        data['flow_results'] = result
        self._cached_result = result
        self._run_btn.setEnabled(True)
        self._main_window.update_status("Wind flow modeling complete.")
        self._main_window.show_progress(False)
        self._populate_table(result.get('turbine_corrections', []))

        # Populate sector combo
        speedup_maps = result.get('terrain_speedup', {})
        self._sector_combo.blockSignals(True)
        self._sector_combo.clear()
        if speedup_maps:
            self._sector_combo.addItem("All Sectors (mean)")
            for sector_key in sorted(speedup_maps.keys()):
                direction = int(sector_key) * 30
                self._sector_combo.addItem(
                    f"Sector {int(sector_key)} ({direction}\u00b0)")
            self._sector_combo.setEnabled(True)
            self._sector_combo.setCurrentIndex(1)  # Default to first sector
        else:
            self._sector_combo.setEnabled(False)
        self._sector_combo.blockSignals(False)

        self._preview_speedup(result)

    def _on_finished_err(self, msg):
        self._log.append(f"ERROR: {msg}")
        self._run_btn.setEnabled(True)
        self._main_window.update_status("Wind flow modeling failed.")
        self._main_window.show_progress(False)
        QMessageBox.critical(self, "Error", f"Flow model failed:\n{msg}")

    def _on_sector_changed(self, index):
        """Update the speed-up preview when a different sector is selected."""
        if self._cached_result and index >= 0:
            self._preview_speedup(self._cached_result, sector_index=index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_wtg_positions(self, data):
        """Rebuild WTGPosition list from project data."""
        from src.core.layout_optimizer import WTGPosition

        layout_results = data.get('layout_results')
        if not layout_results or not layout_results.get('positions'):
            return []

        turb_spec = data.get('turbine_spec', {})
        positions = []
        for lat, lon, x, y, name in layout_results['positions']:
            p = WTGPosition(
                name=name, lat=lat, lon=lon,
                x_local=x, y_local=y,
                hub_height_m=turb_spec.get('hub_height_m', 80))
            positions.append(p)
        return positions

    def _populate_table(self, corrections):
        self._table.setRowCount(len(corrections))
        for row, corr in enumerate(corrections):
            self._table.setItem(row, 0, QTableWidgetItem(corr.get('name', '')))

            terrain_vals = corr.get('terrain_speedup_per_sector', [])
            avg_terrain = f"{np.mean(terrain_vals):.4f}" if terrain_vals else "\u2014"
            self._table.setItem(row, 1, QTableWidgetItem(avg_terrain))

            rough_vals = corr.get('roughness_ratio_per_sector', [])
            avg_rough = f"{np.mean(rough_vals):.4f}" if rough_vals else "\u2014"
            self._table.setItem(row, 2, QTableWidgetItem(avg_rough))

            combined_vals = corr.get('combined_factor_per_sector', [])
            avg_combined = f"{np.mean(combined_vals):.4f}" if combined_vals else "\u2014"
            self._table.setItem(row, 3, QTableWidgetItem(avg_combined))

    def _get_dem_geo_info(self, result):
        """Get DEM geographic bounds and transform from flow results.

        Returns (dem_bounds_dict, dem_transform) or (None, None).
        """
        # First try the cached info from the worker
        dem_bounds = result.get('_dem_bounds')
        dem_transform = result.get('_dem_transform')

        if dem_bounds and dem_transform:
            return dem_bounds, dem_transform

        # Fallback: load from the terrain mosaic file
        mosaic_path = self._main_window.get_project_data().get(
            'terrain', {}).get('mosaic_path')
        if mosaic_path:
            try:
                import rasterio
                with rasterio.open(mosaic_path) as src:
                    b = src.bounds
                    dem_bounds = {
                        'left': float(b.left), 'bottom': float(b.bottom),
                        'right': float(b.right), 'top': float(b.top),
                    }
                    dem_transform = src.transform
                return dem_bounds, dem_transform
            except Exception:
                pass

        return None, None

    def _preview_speedup(self, result, sector_index=None):
        """Plot speed-up map with geographic coordinates, turbines, boundary, and OSM."""
        speedup_maps = result.get('terrain_speedup', {})
        if not speedup_maps:
            self._log.append("Info: No speed-up maps available for preview.")
            return

        # Determine which sector to show
        if sector_index is None or sector_index <= 0:
            # "All Sectors (mean)" — compute frequency-weighted mean
            combined = self._compute_mean_speedup(speedup_maps)
            if combined is None:
                # Fallback: show first sector
                first_key = min(speedup_maps.keys())
                speedup = speedup_maps[first_key]
                title_suffix = f" (Sector {int(first_key) * 30}\u00b0)"
            else:
                speedup = combined
                title_suffix = " (All Sectors Mean)"
        else:
            # Specific sector selected (index 1-based in combo, 0-based in data)
            sector_idx = sector_index - 1
            sorted_keys = sorted(speedup_maps.keys())
            if sector_idx < len(sorted_keys):
                sector_key = sorted_keys[sector_idx]
                speedup = speedup_maps[sector_key]
                title_suffix = f" (Sector {int(sector_key) * 30}\u00b0)"
            else:
                speedup = speedup_maps[sorted_keys[0]]
                title_suffix = f" (Sector {int(sorted_keys[0]) * 30}\u00b0)"

        # Get geographic info from DEM
        dem_bounds, dem_transform = self._get_dem_geo_info(result)

        if dem_bounds:
            # We have geographic info — plot in lat/lon coordinates
            self._preview_speedup_geo(speedup, dem_bounds, dem_transform, result, title_suffix)
        else:
            # No geographic info — fall back to grid index display
            self._preview_speedup_grid(speedup, title_suffix)

    def _compute_mean_speedup(self, speedup_maps):
        """Compute a simple mean speed-up across all sectors."""
        if not speedup_maps:
            return None
        arrays = list(speedup_maps.values())
        # Use the first array as template for NaN mask
        result = np.zeros_like(arrays[0], dtype=np.float64)
        count = np.zeros_like(arrays[0], dtype=np.float64)
        for arr in arrays:
            valid = ~np.isnan(arr)
            result[valid] += arr[valid]
            count[valid] += 1
        mask = count > 0
        result[mask] /= count[mask]
        result[~mask] = np.nan
        return result

    def _preview_speedup_geo(self, speedup, dem_bounds, dem_transform, result, title_suffix):
        """Plot speed-up in geographic (lat/lon) coordinates with OSM, turbines, boundary."""
        extent = [dem_bounds['left'], dem_bounds['right'],
                  dem_bounds['bottom'], dem_bounds['top']]

        # Mask NaN values
        speedup_masked = np.ma.masked_invalid(speedup)

        # Try with OSM background
        try:
            self._preview_speedup_geo_osm(speedup_masked, extent, dem_transform, result, title_suffix)
            return
        except ImportError:
            self._log.append("Info: contextily not available. Showing speed-up without OSM background.")
        except Exception as e:
            self._log.append(f"Info: OSM background failed ({e}). Showing without.")

        # Fallback without OSM
        self._preview_speedup_geo_basic(speedup_masked, extent, result, title_suffix)

    def _preview_speedup_geo_osm(self, speedup_masked, extent, dem_transform, result, title_suffix):
        """Speed-up map in geo coords with OSM basemap."""
        import contextily as ctx

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        im = ax.imshow(
            speedup_masked, cmap='RdBu_r', aspect='auto',
            extent=extent, origin='upper',
            vmin=0.7, vmax=1.3)

        # Add OSM basemap
        try:
            ctx.add_basemap(
                ax, crs='EPSG:4326',
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom='auto', alpha=0.3)
        except Exception:
            pass

        # Overlay boundary polygon
        boundary = self._main_window.get_project_data().get('boundary')
        if boundary and len(boundary) >= 3:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.plot(bnd_lons, bnd_lats, color='#00FFFF', linewidth=2.0,
                    linestyle='--', label='Boundary', zorder=10)

        # Overlay turbine positions
        positions = self._get_wtg_positions(self._main_window.get_project_data())
        if positions:
            lons = [p.lon for p in positions]
            lats = [p.lat for p in positions]
            ax.scatter(lons, lats, c='red', s=50, marker='^',
                       edgecolors='white', linewidths=0.8, zorder=11,
                       label='Turbines')
            # Label turbines
            for p in positions:
                ax.annotate(
                    p.name, (p.lon, p.lat), fontsize=5,
                    textcoords="offset points", xytext=(4, 4),
                    color='white', fontweight='bold', zorder=12)

        # Colorbar with speed-up ratio labels
        ax.set_title(f"Terrain Speed-up{title_suffix} (with OSM)",
                     color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')

        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='Speed-up Ratio')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
        # Format ticks to 2 decimal places
        cb.ax.yaxis.set_major_formatter(
            __import__('matplotlib.ticker', fromlist=['FormatStrFormatter']).FormatStrFormatter('%.2f'))

        ax.legend(loc='upper right', fontsize=7, facecolor='#3d3d40',
                  edgecolor='#666', labelcolor='white')
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    def _preview_speedup_geo_basic(self, speedup_masked, extent, result, title_suffix):
        """Speed-up map in geo coords without OSM basemap."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        im = ax.imshow(
            speedup_masked, cmap='RdBu_r', aspect='auto',
            extent=extent, origin='upper',
            vmin=0.7, vmax=1.3)

        # Overlay boundary polygon
        boundary = self._main_window.get_project_data().get('boundary')
        if boundary and len(boundary) >= 3:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.plot(bnd_lons, bnd_lats, color='#00FFFF', linewidth=2.0,
                    linestyle='--', label='Boundary', zorder=10)

        # Overlay turbine positions
        positions = self._get_wtg_positions(self._main_window.get_project_data())
        if positions:
            lons = [p.lon for p in positions]
            lats = [p.lat for p in positions]
            ax.scatter(lons, lats, c='red', s=50, marker='^',
                       edgecolors='white', linewidths=0.8, zorder=11,
                       label='Turbines')
            for p in positions:
                ax.annotate(
                    p.name, (p.lon, p.lat), fontsize=5,
                    textcoords="offset points", xytext=(4, 4),
                    color='white', fontweight='bold', zorder=12)

        ax.set_title(f"Terrain Speed-up{title_suffix}",
                     color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')

        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='Speed-up Ratio')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
        cb.ax.yaxis.set_major_formatter(
            __import__('matplotlib.ticker', fromlist=['FormatStrFormatter']).FormatStrFormatter('%.2f'))

        ax.legend(loc='upper right', fontsize=7, facecolor='#3d3d40',
                  edgecolor='#666', labelcolor='white')
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    def _preview_speedup_grid(self, speedup, title_suffix):
        """Fallback: speed-up map using grid indices (when no DEM geo info)."""
        speedup_clean = np.where(np.isnan(speedup), 1.0, speedup)
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        im = ax.imshow(speedup_clean, cmap='RdBu_r', aspect='auto',
                       vmin=0.7, vmax=1.3)

        ax.set_title(f"Terrain Speed-up{title_suffix} (Grid Indices \u2014 No DEM Geo)",
                     color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Grid X", color='white')
        ax.set_ylabel("Grid Y", color='white')
        ax.tick_params(colors='white')
        cb = self._figure.colorbar(im, ax=ax, shrink=0.8, label='Speed-up Ratio')
        cb.ax.yaxis.set_tick_params(color='white')
        cb.ax.yaxis.label.set_color('white')
        cb.ax.yaxis.label.set_fontweight('bold')
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_color('white')
        cb.ax.yaxis.set_major_formatter(
            __import__('matplotlib.ticker', fromlist=['FormatStrFormatter']).FormatStrFormatter('%.2f'))
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_from_project(self):
        data = self._main_window.get_project_data()
        flow_results = data.get('flow_results')
        if flow_results:
            self._cached_result = flow_results

            # Re-populate sector combo
            speedup_maps = flow_results.get('terrain_speedup', {})
            self._sector_combo.blockSignals(True)
            self._sector_combo.clear()
            if speedup_maps:
                self._sector_combo.addItem("All Sectors (mean)")
                for sector_key in sorted(speedup_maps.keys()):
                    direction = int(sector_key) * 30
                    self._sector_combo.addItem(
                        f"Sector {int(sector_key)} ({direction}\u00b0)")
                self._sector_combo.setEnabled(True)
                self._sector_combo.setCurrentIndex(1)
            else:
                self._sector_combo.setEnabled(False)
            self._sector_combo.blockSignals(False)

            self._populate_table(flow_results.get('turbine_corrections', []))
            self._preview_speedup(flow_results)
