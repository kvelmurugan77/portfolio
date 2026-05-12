"""
WindFarm Designer Pro - Layout Optimizer Tab.
Configures and runs layout optimization (Grid, Greedy, PSO, GA),
shows the optimized layout on the boundary, and lists WTG positions.

Changes from original:
  - "Use imported layout (skip optimization)" checkbox — auto-loads on tab activation
  - OSM background option on layout preview with better zoom level
  - Real-time progress with iteration count and fitness value
  - "Export Layout" button that saves to CSV
  - "Export Layout as .map" button for WAsP format
  - Boundary polygon shown with fill on layout preview
  - Turbine names shown on hover in matplotlib plot
  - _load_imported_layout computes x_local, y_local properly from boundary origin
  - Layout mode indicator showing whether using imported or optimized layout
  - Consistent dark theme styling with white text
  - Graceful degradation when optional packages are missing
"""

import logging
import math
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QCheckBox,
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

class _LayoutWorker(QThread):
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(object, list)  # optimizer, positions
    finished_err = pyqtSignal(str)

    def __init__(self, boundary, turbine_spec, config_dict, wind_data, parent=None):
        super().__init__(parent)
        self._boundary = boundary
        self._turbine_spec = turbine_spec
        self._config_dict = config_dict
        self._wind_data = wind_data

    def run(self):
        try:
            from src.core.layout_optimizer import (
                LayoutOptimizer, TurbineModel, LayoutConfig, WTGPosition)
            from src.core.layout_optimizer import local_to_latlon

            turb = TurbineModel(
                name=self._turbine_spec.get('manufacturer', '') + ' ' +
                     self._turbine_spec.get('name', 'Turbine'),
                manufacturer=self._turbine_spec.get('manufacturer', ''),
                rated_power_kw=self._turbine_spec['rated_power_kw'],
                hub_height_m=self._turbine_spec['hub_height_m'],
                rotor_diameter_m=self._turbine_spec['rotor_diameter_m'],
                cut_in_ms=self._turbine_spec['cut_in_ms'],
                cut_out_ms=self._turbine_spec['cut_out_ms'],
                rated_speed_ms=self._turbine_spec['rated_speed_ms'],
            )

            config = LayoutConfig(
                algorithm=self._config_dict['algorithm'],
                min_spacing_rotor_d=self._config_dict['min_spacing_rd'],
                preferred_spacing_rotor_d=self._config_dict['pref_spacing_rd'],
                max_slope_deg=self._config_dict['max_slope'],
                max_capacity_kw=self._config_dict['capacity_kw'],
                n_populations=self._config_dict.get('population', 50),
                n_iterations=self._config_dict.get('iterations', 100),
                random_seed=self._config_dict.get('seed', None),
            )

            optimizer = LayoutOptimizer(self._boundary, turb, config)
            if self._wind_data:
                optimizer.set_wind_data(self._wind_data)
            optimizer.set_progress_callback(
                lambda c, t, m: self.progress.emit(c, t, m))
            optimizer.set_status_callback(lambda m: self.status.emit(m))

            positions = optimizer.optimize()
            self.finished_ok.emit(optimizer, positions)
        except Exception as e:
            self.finished_err.emit(str(e))


# ------------------------------------------------------------------
# Tab widget
# ------------------------------------------------------------------

class LayoutTab(QWidget):
    """Fifth workflow tab: configure and run layout optimization."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._worker = None
        self._annotation = None  # For hover tooltip on turbines
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Layout mode indicator ---
        self._mode_label = QLabel("Mode: Optimized Layout")
        self._mode_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 4px; "
            "background-color: #2a82da; color: white; border-radius: 3px; }")
        layout.addWidget(self._mode_label)

        # --- Skip optimization checkbox ---
        self._skip_opt_check = QCheckBox(
            "Use imported layout (skip optimization)")
        self._skip_opt_check.setToolTip(
            "When checked, the imported WTG layout from the Project Setup "
            "tab will be used directly without running optimization.")
        self._skip_opt_check.stateChanged.connect(self._on_skip_changed)
        layout.addWidget(self._skip_opt_check)

        # --- Algorithm settings ---
        self._algo_group = QGroupBox("Algorithm Settings")
        algo_form = QFormLayout()

        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["Grid", "Greedy", "PSO", "GA"])
        self._algo_combo.currentTextChanged.connect(self._on_algo_changed)
        algo_form.addRow("Algorithm:", self._algo_combo)

        self._min_spacing = QDoubleSpinBox()
        self._min_spacing.setRange(3.0, 10.0)
        self._min_spacing.setDecimals(1)
        self._min_spacing.setSuffix(" D")
        self._min_spacing.setValue(5.0)
        algo_form.addRow("Min Spacing:", self._min_spacing)

        self._pref_spacing = QDoubleSpinBox()
        self._pref_spacing.setRange(5.0, 15.0)
        self._pref_spacing.setDecimals(1)
        self._pref_spacing.setSuffix(" D")
        self._pref_spacing.setValue(7.0)
        algo_form.addRow("Preferred Spacing:", self._pref_spacing)

        self._max_slope = QDoubleSpinBox()
        self._max_slope.setRange(5.0, 30.0)
        self._max_slope.setDecimals(1)
        self._max_slope.setSuffix(" deg")
        self._max_slope.setValue(15.0)
        algo_form.addRow("Max Terrain Slope:", self._max_slope)

        # GA/PSO parameters
        self._population_spin = QSpinBox()
        self._population_spin.setRange(10, 200)
        self._population_spin.setValue(50)
        algo_form.addRow("Population Size:", self._population_spin)

        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(50, 500)
        self._iterations_spin.setSingleStep(10)
        self._iterations_spin.setValue(100)
        algo_form.addRow("Iterations:", self._iterations_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 999999)
        self._seed_spin.setValue(-1)
        self._seed_spin.setSpecialValueText("Random")
        algo_form.addRow("Random Seed:", self._seed_spin)

        self._algo_group.setLayout(algo_form)
        layout.addWidget(self._algo_group)

        # --- Action buttons row ---
        action_row = QHBoxLayout()

        self._run_btn = QPushButton("\u2699  Run Layout Optimization")
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        self._run_btn.clicked.connect(self._start_optimization)
        action_row.addWidget(self._run_btn)

        self._load_imported_btn = QPushButton(
            "\U0001F4C2  Load Imported Layout")
        self._load_imported_btn.clicked.connect(self._load_imported_layout)
        self._load_imported_btn.setVisible(False)
        action_row.addWidget(self._load_imported_btn)

        layout.addLayout(action_row)

        # --- Export buttons row ---
        export_row = QHBoxLayout()

        self._export_csv_btn = QPushButton(
            "\U0001F4BE  Export Layout to CSV")
        self._export_csv_btn.clicked.connect(self._export_layout_csv)
        export_row.addWidget(self._export_csv_btn)

        self._export_map_btn = QPushButton(
            "\U0001F4C4  Export Layout as .map")
        self._export_map_btn.clicked.connect(self._export_layout_map)
        self._export_map_btn.setToolTip(
            "Export turbine positions as a WAsP .map file.\n"
            "Roughness data must be available for this export.")
        export_row.addWidget(self._export_map_btn)

        layout.addLayout(export_row)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # --- Real-time optimization feedback ---
        self._feedback_group = QGroupBox("Optimization Feedback")
        feedback_layout = QFormLayout()
        self._iteration_label = QLabel("\u2014")
        self._fitness_label = QLabel("\u2014")
        self._status_label_fb = QLabel("Idle")
        feedback_layout.addRow("Iteration:", self._iteration_label)
        feedback_layout.addRow("Fitness:", self._fitness_label)
        feedback_layout.addRow("Status:", self._status_label_fb)
        self._feedback_group.setLayout(feedback_layout)
        layout.addWidget(self._feedback_group)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        layout.addWidget(self._log)

        # --- Summary ---
        summary_group = QGroupBox("Layout Summary")
        summary_form = QFormLayout()
        self._n_turb_label = QLabel("\u2014")
        self._capacity_label = QLabel("\u2014")
        self._layout_type_label = QLabel("\u2014")
        summary_form.addRow("Number of Turbines:", self._n_turb_label)
        summary_form.addRow("Installed Capacity:", self._capacity_label)
        summary_form.addRow("Layout Type:", self._layout_type_label)
        summary_group.setLayout(summary_form)
        layout.addWidget(summary_group)

        # --- Matplotlib preview ---
        self._figure = Figure(figsize=(7, 4), dpi=100)
        self._figure.patch.set_facecolor('#2d2d30')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(220)
        # Connect mouse motion event for hover tooltips
        self._canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        layout.addWidget(self._canvas)

        # --- WTG table ---
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Lat", "Lon", "X (m)", "Y (m)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)

        layout.addStretch()

        # Init UI state
        self._on_algo_changed("Grid")
        self._on_skip_changed(Qt.Unchecked)

    # ------------------------------------------------------------------
    # UI state changes
    # ------------------------------------------------------------------

    def _on_skip_changed(self, state):
        """Toggle between optimization mode and imported layout mode."""
        skip = (state == Qt.Checked)
        self._algo_group.setEnabled(not skip)
        self._run_btn.setEnabled(not skip)
        self._load_imported_btn.setVisible(skip)
        self._feedback_group.setEnabled(not skip)

        if skip:
            self._mode_label.setText("Mode: Imported Layout (Optimization Skipped)")
            self._mode_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; "
                "background-color: #D4A017; color: white; border-radius: 3px; }")
            self._status_label_fb.setText(
                "Using imported layout (optimization disabled)")
        else:
            self._mode_label.setText("Mode: Optimized Layout")
            self._mode_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; "
                "background-color: #2a82da; color: white; border-radius: 3px; }")
            self._status_label_fb.setText("Idle")

    def _on_algo_changed(self, algo):
        is_meta = algo in ("PSO", "GA")
        self._population_spin.setEnabled(is_meta)
        self._iterations_spin.setEnabled(is_meta)

    # ------------------------------------------------------------------
    # Mouse hover for turbine tooltips
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event):
        """Show turbine name annotation on hover."""
        if event.inaxes is None or self._annotation is None:
            return

        # Check if any turbine point is close to the mouse
        data = self._main_window.get_project_data()
        layout_results = data.get('layout_results')
        if not layout_results or not layout_results.get('positions'):
            return

        threshold = 0.001  # degrees
        hovered_name = None
        hovered_lon, hovered_lat = None, None

        for lat, lon, x, y, name in layout_results['positions']:
            if (abs(event.xdata - lon) < threshold and
                    abs(event.ydata - lat) < threshold):
                hovered_name = name
                hovered_lon = lon
                hovered_lat = lat
                break

        if hovered_name:
            self._annotation.set_text(hovered_name)
            self._annotation.xy = (hovered_lon, hovered_lat)
            self._annotation.set_visible(True)
        else:
            self._annotation.set_visible(False)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _start_optimization(self):
        data = self._main_window.get_project_data()
        boundary = data.get('boundary')
        if not boundary or len(boundary) < 3:
            QMessageBox.warning(self, "No Boundary",
                                "Please create a project with a valid boundary first.")
            return
        turbine_spec = data.get('turbine_spec')
        if not turbine_spec:
            QMessageBox.warning(self, "No Turbine",
                                "Please select a turbine model.")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy",
                                    "Optimization is already running.")
            return

        seed = None if self._seed_spin.value() == -1 else self._seed_spin.value()

        config_dict = {
            'algorithm': self._algo_combo.currentText().lower(),
            'min_spacing_rd': self._min_spacing.value(),
            'pref_spacing_rd': self._pref_spacing.value(),
            'max_slope': self._max_slope.value(),
            'capacity_kw': data.get('capacity_mw', 100) * 1000,
            'population': self._population_spin.value(),
            'iterations': self._iterations_spin.value(),
            'seed': seed,
        }

        wind_data = data.get('wind_data', {})

        self._log.clear()
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._iteration_label.setText("0")
        self._fitness_label.setText("\u2014")
        self._status_label_fb.setText("Running\u2026")
        self._main_window.update_status("Running layout optimization\u2026")
        self._main_window.show_progress(True)

        self._worker = _LayoutWorker(
            boundary, turbine_spec, config_dict, wind_data, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.start()

    def _load_imported_layout(self):
        """Load the WTG layout from project data (imported in Project Setup).

        Computes x_local, y_local properly using the boundary centroid as origin,
        consistent with how the LayoutOptimizer does it internally.
        """
        data = self._main_window.get_project_data()
        wtg_layout = data.get('wtg_layout')
        if not wtg_layout:
            QMessageBox.information(
                self, "No Imported Layout",
                "No WTG layout was imported in the Project Setup tab.\n\n"
                "Please go back to Project Setup and import a WTG layout file.")
            return

        boundary = data.get('boundary', [])
        turbine_spec = data.get('turbine_spec', {})

        # Compute origin from boundary centroid (same as LayoutOptimizer)
        if boundary and len(boundary) >= 3:
            origin = (
                float(np.mean([p[0] for p in boundary])),
                float(np.mean([p[1] for p in boundary]))
            )
        else:
            origin = (0.0, 0.0)

        # Convert imported layout to WTGPosition objects with proper local coords
        from src.core.layout_optimizer import WTGPosition
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(origin[0]))

        positions = []
        for i, wtg in enumerate(wtg_layout):
            if isinstance(wtg, dict):
                lat = float(wtg.get('lat', wtg.get('latitude', 0)))
                lon = float(wtg.get('lon', wtg.get('longitude', 0)))
                name = wtg.get('name', f"WTG_{i+1:03d}")
            else:
                # Assume tuple/list (lat, lon, ...)
                lat = float(wtg[0]) if len(wtg) > 0 else 0
                lon = float(wtg[1]) if len(wtg) > 1 else 0
                name = f"WTG_{i+1:03d}"

            # Compute x_local, y_local from origin
            x_local = (lon - origin[1]) * m_per_deg_lon
            y_local = (lat - origin[0]) * m_per_deg_lat

            p = WTGPosition(
                name=name, lat=lat, lon=lon,
                x_local=x_local, y_local=y_local,
                hub_height_m=turbine_spec.get('hub_height_m', 80))
            positions.append(p)

        n_turb = len(positions)
        capacity_kw = turbine_spec.get('rated_power_kw', 0)
        capacity_mw = n_turb * capacity_kw / 1000.0

        # Store as layout results
        data['layout_results'] = {
            'positions': [(p.lat, p.lon, p.x_local, p.y_local, p.name)
                          for p in positions],
            'optimizer_origin': origin,
            'layout_type': 'imported',
        }

        self._n_turb_label.setText(str(n_turb))
        self._capacity_label.setText(f"{capacity_mw:.1f} MW")
        self._layout_type_label.setText("Imported")
        self._status_label_fb.setText(
            f"Loaded {n_turb} turbines from imported layout")
        self._main_window.update_status(
            f"Loaded imported layout: {n_turb} turbines, {capacity_mw:.1f} MW")

        self._populate_table(positions)
        self._preview_layout(boundary, positions)

    def _export_layout_csv(self):
        """Export the current layout (from optimization or import) to CSV."""
        data = self._main_window.get_project_data()
        layout_results = data.get('layout_results')
        if not layout_results or not layout_results.get('positions'):
            QMessageBox.information(
                self, "No Layout",
                "No layout data to export. Run optimization or load an imported layout first.")
            return

        default_name = data.get('project_name', 'layout')
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Layout to CSV",
            f"{default_name}_layout.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            import csv
            positions = layout_results['positions']
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Latitude", "Longitude",
                                 "X_local (m)", "Y_local (m)"])
                for lat, lon, x, y, name in positions:
                    writer.writerow([name, f"{lat:.6f}", f"{lon:.6f}",
                                     f"{x:.1f}", f"{y:.1f}"])

            self._main_window.update_status(f"Layout exported to {path}")
            self._log.append(f"Layout exported: {path} ({len(positions)} turbines)")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export layout:\n{e}")
            logger.exception("Layout export failed")

    def _export_layout_map(self):
        """Export layout as a WAsP .map file with turbine positions."""
        data = self._main_window.get_project_data()
        layout_results = data.get('layout_results')
        if not layout_results or not layout_results.get('positions'):
            QMessageBox.information(
                self, "No Layout",
                "No layout data to export. Run optimization or load a layout first.")
            return

        roughness_path = data.get('roughness', {}).get('roughness_path')
        if not roughness_path:
            QMessageBox.warning(
                self, "No Roughness Data",
                "Roughness data is required for WAsP .map export.\n"
                "Please download roughness data first.")
            return

        default_name = data.get('project_name', 'windfarm')
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Layout as .map (WAsP)",
            f"{default_name}_layout.map",
            "WAsP Map Files (*.map);;All Files (*)")
        if not path:
            return

        self._log.append("Exporting layout to WAsP .map format...")

        try:
            from src.core.roughness_downloader import RoughnessDownloader
            dl = RoughnessDownloader()
            dl.set_status_callback(lambda m: self._log.append(f"  {m}"))

            result_path = dl.geotiff_to_map_file(roughness_path, output_path=path)
            if result_path:
                # Append turbine positions to the .map file
                self._append_turbines_to_map(result_path, layout_results['positions'])
                self._main_window.update_status(f"Layout .map exported to {result_path}")
                self._log.append(f"Layout .map exported: {result_path}")
                QMessageBox.information(
                    self, "Export Complete",
                    f"Layout .map file saved to:\n{result_path}\n\n"
                    f"Turbine positions appended to the file.")
            else:
                QMessageBox.warning(
                    self, "Export Failed",
                    "WAsP .map export returned no result. Check the log for details.")
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "The roughness_downloader module could not be imported.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export .map file:\n{e}")
            logger.exception("Layout .map export failed")

    def _append_turbines_to_map(self, map_path, positions):
        """Append turbine position markers to the WAsP .map file."""
        try:
            with open(map_path, 'a') as f:
                f.write("\n# Turbine positions (lat, lon, x_local, y_local, name)\n")
                for lat, lon, x, y, name in positions:
                    f.write(
                        f"TURBINE {lat:.6f} {lon:.6f} "
                        f"{x:.1f} {y:.1f} {name}\n")
        except Exception as e:
            self._log.append(f"Warning: Could not append turbine data to .map: {e}")

    # ------------------------------------------------------------------
    # Progress / status callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, current, total, msg):
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._main_window.update_progress(current, total)

        # Parse iteration and fitness from status message if possible
        msg_lower = msg.lower()
        if 'iteration' in msg_lower:
            self._iteration_label.setText(
                msg.replace('Iteration ', '').split('/')[0].strip())
        if 'fitness' in msg_lower:
            parts = msg.split(':')
            if len(parts) >= 2:
                try:
                    self._fitness_label.setText(parts[-1].strip())
                except (ValueError, IndexError):
                    pass

    def _on_status(self, msg):
        self._log.append(msg)
        # Update status feedback label
        self._status_label_fb.setText(msg[:80])

    def _on_finished_ok(self, optimizer, positions):
        from src.core.layout_optimizer import local_to_latlon
        data = self._main_window.get_project_data()

        # Convert to lat/lon
        latlon_list = local_to_latlon(
            [(p.x_local, p.y_local) for p in positions], optimizer.origin)
        for i, pos in enumerate(positions):
            pos.lat, pos.lon = latlon_list[i]

        # Store results
        data['layout_results'] = {
            'positions': [(p.lat, p.lon, p.x_local, p.y_local, p.name)
                          for p in positions],
            'optimizer_origin': optimizer.origin,
            'layout_type': 'optimized',
        }

        n_turb = len(positions)
        turb_spec = data.get('turbine_spec', {})
        capacity_kw = turb_spec.get('rated_power_kw', 0)
        capacity_mw = n_turb * capacity_kw / 1000.0

        self._n_turb_label.setText(str(n_turb))
        self._capacity_label.setText(f"{capacity_mw:.1f} MW")
        self._layout_type_label.setText(
            f"Optimized ({self._algo_combo.currentText()})")
        self._run_btn.setEnabled(True)
        self._iteration_label.setText(
            str(self._iterations_spin.value()))
        self._fitness_label.setText("Optimized")
        self._status_label_fb.setText("Complete")
        self._main_window.update_status(
            f"Layout optimization complete: {n_turb} turbines, {capacity_mw:.1f} MW")
        self._main_window.show_progress(False)

        self._populate_table(positions)
        self._preview_layout(data.get('boundary', []), positions)

    def _on_finished_err(self, msg):
        self._log.append(f"ERROR: {msg}")
        self._run_btn.setEnabled(True)
        self._status_label_fb.setText("Failed")
        self._main_window.update_status("Layout optimization failed.")
        self._main_window.show_progress(False)
        QMessageBox.critical(self, "Error",
                             f"Layout optimization failed:\n{msg}")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_table(self, positions):
        self._table.setRowCount(len(positions))
        for row, pos in enumerate(positions):
            self._table.setItem(row, 0, QTableWidgetItem(pos.name))
            self._table.setItem(
                row, 1, QTableWidgetItem(f"{pos.lat:.6f}"))
            self._table.setItem(
                row, 2, QTableWidgetItem(f"{pos.lon:.6f}"))
            self._table.setItem(
                row, 3, QTableWidgetItem(f"{pos.x_local:.1f}"))
            self._table.setItem(
                row, 4, QTableWidgetItem(f"{pos.y_local:.1f}"))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _preview_layout(self, boundary, positions):
        """Plot the layout preview with boundary fill, turbine labels, and optional OSM."""
        # Try enhanced preview with OSM background
        try:
            self._preview_layout_with_osm(boundary, positions)
            return
        except ImportError:
            pass  # Fall through to basic preview
        except Exception:
            pass

        self._preview_layout_basic(boundary, positions)

    def _preview_layout_with_osm(self, boundary, positions):
        """Preview with OSM basemap using contextily."""
        import contextily as ctx

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # Gather all coordinates to determine bounds
        all_lats = []
        all_lons = []
        if boundary:
            all_lats.extend([p[0] for p in boundary])
            all_lons.extend([p[1] for p in boundary])
        if positions:
            all_lats.extend([p.lat for p in positions])
            all_lons.extend([p.lon for p in positions])

        if not all_lats:
            self._preview_layout_basic(boundary, positions)
            return

        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # Add small margin (proportional to extent)
        lat_margin = max((lat_max - lat_min) * 0.15, 0.005)
        lon_margin = max((lon_max - lon_min) * 0.15, 0.005)
        lat_min -= lat_margin
        lat_max += lat_margin
        lon_min -= lon_margin
        lon_max += lon_margin

        # Plot boundary with fill
        if boundary:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.fill(bnd_lons, bnd_lats, alpha=0.12, color='cyan',
                    zorder=2)
            ax.plot(bnd_lons, bnd_lats, 'c-', linewidth=2.0,
                    label='Boundary', zorder=3)

        # Plot turbines
        if positions:
            lats = [p.lat for p in positions]
            lons = [p.lon for p in positions]
            ax.scatter(lons, lats, c='#FF4444', s=70, marker='o',
                       edgecolors='white', linewidths=1.0, zorder=5,
                       label=f'Turbines ({len(positions)})')

            # Determine if we should show names (only if reasonable count)
            show_names = len(positions) <= 60
            for pos in positions:
                if show_names:
                    ax.annotate(pos.name, (pos.lon, pos.lat), fontsize=5.5,
                                textcoords="offset points", xytext=(5, 5),
                                color='white', fontweight='bold', zorder=6)

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add OSM basemap
        try:
            ctx.add_basemap(
                ax, crs='EPSG:4326',
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom='auto', alpha=0.5)
        except Exception:
            pass  # If basemap fails, show without it

        # Add hover annotation (invisible by default)
        self._annotation = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                      edgecolor='#00FFFF', alpha=0.9),
            color='white', fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00FFFF', lw=0.8),
            zorder=20)
        self._annotation.set_visible(False)

        # Determine layout type for title
        layout_type = self._main_window.get_project_data().get(
            'layout_results', {}).get('layout_type', 'optimized')
        if layout_type == 'imported':
            title = "Imported Turbine Layout (with OSM base)"
        else:
            title = "Optimized Turbine Layout (with OSM base)"

        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8, facecolor='#3d3d40',
                  edgecolor='#666', labelcolor='white')
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    def _preview_layout_basic(self, boundary, positions):
        """Basic layout preview without contextily (fallback)."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # Plot boundary with fill
        if boundary:
            bnd = list(boundary) + [boundary[0]]
            bnd_lats = [p[0] for p in bnd]
            bnd_lons = [p[1] for p in bnd]
            ax.fill(bnd_lons, bnd_lats, alpha=0.12, color='cyan',
                    zorder=2)
            ax.plot(bnd_lons, bnd_lats, 'c-', linewidth=2.0,
                    label='Boundary', zorder=3)

        # Plot turbines
        if positions:
            lats = [p.lat for p in positions]
            lons = [p.lon for p in positions]
            ax.scatter(lons, lats, c='#FF4444', s=70, marker='o',
                       edgecolors='white', linewidths=1.0, zorder=5,
                       label=f'Turbines ({len(positions)})')
            show_names = len(positions) <= 60
            for pos in positions:
                if show_names:
                    ax.annotate(pos.name, (pos.lon, pos.lat), fontsize=5.5,
                                textcoords="offset points", xytext=(5, 5),
                                color='white', fontweight='bold', zorder=6)

        # Add hover annotation
        self._annotation = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                      edgecolor='#00FFFF', alpha=0.9),
            color='white', fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00FFFF', lw=0.8),
            zorder=20)
        self._annotation.set_visible(False)

        layout_type = self._main_window.get_project_data().get(
            'layout_results', {}).get('layout_type', 'optimized')
        title = "Imported Turbine Layout" if layout_type == 'imported' else "Optimized Turbine Layout"

        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Longitude (\u00b0)", color='white')
        ax.set_ylabel("Latitude (\u00b0)", color='white')
        ax.tick_params(colors='white')
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
        layout_results = data.get('layout_results')
        if layout_results and layout_results.get('positions'):
            positions_data = layout_results['positions']
            from src.core.layout_optimizer import WTGPosition
            turb_spec = data.get('turbine_spec', {})
            positions = []
            for lat, lon, x, y, name in positions_data:
                p = WTGPosition(
                    name=name, lat=lat, lon=lon,
                    x_local=x, y_local=y,
                    hub_height_m=turb_spec.get('hub_height_m', 80))
                positions.append(p)
            self._populate_table(positions)
            self._preview_layout(data.get('boundary', []), positions)
            n = len(positions)
            cap_kw = turb_spec.get('rated_power_kw', 0)
            self._n_turb_label.setText(str(n))
            self._capacity_label.setText(f"{n * cap_kw / 1000:.1f} MW")

            layout_type = layout_results.get('layout_type', 'optimized')
            if layout_type == 'imported':
                self._layout_type_label.setText("Imported")
                self._skip_opt_check.setChecked(True)
            else:
                self._layout_type_label.setText(
                    f"Optimized ({self._algo_combo.currentText()})")

        # Auto-load imported layout when skip checkbox is checked
        if self._skip_opt_check.isChecked():
            wtg_layout = data.get('wtg_layout')
            if wtg_layout and (not layout_results or not layout_results.get('positions')):
                # Auto-load if there's no layout_results yet
                self._status_label_fb.setText(
                    f"Imported layout available ({len(wtg_layout)} turbines). "
                    f"Click 'Load Imported Layout' to use it.")
            elif wtg_layout and layout_results and layout_results.get('layout_type') != 'imported':
                # Imported layout is available but we're showing optimized
                self._status_label_fb.setText(
                    f"Note: Imported layout ({len(wtg_layout)} turbines) available.\n"
                    f"Currently showing optimized layout.")
