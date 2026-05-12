"""
WindFarm Designer Pro - AEP Results Tab.
Calculates Annual Energy Production (gross / net), displays per-turbine
results, supports CSV and PDF export.
"""

import logging
import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QComboBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFileDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

class _AEPWorker(QThread):
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, positions, turbine_spec, power_curve, wind_data,
                 loss_factors_dict, wake_config_dict, include_wakes, parent=None):
        super().__init__(parent)
        self._positions = positions
        self._turbine_spec = turbine_spec
        self._power_curve = power_curve
        self._wind_data = wind_data
        self._loss_factors_dict = loss_factors_dict
        self._wake_config_dict = wake_config_dict
        self._include_wakes = include_wakes

    def run(self):
        try:
            from src.core.aep_calculator import AEPCalculator, LossFactors
            from src.core.layout_optimizer import TurbineModel
            from src.core.wake_model import WakeModelConfig

            turb = TurbineModel(
                name=self._turbine_spec.get('name', 'Turbine'),
                manufacturer=self._turbine_spec.get('manufacturer', ''),
                rated_power_kw=self._turbine_spec['rated_power_kw'],
                hub_height_m=self._turbine_spec['hub_height_m'],
                rotor_diameter_m=self._turbine_spec['rotor_diameter_m'],
                cut_in_ms=self._turbine_spec['cut_in_ms'],
                cut_out_ms=self._turbine_spec['cut_out_ms'],
                rated_speed_ms=self._turbine_spec['rated_speed_ms'],
            )

            lf = LossFactors(
                availability_pct=self._loss_factors_dict['availability'],
                electrical_loss_pct=self._loss_factors_dict['electrical'],
                curtailment_pct=self._loss_factors_dict['curtailment'],
                environmental_pct=self._loss_factors_dict['environmental'],
                icing_pct=self._loss_factors_dict['icing'],
                other_losses_pct=self._loss_factors_dict['other'],
            )

            wc = WakeModelConfig(
                model_type=self._wake_config_dict['model_type'],
                wake_decay_constant=self._wake_config_dict['decay_constant'],
                turbulence_intensity=self._wake_config_dict['turbulence_intensity'],
                wind_sectors=12,
            )

            calc = AEPCalculator(turb, self._power_curve, lf, wc)
            # Override internal status with signal
            orig_status = calc._report_status
            def _safe_status(msg):
                try:
                    self.status.emit(msg)
                except RuntimeError:
                    pass
            calc._report_status = _safe_status

            result = calc.compute_farm_aep(
                self._positions, self._wind_data, include_wakes=self._include_wakes)
            self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


# ------------------------------------------------------------------
# Tab widget
# ------------------------------------------------------------------

class ResultsTab(QWidget):
    """Seventh workflow tab: AEP calculation and export."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._worker = None
        self._aep_results = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)

        # --- Wake model settings ---
        wake_group = QGroupBox("Wake Model Settings")
        wake_form = QFormLayout()

        self._wake_check = QCheckBox("Include wake losses")
        self._wake_check.setChecked(True)
        wake_form.addRow(self._wake_check)

        self._wake_model_combo = QComboBox()
        self._wake_model_combo.addItems(["Jensen", "Frandsen", "Ainslie"])
        wake_form.addRow("Wake Model:", self._wake_model_combo)

        self._decay_spin = QDoubleSpinBox()
        self._decay_spin.setRange(0.03, 0.10)
        self._decay_spin.setDecimals(3)
        self._decay_spin.setSingleStep(0.005)
        self._decay_spin.setValue(0.05)
        wake_form.addRow("Wake Decay Constant:", self._decay_spin)

        self._ti_spin = QDoubleSpinBox()
        self._ti_spin.setRange(0.05, 0.25)
        self._ti_spin.setDecimals(2)
        self._ti_spin.setSingleStep(0.01)
        self._ti_spin.setValue(0.10)
        wake_form.addRow("Turbulence Intensity:", self._ti_spin)

        wake_group.setLayout(wake_form)
        layout.addWidget(wake_group)

        # --- Loss factors ---
        loss_group = QGroupBox("Loss Factors (%)")
        loss_form = QGridLayout()

        self._loss_availability = QDoubleSpinBox()
        self._loss_availability.setRange(80, 100); self._loss_availability.setDecimals(1)
        self._loss_availability.setValue(97.0); self._loss_availability.setSuffix(" %")
        self._loss_electrical = QDoubleSpinBox()
        self._loss_electrical.setRange(0, 20); self._loss_electrical.setDecimals(1)
        self._loss_electrical.setValue(2.0); self._loss_electrical.setSuffix(" %")
        self._loss_curtailment = QDoubleSpinBox()
        self._loss_curtailment.setRange(0, 20); self._loss_curtailment.setDecimals(1)
        self._loss_curtailment.setValue(1.0); self._loss_curtailment.setSuffix(" %")
        self._loss_environmental = QDoubleSpinBox()
        self._loss_environmental.setRange(0, 10); self._loss_environmental.setDecimals(1)
        self._loss_environmental.setValue(0.5); self._loss_environmental.setSuffix(" %")
        self._loss_icing = QDoubleSpinBox()
        self._loss_icing.setRange(0, 20); self._loss_icing.setDecimals(1)
        self._loss_icing.setValue(0.5); self._loss_icing.setSuffix(" %")
        self._loss_other = QDoubleSpinBox()
        self._loss_other.setRange(0, 20); self._loss_other.setDecimals(1)
        self._loss_other.setValue(1.0); self._loss_other.setSuffix(" %")

        labels_spins = [
            ("Availability:", self._loss_availability),
            ("Electrical Loss:", self._loss_electrical),
            ("Curtailment:", self._loss_curtailment),
            ("Environmental:", self._loss_environmental),
            ("Icing:", self._loss_icing),
            ("Other:", self._loss_other),
        ]
        for row_i, (label_text, spin) in enumerate(labels_spins):
            loss_form.addWidget(QLabel(label_text), row_i, 0)
            loss_form.addWidget(spin, row_i, 1)

        loss_group.setLayout(loss_form)
        layout.addWidget(loss_group)

        # --- Calculate button ---
        self._calc_btn = QPushButton("\U0001F4CA  Calculate AEP")
        self._calc_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; font-weight: bold; }")
        self._calc_btn.clicked.connect(self._calculate)
        layout.addWidget(self._calc_btn)

        # --- Farm Summary ---
        summary_group = QGroupBox("Farm Summary")
        summary_form = QFormLayout()
        self._summary_n_turb = QLabel("\u2014")
        self._summary_capacity = QLabel("\u2014")
        self._summary_gross_aep = QLabel("\u2014")
        self._summary_net_aep = QLabel("\u2014")
        self._summary_wake_loss = QLabel("\u2014")
        self._summary_cf = QLabel("\u2014")
        self._summary_flh = QLabel("\u2014")
        self._summary_other_loss = QLabel("\u2014")

        summary_form.addRow("Number of Turbines:", self._summary_n_turb)
        summary_form.addRow("Installed Capacity:", self._summary_capacity)
        summary_form.addRow("Gross AEP:", self._summary_gross_aep)
        summary_form.addRow("Net AEP:", self._summary_net_aep)
        summary_form.addRow("Wake Loss:", self._summary_wake_loss)
        summary_form.addRow("Other Losses Factor:", self._summary_other_loss)
        summary_form.addRow("Capacity Factor:", self._summary_cf)
        summary_form.addRow("Full Load Hours:", self._summary_flh)

        summary_group.setLayout(summary_form)
        layout.addWidget(summary_group)

        # --- Per-turbine table ---
        table_group = QGroupBox("Per-Turbine AEP Results")
        table_layout = QVBoxLayout()
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Name", "WS (m/s)", "Gross (GWh)", "Wake %", "Net (GWh)",
            "Final (GWh)", "CF (%)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setMinimumHeight(180)
        table_layout.addWidget(self._table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # --- AEP bar chart ---
        self._figure = Figure(figsize=(8, 3.5), dpi=100)
        self._figure.patch.set_facecolor('#2d2d30')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(200)
        layout.addWidget(self._canvas)

        # --- Export buttons ---
        export_row = QHBoxLayout()
        btn_csv = QPushButton("\U0001F4C4  Export to CSV")
        btn_csv.clicked.connect(self.export_to_csv)
        btn_pdf = QPushButton("\U0001F5A8  Export to PDF Report")
        btn_pdf.clicked.connect(self.export_to_pdf)
        export_row.addWidget(btn_csv)
        export_row.addWidget(btn_pdf)
        layout.addLayout(export_row)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _calculate(self):
        data = self._main_window.get_project_data()
        positions = self._get_wtg_positions(data)
        if not positions:
            QMessageBox.warning(self, "No Layout",
                                "Please run layout optimization first.")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "AEP calculation is already running.")
            return

        turbine_spec = data.get('turbine_spec')
        power_curve = data.get('power_curve')
        wind_data = data.get('wind_data', {})

        if not turbine_spec:
            QMessageBox.warning(self, "No Turbine", "Please select a turbine model.")
            return

        loss_dict = {
            'availability': self._loss_availability.value(),
            'electrical': self._loss_electrical.value(),
            'curtailment': self._loss_curtailment.value(),
            'environmental': self._loss_environmental.value(),
            'icing': self._loss_icing.value(),
            'other': self._loss_other.value(),
        }

        wake_type_map = {"Jensen": "jensen", "Frandsen": "frandsen", "Ainslie": "ainslie"}
        wake_dict = {
            'model_type': wake_type_map.get(self._wake_model_combo.currentText(), "jensen"),
            'decay_constant': self._decay_spin.value(),
            'turbulence_intensity': self._ti_spin.value(),
        }

        include_wakes = self._wake_check.isChecked()

        self._calc_btn.setEnabled(False)
        self._main_window.update_status("Calculating AEP\u2026")
        self._main_window.show_progress(True)

        self._worker = _AEPWorker(
            positions, turbine_spec, power_curve, wind_data,
            loss_dict, wake_dict, include_wakes, parent=self)
        self._worker.status.connect(lambda m: self._main_window.update_status(m))
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.start()

    def _on_finished_ok(self, result):
        self._aep_results = result
        data = self._main_window.get_project_data()
        data['aep_results'] = result

        summary = result.get('summary', {})
        self._summary_n_turb.setText(str(summary.get('n_turbines', 0)))
        self._summary_capacity.setText(f"{summary.get('installed_capacity_mw', 0):.1f} MW")
        self._summary_gross_aep.setText(f"{summary.get('total_gross_aep_gwh', 0):.2f} GWh/yr")
        self._summary_net_aep.setText(f"{summary.get('total_net_aep_gwh', 0):.2f} GWh/yr")
        self._summary_wake_loss.setText(f"{summary.get('total_wake_loss_pct', 0):.1f} %")
        self._summary_cf.setText(f"{summary.get('capacity_factor_pct', 0):.1f} %")
        self._summary_flh.setText(f"{summary.get('full_load_hours', 0):.0f} h/yr")

        loss_info = result.get('loss_factors', {})
        self._summary_other_loss.setText(
            f"{loss_info.get('total_loss_factor_pct', 0):.2f} %")

        self._populate_table(result.get('wtg_results', []))
        self._preview_aep(result.get('wtg_results', []))

        self._calc_btn.setEnabled(True)
        self._main_window.update_status("AEP calculation complete.")
        self._main_window.show_progress(False)

    def _on_finished_err(self, msg):
        self._calc_btn.setEnabled(True)
        self._main_window.update_status("AEP calculation failed.")
        self._main_window.show_progress(False)
        QMessageBox.critical(self, "Error", f"AEP calculation failed:\n{msg}")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_table(self, wtg_results):
        self._table.setRowCount(len(wtg_results))
        for row, r in enumerate(wtg_results):
            self._table.setItem(row, 0, QTableWidgetItem(r.get('name', '')))
            self._table.setItem(row, 1, QTableWidgetItem(f"{r.get('mean_wind_speed_ms', 0):.1f}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{r.get('gross_aep_gwh', 0):.4f}"))
            self._table.setItem(row, 3, QTableWidgetItem(f"{r.get('wake_loss_pct', 0):.2f}"))
            self._table.setItem(row, 4, QTableWidgetItem(f"{r.get('net_aep_gwh', 0):.4f}"))
            self._table.setItem(row, 5, QTableWidgetItem(f"{r.get('final_aep_gwh', 0):.4f}"))
            self._table.setItem(row, 6, QTableWidgetItem(f"{r.get('capacity_factor_pct', 0):.1f}"))

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------

    def _preview_aep(self, wtg_results):
        if not wtg_results:
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        names = [r.get('name', '') for r in wtg_results]
        gross = [r.get('gross_aep_gwh', 0) for r in wtg_results]
        net = [r.get('net_aep_gwh', 0) for r in wtg_results]
        final = [r.get('final_aep_gwh', 0) for r in wtg_results]

        x = np.arange(len(names))
        width = 0.25

        bars1 = ax.bar(x - width, gross, width, label='Gross AEP', color='#4CAF50', alpha=0.8)
        bars2 = ax.bar(x, net, width, label='Net AEP', color='#FF9800', alpha=0.8)
        bars3 = ax.bar(x + width, final, width, label='Final AEP', color='#2196F3', alpha=0.8)

        ax.set_xlabel("Turbine", color='white')
        ax.set_ylabel("AEP (GWh/yr)", color='white')
        ax.set_title("Annual Energy Production per Turbine", color='white', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.tick_params(colors='white')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_facecolor('#2d2d30')
        self._figure.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_csv(self):
        if not self._aep_results:
            QMessageBox.information(self, "No Data", "Please calculate AEP first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "aep_results.csv", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            from src.utils.data_utils import export_results_to_csv
            export_results_to_csv(self._aep_results, path)
            self._main_window.update_status(f"AEP results exported to {path}")
            QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    def export_to_pdf(self):
        """Generate a simple PDF report of the AEP results."""
        if not self._aep_results:
            QMessageBox.information(self, "No Data", "Please calculate AEP first.")
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            path, _ = QFileDialog.getSaveFileName(
                self, "Export PDF Report", "aep_report.pdf",
                "PDF Files (*.pdf);;All Files (*)")
            if not path:
                return

            summary = self._aep_results.get('summary', {})
            wtg_results = self._aep_results.get('wtg_results', [])

            with PdfPages(path) as pdf:
                # Page 1: Summary
                fig1 = Figure(figsize=(8.5, 11), dpi=100)
                fig1.patch.set_facecolor('white')
                ax1 = fig1.add_subplot(111)
                ax1.axis('off')

                title_text = (
                    f"WindFarm Designer Pro \u2014 AEP Report\n"
                    f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                )
                summary_text = (
                    f"Farm Summary\n"
                    f"{'='*40}\n"
                    f"Turbine Model: {summary.get('turbine_model', 'N/A')}\n"
                    f"Number of Turbines: {summary.get('n_turbines', 0)}\n"
                    f"Installed Capacity: {summary.get('installed_capacity_mw', 0):.1f} MW\n"
                    f"Gross AEP: {summary.get('total_gross_aep_gwh', 0):.2f} GWh/yr\n"
                    f"Wake Loss: {summary.get('total_wake_loss_pct', 0):.1f} %\n"
                    f"Net AEP: {summary.get('total_net_aep_gwh', 0):.2f} GWh/yr\n"
                    f"Total AEP (with losses): {summary.get('total_aep_gwh', 0):.2f} GWh/yr\n"
                    f"Capacity Factor: {summary.get('capacity_factor_pct', 0):.1f} %\n"
                    f"Full Load Hours: {summary.get('full_load_hours', 0):.0f} h/yr\n"
                )
                ax1.text(0.05, 0.95, title_text + summary_text,
                         transform=ax1.transAxes, fontsize=10,
                         verticalalignment='top', fontfamily='monospace')
                pdf.savefig(fig1)

                # Page 2: Bar chart
                if wtg_results:
                    fig2 = Figure(figsize=(8.5, 6), dpi=100)
                    fig2.patch.set_facecolor('white')
                    ax2 = fig2.add_subplot(111)

                    names = [r['name'] for r in wtg_results]
                    final_aep = [r.get('final_aep_gwh', 0) for r in wtg_results]
                    ax2.bar(range(len(names)), final_aep, color='steelblue')
                    ax2.set_xticks(range(len(names)))
                    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
                    ax2.set_ylabel("Final AEP (GWh/yr)")
                    ax2.set_title("Annual Energy Production per Turbine")
                    ax2.grid(axis='y', alpha=0.3)
                    fig2.tight_layout()
                    pdf.savefig(fig2)

            self._main_window.update_status(f"PDF report exported to {path}")
            QMessageBox.information(self, "Exported", f"PDF report saved to:\n{path}")
        except ImportError:
            QMessageBox.warning(self, "Missing Library",
                                "matplotlib.backends.backend_pdf is required for PDF export.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"PDF export failed:\n{e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_wtg_positions(self, data):
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

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_from_project(self):
        data = self._main_window.get_project_data()
        aep_results = data.get('aep_results')
        if aep_results:
            self._aep_results = aep_results
            summary = aep_results.get('summary', {})
            self._summary_n_turb.setText(str(summary.get('n_turbines', 0)))
            self._summary_capacity.setText(f"{summary.get('installed_capacity_mw', 0):.1f} MW")
            self._summary_gross_aep.setText(f"{summary.get('total_gross_aep_gwh', 0):.2f} GWh/yr")
            self._summary_net_aep.setText(f"{summary.get('total_net_aep_gwh', 0):.2f} GWh/yr")
            self._summary_wake_loss.setText(f"{summary.get('total_wake_loss_pct', 0):.1f} %")
            self._summary_cf.setText(f"{summary.get('capacity_factor_pct', 0):.1f} %")
            self._summary_flh.setText(f"{summary.get('full_load_hours', 0):.0f} h/yr")
            self._populate_table(aep_results.get('wtg_results', []))
            self._preview_aep(aep_results.get('wtg_results', []))
