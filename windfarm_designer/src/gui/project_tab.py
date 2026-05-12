"""
WindFarm Designer Pro - Project Setup Tab.
Provides fields for defining the project name, location, boundary,
turbine selection, capacity target, and file imports for mast / WTG data.
Includes an interactive map (folium + QWebEngineView) for boundary selection.
"""

import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton,
    QComboBox, QGroupBox, QFileDialog, QMessageBox, QScrollArea,
    QCheckBox, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QUrl

# Graceful import for QWebEngineView (may not be installed)
_WEBENGINE_AVAILABLE = False
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWebChannel import QWebChannel
    _WEBENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = None  # type: ignore[assignment,misc]
    QWebChannel = None  # type: ignore[assignment,misc]

from src.utils.data_utils import (
    get_turbine_names, get_turbine_spec, generate_default_power_curve,
    load_boundary, load_wtg_layout, load_mast_data,
)
from src.utils.geo_utils import bbox_to_polygon, compute_area_km2

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# JavaScript <-> Python bridge for interactive map
# ------------------------------------------------------------------

class MapBridge(QObject):
    """Bridge object exposed to JavaScript in the QWebEngineView.

    The JavaScript side calls ``window.pybridge.sendBoundary(coords)``
    which triggers the ``boundarySelected`` signal on the Python side.
    """
    boundarySelected = pyqtSignal(list)

    @pyqtSlot(list)
    def sendBoundary(self, coords):
        self.boundarySelected.emit(coords)


# ------------------------------------------------------------------
# HTML template for the Leaflet + Leaflet.Draw interactive map
# ------------------------------------------------------------------

_MAP_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
    <style>
        html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
    </style>
</head>
<body>
    <div id="map" style="width:100%;height:100%;"></div>
    <script>
        var map = L.map('map').setView([20, 0], 3);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        var drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);
        var drawControl = new L.Control.Draw({{
            draw: {{
                polygon: true,
                polyline: false,
                circle: false,
                marker: false,
                circlemarker: false,
                rectangle: false
            }},
            edit: {{ featureGroup: drawnItems }}
        }});
        map.addControl(drawControl);
        map.on(L.Draw.Event.CREATED, function(e) {{
            drawnItems.clearLayers();
            drawnItems.addLayer(e.layer);
            var coords = e.layer.getLatLngs()[0].map(function(ll) {{
                return [ll.lat, ll.lng];
            }});
            window.pybridge.sendBoundary(coords);
        }});
        function searchLocation(query) {{
            fetch('https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(query))
            .then(function(r) {{ return r.json(); }})
            .then(function(data) {{
                if (data.length > 0) {{
                    map.setView([parseFloat(data[0].lat), parseFloat(data[0].lon)], 13);
                }}
            }});
        }}
    </script>
</body>
</html>"""


class ProjectTab(QWidget):
    """First workflow tab: define the wind farm project parameters."""

    project_created = pyqtSignal()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._boundary_file_path = ""
        self._wtg_file_path = ""
        self._mast_file_path = ""
        self._map_boundary = None  # List of [lat, lon] from map
        self._webview = None  # QWebEngineView instance (if available)
        self._map_bridge = None  # MapBridge instance
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(10)

        # --- Row 1: Project Name + Location Search ---
        row1_layout = QHBoxLayout()

        name_group = QGroupBox("Project Information")
        name_form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. My Wind Farm")
        name_form.addRow("Project Name:", self._name_edit)
        name_group.setLayout(name_form)
        row1_layout.addWidget(name_group, 1)

        search_group = QGroupBox("Location Search")
        search_layout = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search for a location...")
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self._search_location)
        search_layout.addWidget(self._search_edit, 1)
        search_layout.addWidget(btn_search)
        search_group.setLayout(search_layout)
        row1_layout.addWidget(search_group, 1)

        main_layout.addLayout(row1_layout)

        # --- Row 2: Interactive Map (or fallback coords) ---
        map_group = QGroupBox("Boundary Definition (Interactive Map)")
        map_layout = QVBoxLayout()

        if _WEBENGINE_AVAILABLE:
            # Interactive map with QWebEngineView
            self._map_bridge = MapBridge(self)
            self._map_bridge.boundarySelected.connect(self._on_boundary_selected)

            self._webview = QWebEngineView()
            self._webview.setMinimumHeight(400)

            # Set up WebChannel
            self._webview.page().setWebChannel(self._webview.page().webChannel())
            channel = QWebChannel(self._webview.page())
            channel.registerObject("pybridge", self._map_bridge)
            self._webview.page().setWebChannel(channel)

            self._webview.setHtml(_MAP_HTML_TEMPLATE)
            map_layout.addWidget(self._webview)

            self._boundary_info_label = QLabel(
                "Draw a polygon on the map to define the wind farm boundary.\n"
                "Use the polygon tool (left toolbar on map) to draw a boundary.")
            self._boundary_info_label.setWordWrap(True)
            map_layout.addWidget(self._boundary_info_label)

            # Manual lat/lon display (read-only, updated from map)
            coord_display = QHBoxLayout()
            self._lat_display = QLabel("Lat: ---")
            self._lon_display = QLabel("Lon: ---")
            self._vertex_count_label = QLabel("Vertices: ---")
            coord_display.addWidget(self._lat_display)
            coord_display.addWidget(self._lon_display)
            coord_display.addWidget(self._vertex_count_label)
            map_layout.addLayout(coord_display)

            # Secondary option: import boundary from file
            self._show_file_import_toggle = QPushButton(
                "Or import boundary from file\u2026")
            self._show_file_import_toggle.setFlat(True)
            self._show_file_import_toggle.clicked.connect(
                self._toggle_file_import_section)
            map_layout.addWidget(self._show_file_import_toggle)

            # Hidden file import section
            self._file_import_widget = QWidget()
            file_import_layout = QVBoxLayout(self._file_import_widget)
            file_import_layout.setContentsMargins(0, 0, 0, 0)
            file_row = QHBoxLayout()
            self._boundary_file_label = QLabel("No file selected")
            self._boundary_file_label.setWordWrap(True)
            btn_boundary = QPushButton("Browse\u2026")
            btn_boundary.clicked.connect(self._browse_boundary)
            file_row.addWidget(self._boundary_file_label, 1)
            file_row.addWidget(btn_boundary)
            file_import_layout.addLayout(file_row)
            self._file_import_widget.setVisible(False)
            map_layout.addWidget(self._file_import_widget)

        else:
            # Fallback: manual coordinate entry
            self._boundary_info_label = QLabel(
                "QWebEngineView is not available.\n"
                "Please define the boundary manually using the fields below.")
            self._boundary_info_label.setWordWrap(True)
            map_layout.addWidget(self._boundary_info_label)

            # Option A: import file
            file_row = QHBoxLayout()
            self._boundary_file_label = QLabel("No file selected")
            self._boundary_file_label.setWordWrap(True)
            btn_boundary = QPushButton("Browse\u2026")
            btn_boundary.clicked.connect(self._browse_boundary)
            file_row.addWidget(self._boundary_file_label, 1)
            file_row.addWidget(btn_boundary)
            map_layout.addLayout(file_row)

            # Option B: set from 4 coords
            coord_group = QGroupBox("Set Boundary from Coordinates")
            coord_layout = QGridLayout()

            self._lat_min = QDoubleSpinBox()
            self._lat_min.setRange(-90, 90)
            self._lat_min.setDecimals(6)
            self._lat_max = QDoubleSpinBox()
            self._lat_max.setRange(-90, 90)
            self._lat_max.setDecimals(6)
            self._lon_min = QDoubleSpinBox()
            self._lon_min.setRange(-180, 180)
            self._lon_min.setDecimals(6)
            self._lon_max = QDoubleSpinBox()
            self._lon_max.setRange(-180, 180)
            self._lon_max.setDecimals(6)

            coord_layout.addWidget(QLabel("Lat min:"), 0, 0)
            coord_layout.addWidget(self._lat_min, 0, 1)
            coord_layout.addWidget(QLabel("Lat max:"), 0, 2)
            coord_layout.addWidget(self._lat_max, 0, 3)
            coord_layout.addWidget(QLabel("Lon min:"), 1, 0)
            coord_layout.addWidget(self._lon_min, 1, 1)
            coord_layout.addWidget(QLabel("Lon max:"), 1, 2)
            coord_layout.addWidget(self._lon_max, 1, 3)

            btn_set_boundary = QPushButton("Set Boundary from Coords")
            btn_set_boundary.clicked.connect(self._set_boundary_from_coords)
            coord_layout.addWidget(btn_set_boundary, 2, 0, 1, 4)
            coord_group.setLayout(coord_layout)
            map_layout.addWidget(coord_group)

            self._file_import_widget = None
            self._lat_display = None
            self._lon_display = None
            self._vertex_count_label = None

        map_group.setLayout(map_layout)
        main_layout.addWidget(map_group)

        # --- Row 3: Capacity, Turbine, Buffer (compact horizontal) ---
        params_group = QGroupBox("Turbine & Capacity Settings")
        params_layout = QHBoxLayout()

        params_form = QFormLayout()

        self._capacity_spin = QDoubleSpinBox()
        self._capacity_spin.setRange(1.0, 10000.0)
        self._capacity_spin.setDecimals(1)
        self._capacity_spin.setSuffix(" MW")
        self._capacity_spin.setValue(100.0)
        params_form.addRow("Capacity:", self._capacity_spin)

        self._turbine_combo = QComboBox()
        names = get_turbine_names()
        self._turbine_combo.addItems(names)
        params_form.addRow("Turbine:", self._turbine_combo)

        self._buffer_spin = QDoubleSpinBox()
        self._buffer_spin.setRange(0.0, 200.0)
        self._buffer_spin.setDecimals(1)
        self._buffer_spin.setSuffix(" km")
        self._buffer_spin.setValue(20.0)
        params_form.addRow("Buffer:", self._buffer_spin)

        params_layout.addLayout(params_form)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # --- Row 4: Import options (WTG layout, Mast data) ---
        import_group = QGroupBox("Data Import Options")
        import_layout = QVBoxLayout()

        wtg_row = QHBoxLayout()
        self._wtg_label = QLabel("No WTG layout file")
        self._wtg_label.setWordWrap(True)
        btn_wtg = QPushButton("Import WTG Layout\u2026")
        btn_wtg.clicked.connect(self._browse_wtg)
        wtg_row.addWidget(self._wtg_label, 1)
        wtg_row.addWidget(btn_wtg)
        import_layout.addLayout(wtg_row)

        mast_row = QHBoxLayout()
        self._mast_label = QLabel("No mast data file")
        self._mast_label.setWordWrap(True)
        btn_mast = QPushButton("Import Mast Data\u2026")
        btn_mast.clicked.connect(self._browse_mast)
        mast_row.addWidget(self._mast_label, 1)
        mast_row.addWidget(btn_mast)
        import_layout.addLayout(mast_row)

        import_group.setLayout(import_layout)
        main_layout.addWidget(import_group)

        # --- Row 5: Create Project Button ---
        self._create_btn = QPushButton("\U0001F680  Create Project")
        self._create_btn.setStyleSheet(
            "QPushButton { font-size: 14px; padding: 8px; font-weight: bold; }"
        )
        self._create_btn.clicked.connect(self._create_project)
        main_layout.addWidget(self._create_btn)

        # --- Row 6: Project Summary ---
        summary_group = QGroupBox("Project Summary")
        summary_layout = QFormLayout()
        self._summary_name = QLabel("\u2014")
        self._summary_location = QLabel("\u2014")
        self._summary_area = QLabel("\u2014")
        self._summary_capacity = QLabel("\u2014")
        self._summary_turbine = QLabel("\u2014")
        self._summary_n_wtg = QLabel("\u2014")
        summary_layout.addRow("Name:", self._summary_name)
        summary_layout.addRow("Location:", self._summary_location)
        summary_layout.addRow("Farm Area:", self._summary_area)
        summary_layout.addRow("Target Capacity:", self._summary_capacity)
        summary_layout.addRow("Turbine:", self._summary_turbine)
        summary_layout.addRow("WTGs Imported:", self._summary_n_wtg)
        summary_group.setLayout(summary_layout)
        main_layout.addWidget(summary_group)

        main_layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Map interaction helpers
    # ------------------------------------------------------------------

    def _search_location(self):
        """Run the JavaScript ``searchLocation`` function on the map."""
        if not _WEBENGINE_AVAILABLE or not self._webview:
            QMessageBox.information(
                self, "Not Available",
                "Interactive map is not available. "
                "Please install PyQtWebEngine to use this feature.")
            return
        query = self._search_edit.text().strip()
        if not query:
            return
        js = f"searchLocation('{query.replace(chr(39), chr(92) + chr(39))}');"
        self._webview.page().runJavaScript(js)
        self._main_window.update_status(f"Searching for: {query}")

    def _on_boundary_selected(self, coords):
        """Called when the user draws a polygon on the map."""
        if not coords or len(coords) < 3:
            QMessageBox.warning(
                self, "Invalid Boundary",
                "Please draw a polygon with at least 3 vertices.")
            return

        # Store as list of (lat, lon) tuples
        self._map_boundary = [(c[0], c[1]) for c in coords]

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Update display labels
        if self._lat_display:
            self._lat_display.setText(
                f"Lat: {min(lats):.6f} to {max(lats):.6f} (center {center_lat:.6f})")
        if self._lon_display:
            self._lon_display.setText(
                f"Lon: {min(lons):.6f} to {max(lons):.6f} (center {center_lon:.6f})")
        if self._vertex_count_label:
            self._vertex_count_label.setText(f"Vertices: {len(coords)}")

        self._boundary_info_label.setText(
            f"Boundary defined with {len(coords)} vertices on the map.\n"
            f"Center: {center_lat:.4f}\u00b0, {center_lon:.4f}\u00b0")

        self._main_window.update_status(
            f"Boundary selected on map: {len(coords)} vertices, "
            f"center at ({center_lat:.4f}, {center_lon:.4f})")

    def _toggle_file_import_section(self):
        """Show/hide the file import section (secondary option)."""
        if self._file_import_widget:
            visible = not self._file_import_widget.isVisible()
            self._file_import_widget.setVisible(visible)
            if visible:
                self._show_file_import_toggle.setText("Hide file import option")
            else:
                self._show_file_import_toggle.setText(
                    "Or import boundary from file\u2026")

    # ------------------------------------------------------------------
    # File browsers
    # ------------------------------------------------------------------

    def _browse_boundary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Boundary", "",
            "Supported Files (*.csv *.geojson *.json *.kml);;CSV (*.csv);;"
            "GeoJSON (*.geojson *.json);;KML (*.kml);;All Files (*)")
        if path:
            self._boundary_file_path = path
            self._boundary_file_label.setText(path)
            self._try_load_boundary(path)

    def _browse_wtg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import WTG Layout", "",
            "CSV Files (*.csv);;All Files (*)")
        if path:
            self._wtg_file_path = path
            self._wtg_label.setText(path)

    def _browse_mast(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Mast Data", "",
            "CSV Files (*.csv);;All Files (*)")
        if path:
            self._mast_file_path = path
            self._mast_label.setText(path)

    # ------------------------------------------------------------------
    # Boundary handling
    # ------------------------------------------------------------------

    def _try_load_boundary(self, filepath: str):
        try:
            boundary = load_boundary(filepath)
            if boundary and len(boundary) >= 3:
                lats = [p[0] for p in boundary]
                lons = [p[1] for p in boundary]
                # Update map display labels
                if self._lat_display:
                    self._lat_display.setText(
                        f"Lat: {min(lats):.6f} to {max(lats):.6f}")
                if self._lon_display:
                    self._lon_display.setText(
                        f"Lon: {min(lons):.6f} to {max(lons):.6f}")
                if self._vertex_count_label:
                    self._vertex_count_label.setText(
                        f"Vertices: {len(boundary)}")
                # Also update fallback coordinate spinboxes
                if hasattr(self, '_lat_min'):
                    self._lat_min.setValue(min(lats))
                    self._lat_max.setValue(max(lats))
                    self._lon_min.setValue(min(lons))
                    self._lon_max.setValue(max(lons))
                self._boundary_info_label.setText(
                    f"Boundary loaded from file: {len(boundary)} vertices")
                self._main_window.update_status(
                    f"Boundary loaded: {len(boundary)} vertices from {filepath}")
            else:
                QMessageBox.warning(self, "Boundary Error",
                                    "Boundary file contains fewer than 3 valid points.")
        except Exception as e:
            QMessageBox.critical(self, "Boundary Error",
                                 f"Failed to load boundary:\n{e}")
            logger.exception("Boundary load failed")

    def _set_boundary_from_coords(self):
        """Fallback: set boundary from manual lat/lon spinboxes."""
        lat_min = self._lat_min.value()
        lat_max = self._lat_max.value()
        lon_min = self._lon_min.value()
        lon_max = self._lon_max.value()

        if lat_min >= lat_max or lon_min >= lon_max:
            QMessageBox.warning(self, "Invalid Coordinates",
                                "Lat min must be < Lat max and Lon min < Lon max.")
            return

        boundary = [
            (lat_min, lon_min), (lat_min, lon_max),
            (lat_max, lon_max), (lat_max, lon_min),
        ]
        self._boundary_file_label.setText("Boundary set from coordinates (rectangular)")
        self._main_window.update_status(
            f"Boundary set from coords: [{lat_min:.4f}, {lon_min:.4f}, "
            f"{lat_max:.4f}, {lon_max:.4f}]")

    # ------------------------------------------------------------------
    # Create Project
    # ------------------------------------------------------------------

    def _create_project(self):
        """Validate inputs and store everything into project data."""
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Please enter a project name.")
            return

        capacity = self._capacity_spin.value()
        turbine_name = self._turbine_combo.currentText()
        buffer_km = self._buffer_spin.value()
        turbine_spec = get_turbine_spec(turbine_name)

        if not turbine_spec:
            QMessageBox.warning(self, "Validation", "Please select a valid turbine model.")
            return

        # Determine boundary (priority: map > file > manual coords)
        boundary = None

        # 1. Map-drawn boundary (highest priority)
        if self._map_boundary and len(self._map_boundary) >= 3:
            boundary = list(self._map_boundary)

        # 2. Imported file boundary
        if boundary is None and self._boundary_file_path:
            try:
                boundary = load_boundary(self._boundary_file_path)
            except Exception:
                QMessageBox.warning(self, "Error", "Could not re-load boundary file.")
                return

        # 3. Manual coordinate entry (fallback only)
        if boundary is None and hasattr(self, '_lat_min'):
            lat_min = self._lat_min.value()
            lat_max = self._lat_max.value()
            lon_min = self._lon_min.value()
            lon_max = self._lon_max.value()
            if lat_min < lat_max and lon_min < lon_max:
                boundary = [
                    (lat_min, lon_min), (lat_min, lon_max),
                    (lat_max, lon_max), (lat_max, lon_min),
                ]

        if boundary is None or len(boundary) < 3:
            QMessageBox.warning(self, "Validation",
                                "Please define a boundary (draw on map, import file, "
                                "or enter coordinates).")
            return

        # Compute bbox and center from boundary
        lats = [p[0] for p in boundary]
        lons = [p[1] for p in boundary]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Compute area
        try:
            polygon = bbox_to_polygon(bbox)
            area = compute_area_km2(polygon)
        except Exception:
            area = 0.0

        # Generate power curve
        power_curve = generate_default_power_curve(turbine_spec)

        # Load WTG layout if provided
        wtg_layout = None
        if self._wtg_file_path:
            try:
                wtg_layout = load_wtg_layout(self._wtg_file_path)
            except Exception as e:
                self._main_window.update_status(
                    f"Warning: could not load WTG layout: {e}")

        # Load mast data if provided
        mast_data = None
        if self._mast_file_path:
            try:
                mast_data = load_mast_data(self._mast_file_path)
            except Exception as e:
                self._main_window.update_status(
                    f"Warning: could not load mast data: {e}")

        # Store in project data
        data = self._main_window.get_project_data()
        data['project_name'] = name
        data['center_lat'] = center_lat
        data['center_lon'] = center_lon
        data['boundary'] = boundary
        data['bbox'] = bbox
        data['capacity_mw'] = capacity
        data['turbine_model'] = turbine_name
        data['turbine_spec'] = turbine_spec
        data['power_curve'] = power_curve
        data['buffer_km'] = buffer_km
        data['wtg_layout'] = wtg_layout
        data['mast_data'] = mast_data

        # Update summary labels
        self._summary_name.setText(name)
        self._summary_location.setText(
            f"{center_lat:.4f}\u00b0, {center_lon:.4f}\u00b0")
        self._summary_area.setText(
            f"{area:.1f} km\u00b2" if area > 0 else "\u2014")
        self._summary_capacity.setText(f"{capacity:.0f} MW")
        self._summary_turbine.setText(
            f"{turbine_name} ({turbine_spec['rated_power_kw']} kW, "
            f"D={turbine_spec['rotor_diameter_m']}m)")
        self._summary_n_wtg.setText(
            str(len(wtg_layout)) if wtg_layout
            else "None \u2013 use Layout Optimizer")

        self._main_window.update_status(
            f"Project '{name}' created: {capacity} MW, "
            f"{len(boundary)}-vertex boundary, area {area:.1f} km\u00b2")
        self.project_created.emit()

        QMessageBox.information(self, "Project Created",
                                f"Project '{name}' has been created successfully.\n\n"
                                f"Capacity: {capacity} MW\n"
                                f"Turbine: {turbine_name}\n"
                                f"Area: {area:.1f} km\u00b2\n\n"
                                f"Proceed to the Terrain Data tab.")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_from_project(self):
        """Reload widget values from the central project data store."""
        data = self._main_window.get_project_data()
        name = data.get('project_name', '')
        if name:
            self._name_edit.setText(name)
        if data.get('center_lat'):
            center_lat = data['center_lat']
        else:
            center_lat = 0.0
        if data.get('center_lon'):
            center_lon = data['center_lon']
        else:
            center_lon = 0.0
        if data.get('capacity_mw'):
            self._capacity_spin.setValue(data['capacity_mw'])
        if data.get('buffer_km'):
            self._buffer_spin.setValue(data['buffer_km'])
        if data.get('turbine_model'):
            idx = self._turbine_combo.findText(data['turbine_model'])
            if idx >= 0:
                self._turbine_combo.setCurrentIndex(idx)

        boundary = data.get('boundary')
        bbox = data.get('bbox')
        if bbox:
            if hasattr(self, '_lat_min'):
                self._lat_min.setValue(bbox[1])
                self._lat_max.setValue(bbox[3])
                self._lon_min.setValue(bbox[0])
                self._lon_max.setValue(bbox[2])
            if self._lat_display and boundary:
                lats = [p[0] for p in boundary]
                lons = [p[1] for p in boundary]
                self._lat_display.setText(
                    f"Lat: {min(lats):.6f} to {max(lats):.6f} "
                    f"(center {center_lat:.6f})")
                self._lon_display.setText(
                    f"Lon: {min(lons):.6f} to {max(lons):.6f} "
                    f"(center {center_lon:.6f})")
                if self._vertex_count_label:
                    self._vertex_count_label.setText(
                        f"Vertices: {len(boundary)}")
                self._boundary_info_label.setText(
                    f"Boundary loaded: {len(boundary)} vertices")
