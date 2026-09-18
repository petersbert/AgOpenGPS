import sys
import os
import pandas as pd
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import configparser
import math
from scipy.spatial import ConvexHull 
from datetime import datetime

# --- PDF LIBRARIES ---
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLineEdit, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QSpacerItem, QSizePolicy,
                             QGroupBox, QListWidget, QListWidgetItem, QTabWidget,
                             QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QGridLayout,
                             QScrollArea, QSplitter, QInputDialog)
from PyQt5.QtCore import Qt, QLocale, QSize
from PyQt5.QtGui import QDoubleValidator, QFont

# --- CONFIGURATION ---
CONFIG_FILE = "terrain_analyzer_config.ini"

def save_last_csv_path(path):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'last_csv_path': path}
    try:
        with open(CONFIG_FILE, 'w') as cf:
            config.write(cf)
    except IOError:
        pass

def load_last_csv_path():
    if os.path.exists(CONFIG_FILE):
        config = configparser.ConfigParser()
        try:
            config.read(CONFIG_FILE)
            return config['DEFAULT'].get('last_csv_path', '')
        except Exception:
            pass
    return ''

# --- TACTILE NUMERIC PAD ---
class NumPad(QDialog):
    def __init__(self, initial_value="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Input")
        self.setFixedSize(320, 450)
        
        layout = QVBoxLayout(self)
        self.display = QLineEdit(initial_value)
        self.display.setFixedHeight(50)
        self.display.setFont(QFont("Arial", 20, QFont.Bold))
        self.display.setAlignment(Qt.AlignRight)
        layout.addWidget(self.display)
        
        grid = QGridLayout(); layout.addLayout(grid)
        btn_style = "QPushButton { font-size: 20px; font-weight: bold; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; min-width: 60px; min-height: 60px; } QPushButton:pressed { background-color: #d0d0d0; }"
        
        keys = [('7', 0, 0), ('8', 0, 1), ('9', 0, 2), ('4', 1, 0), ('5', 1, 1), ('6', 1, 2), ('1', 2, 0), ('2', 2, 1), ('3', 2, 2), ('-', 3, 0), ('0', 3, 1), ('.', 3, 2)]
        for text, r, c in keys:
            btn = QPushButton(text); btn.setStyleSheet(btn_style); btn.clicked.connect(lambda ch, t=text: self.on_digit(t)); grid.addWidget(btn, r, c)
            
        btn_bksp = QPushButton("⌫"); btn_bksp.setStyleSheet("background-color: #ffccbc; font-size: 20px; font-weight: bold; border-radius: 5px; min-height: 60px;")
        btn_bksp.clicked.connect(self.on_backspace); grid.addWidget(btn_bksp, 0, 3)
        btn_clr = QPushButton("C"); btn_clr.setStyleSheet("background-color: #ffab91; font-size: 20px; font-weight: bold; border-radius: 5px; min-height: 60px;")
        btn_clr.clicked.connect(self.on_clear); grid.addWidget(btn_clr, 1, 3)
        btn_ok = QPushButton("Accept"); btn_ok.setStyleSheet("background-color: #c8e6c9; font-size: 20px; font-weight: bold; border-radius: 5px; min-height: 130px;")
        btn_ok.clicked.connect(self.accept); grid.addWidget(btn_ok, 2, 3, 2, 1)

    def on_digit(self, digit): self.display.setText(self.display.text() + digit)
    def on_backspace(self): self.display.setText(self.display.text()[:-1])
    def on_clear(self): self.display.clear()
    def text(self): return self.display.text()

class TouchLineEdit(QLineEdit):
    def __init__(self, text="", parent=None): super().__init__(text, parent)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            dlg = NumPad(self.text(), self)
            if dlg.exec_() == QDialog.Accepted: self.setText(dlg.text()); self.editingFinished.emit()
        super().mousePressEvent(event)

# --- DATA LOADING ---
def load_gnss_data(csv_path):
    try:
        skip_lines = 0
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                line_lower = line.lower()
                if 'latitude' in line_lower or 'easting' in line_lower:
                    skip_lines = i
                    break

        df = pd.read_csv(csv_path, skiprows=skip_lines)
        df.columns = df.columns.str.strip()

        rename_dict = {}
        for col in df.columns:
            c_low = col.lower()
            if c_low == 'latitude': rename_dict[col] = 'Latitude'
            if c_low == 'longitude': rename_dict[col] = 'Longitude'
            if c_low in ['altitude', 'elevation']: rename_dict[col] = 'Elevation'
            if c_low == 'easting': rename_dict[col] = 'Easting'
            if c_low == 'northing': rename_dict[col] = 'Northing'
            
        df = df.rename(columns=rename_dict)

        req = ['Latitude', 'Longitude', 'Elevation']
        if not all(c in df.columns for c in req):
            if len(df.columns) >= 3:
                df_s = df.iloc[:, :3].copy()
                df_s.columns = req
                df = df_s
                
        for c in req: df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(subset=req, inplace=True)
        df = df[abs(df['Elevation']) > 0.1] 
        
        mean = df['Elevation'].mean()
        std = df['Elevation'].std()
        df = df[(df['Elevation'] >= mean - 3*std) & (df['Elevation'] <= mean + 3*std)]
        
        return df
    except Exception as e: 
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_flags_data(txt_path):
    try:
        data = []
        with open(txt_path, 'r') as f: lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('$') or len(line) < 5: continue
            parts = line.split(',')
            if len(parts) >= 5:
                try: data.append({'Latitude': float(parts[0]), 'Longitude': float(parts[1]), 'Elevation': float(parts[4]), 'Label': parts[-1].strip() if len(parts) > 7 else "Flags"})
                except ValueError: continue
        return pd.DataFrame(data)
    except Exception as e: return pd.DataFrame()

def get_plane_z(x_points, y_points, line):
    P1 = np.array([line["x1"], line["y1"]])
    P2 = np.array([line["x2"], line["y2"]])
    Z1, Z2 = line["z1"], line["z2"]
    slope = line["cross_slope"]
    
    V = P2 - P1
    len_sq = np.dot(V, V)
    if len_sq < 1e-6: return np.zeros_like(x_points)
    
    t = (x_points - P1[0]) * V[0] + (y_points - P1[1]) * V[1]
    t /= len_sq
    Z_line = Z1 + t * (Z2 - Z1)
    
    length = np.sqrt(len_sq)
    signed_dist = ((x_points - P1[0]) * V[1] - (y_points - P1[1]) * V[0]) / length
    return Z_line + (signed_dist * slope)

def calculate_z_from_lines(x_points, y_points, lines):
    if not lines: return np.zeros_like(x_points)
    final_z = get_plane_z(x_points, y_points, lines[0])
    for i in range(1, len(lines)):
        prev_line = lines[i-1]
        curr_line = lines[i]
        c_prev = np.array([(prev_line["x1"]+prev_line["x2"])/2, (prev_line["y1"]+prev_line["y2"])/2])
        c_curr = np.array([(curr_line["x1"]+curr_line["x2"])/2, (curr_line["y1"]+curr_line["y2"])/2])
        flow_vec = c_curr - c_prev
        if np.dot(flow_vec, flow_vec) < 1e-6: continue
        curr_z = get_plane_z(x_points, y_points, curr_line)
        PX = x_points - c_curr[0]
        PY = y_points - c_curr[1]
        is_forward = (PX * flow_vec[0] + PY * flow_vec[1]) >= 0
        final_z[is_forward] = curr_z[is_forward]
    return final_z

# --- GUI ---
class TerrainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app_name = "Land Leveling & Grading"
        self.app_version = "4.2"
        self.setWindowTitle(f"{self.app_name} v{self.app_version}")
        self.setGeometry(100, 100, 1400, 950)
        self.gnss_df = pd.DataFrame(); self.flags_df = pd.DataFrame(); self.plot_data_df = pd.DataFrame(); self.flags_plot_df = pd.DataFrame()
        self.project_lines = []; self.current_line_index = -1
        self.line_keys = ["x1","y1","z1","x2","y2","z2","cross_slope"]
        self.temp_line_params = {k: 0.0 for k in self.line_keys}; self.temp_line_params["cross_slope"] = -0.002
        self.z_exaggeration = 5.0; self.z_mean = 0.0; self.current_colormap = 'terrain'
        self.terrain_actor = None; self.design_actor = None
        self.ridge_actors = []; self.label_actors = []; self.flag_actors = []; self.compass_actors = []
        self.max_min_actors = [] 
        self.picking_mode = None; self.pick_azimuth_pts = []
        self.stats_cache = {}
        
        self.setup_ui(); self.init_plot_view()

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(380) 
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        
        ctrl = QWidget(); ctrl_layout = QVBoxLayout(ctrl); ctrl_layout.setContentsMargins(10, 10, 20, 10)
        
        btn_layout = QHBoxLayout()
        self.btn_load_gnss = QPushButton("📂 Survey (TXT/CSV)"); self.btn_load_gnss.clicked.connect(self.open_gnss)
        self.btn_load_flags = QPushButton("🚩 Flags (TXT)"); self.btn_load_flags.clicked.connect(self.open_flags)
        btn_layout.addWidget(self.btn_load_gnss); btn_layout.addWidget(self.btn_load_flags)
        ctrl_layout.addLayout(btn_layout); ctrl_layout.addSpacing(10)
        
        val_coord = QDoubleValidator(-1e6, 1e6, 3, self); val_coord.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        val_slope = QDoubleValidator(-1000.0, 1000.0, 5, self); val_slope.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        val_z = QDoubleValidator(0.1, 100.0, 1, self); val_z.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

        # --- SECTION: CENTER AND NATURAL SLOPES ---
        gb_coords = QGroupBox("Field Center & Natural Slope")
        gb_coords.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        coords_lay = QVBoxLayout()
        
        self.le_center_coords = QLineEdit(); self.le_center_coords.setReadOnly(True)
        self.le_center_coords.setStyleSheet("font-size: 13px; padding: 4px; border: 1px solid #ccc; background: #f5f5f5;")
        coords_lay.addWidget(self.le_center_coords)

        grid_nat = QGridLayout()
        
        # Slope X (East-West)
        self.btn_calc_slope_x = QPushButton("📐 Calc Natural X")
        self.btn_calc_slope_x.clicked.connect(lambda: self.calculate_natural_slope('X'))
        self.le_nat_slope_x = TouchLineEdit("0.000")
        self.le_nat_slope_x.setValidator(val_slope)
        self.btn_apply_x = QPushButton("➡️ Use")
        self.btn_apply_x.setToolTip("Apply to current Cross Slope")
        self.btn_apply_x.clicked.connect(lambda: self.apply_nat_to_active('cross'))
        
        grid_nat.addWidget(self.btn_calc_slope_x, 0, 0)
        grid_nat.addWidget(self.le_nat_slope_x, 1, 0)
        grid_nat.addWidget(self.btn_apply_x, 1, 1)

        # Slope Y (North-South)
        self.btn_calc_slope_y = QPushButton("📐 Calc Natural Y")
        self.btn_calc_slope_y.clicked.connect(lambda: self.calculate_natural_slope('Y'))
        self.le_nat_slope_y = TouchLineEdit("0.000")
        self.le_nat_slope_y.setValidator(val_slope)
        self.btn_apply_y = QPushButton("➡️ Use")
        self.btn_apply_y.setToolTip("Apply to current Longitudinal Slope")
        self.btn_apply_y.clicked.connect(lambda: self.apply_nat_to_active('long'))
        
        grid_nat.addWidget(self.btn_calc_slope_y, 0, 2)
        grid_nat.addWidget(self.le_nat_slope_y, 1, 2)
        grid_nat.addWidget(self.btn_apply_y, 1, 3)

        # --- NEW BUTTON: LASER LEVELING ---
        self.btn_laser_level = QPushButton("🔦 LASER Leveling (Single Plane)")
        self.btn_laser_level.setStyleSheet("background-color: #ffcdd2; font-weight: bold; height: 30px; margin-top: 5px;")
        self.btn_laser_level.clicked.connect(self.apply_laser_leveling)
        grid_nat.addWidget(self.btn_laser_level, 2, 0, 1, 4)
        
        coords_lay.addLayout(grid_nat)
        gb_coords.setLayout(coords_lay); ctrl_layout.addWidget(gb_coords); ctrl_layout.addSpacing(10)
        
        # --- LINE MANAGEMENT ---
        gb_proj = QGroupBox("Line Management"); proj_lay = QVBoxLayout(); tabs = QTabWidget()
        
        tab_man = QWidget(); man_lay = QVBoxLayout(tab_man)
        hl_btns = QHBoxLayout()
        b_add = QPushButton("➕ New"); b_add.clicked.connect(self.new_line)
        b_del = QPushButton("🗑️ Delete"); b_del.clicked.connect(self.del_line)
        b_del_all = QPushButton("💣 Delete All")
        b_del_all.setStyleSheet("color: white; background-color: #d32f2f; font-weight: bold;")
        b_del_all.clicked.connect(self.del_all_lines)
        hl_btns.addWidget(b_add); hl_btns.addWidget(b_del); hl_btns.addWidget(b_del_all); man_lay.addLayout(hl_btns)
        
        b_reset_med = QPushButton("🎯 Reset Active Line")
        b_reset_med.setStyleSheet("background:#ffcc80; font-weight:bold;")
        b_reset_med.clicked.connect(self.new_line); man_lay.addWidget(b_reset_med)

        self.list_w = QListWidget(); self.list_w.setFixedHeight(100); self.list_w.currentItemChanged.connect(self.sel_line)
        man_lay.addWidget(self.list_w)
        
        b_bestfit = QPushButton("✨ Best Fit (from A-B Line)")
        b_bestfit.setStyleSheet("background:#b3e5fc; font-weight:bold; padding: 5px; margin: 5px;")
        b_bestfit.clicked.connect(self.auto_best_fit); man_lay.addWidget(b_bestfit)
        
        b_multi = QPushButton("🌊 Multi-Plane Auto (Smoothing)")
        b_multi.setStyleSheet("background:#81d4fa; font-weight:bold; padding: 5px; margin: 2px;")
        b_multi.clicked.connect(self.auto_multi_plane); man_lay.addWidget(b_multi)

        pk_lay = QHBoxLayout()
        b_pS = QPushButton("🟢 Pick Point A"); b_pS.setStyleSheet("font-weight: bold; color: darkgreen; background-color: #e8f5e9; padding: 5px;"); b_pS.clicked.connect(lambda: self.start_pick("line_start"))
        b_pE = QPushButton("🔴 Pick Point B"); b_pE.setStyleSheet("font-weight: bold; color: darkred; background-color: #ffebee; padding: 5px;"); b_pE.clicked.connect(lambda: self.start_pick("line_end"))
        pk_lay.addWidget(b_pS); pk_lay.addWidget(b_pE); man_lay.addLayout(pk_lay)
        
        self.le_params = {}
        grid = [("🟢 Point A - X:", "x1"), ("🟢 Point A - Y:", "y1"), ("🟢 Point A - Z (Elev):", "z1"), ("🔴 Point B - X:", "x2"), ("🔴 Point B - Y:", "y2"), ("🔴 Point B - Z (Elev):", "z2"), ("📐 Cross Slope (cm/m):", "cross_slope")]
        for lbl, k in grid:
            h = QHBoxLayout(); label_widget = QLabel(lbl)
            if "A" in lbl: label_widget.setStyleSheet("font-weight: bold; color: darkgreen;")
            elif "B" in lbl: label_widget.setStyleSheet("font-weight: bold; color: darkred;")
            else: label_widget.setStyleSheet("font-weight: bold;")
            h.addWidget(label_widget); le = TouchLineEdit(); le.setValidator(val_slope if "slope" in k else val_coord); le.editingFinished.connect(self.update_from_le); self.le_params[k] = le; h.addWidget(le); man_lay.addLayout(h)
        
        h_long = QHBoxLayout(); h_long.addWidget(QLabel("Long. Slope (cm/m):")); self.le_long_slope = TouchLineEdit("0.00"); self.le_long_slope.setValidator(val_slope); self.le_long_slope.setStyleSheet("font-weight: bold; color: #2e7d32; border: 2px solid #a5d6a7;"); self.le_long_slope.editingFinished.connect(self.update_z2_from_slope); h_long.addWidget(self.le_long_slope); man_lay.addLayout(h_long)

        self.gb_stats = QGroupBox("Line Info"); self.gb_stats.setStyleSheet("QGroupBox { font-weight: bold; color: #2e7d32; }"); stats_lay = QVBoxLayout(); self.lbl_stats_len = QLabel("Length: 0.00 m"); stats_lay.addWidget(self.lbl_stats_len); self.gb_stats.setLayout(stats_lay); man_lay.addWidget(self.gb_stats)

        self.gb_vol_stats = QGroupBox("Balance & Volumes"); self.gb_vol_stats.setStyleSheet("QGroupBox { font-weight: bold; color: #d84315; }"); vol_lay = QVBoxLayout()
        self.lbl_area = QLabel("Area: -"); self.lbl_cut = QLabel("Cut: -"); self.lbl_fill = QLabel("Fill: -"); self.lbl_net = QLabel("Net: -"); self.lbl_max_cut = QLabel("Max Cut: -"); self.lbl_max_fill = QLabel("Max Fill: -"); self.lbl_sim = QLabel("Similarity (±5cm): -")
        font_b = QFont(); font_b.setBold(True)
        for l in [self.lbl_area, self.lbl_cut, self.lbl_fill, self.lbl_net, self.lbl_max_cut, self.lbl_max_fill, self.lbl_sim]: l.setFont(font_b); vol_lay.addWidget(l)
        self.gb_vol_stats.setLayout(vol_lay); man_lay.addWidget(self.gb_vol_stats); tabs.addTab(tab_man, "Manual")
        
        tab_auto = QWidget(); auto_lay = QVBoxLayout(tab_auto); auto_lay.addWidget(QLabel("Auto Generation")); auto_lay.addWidget(QLabel("Drain Spacing (m):")); self.le_auto_sp = TouchLineEdit("250.0"); self.le_auto_sp.setValidator(val_coord); auto_lay.addWidget(self.le_auto_sp); auto_lay.addWidget(QLabel("Height Diff (m):")); self.le_auto_dr = TouchLineEdit("0.08"); self.le_auto_dr.setValidator(val_slope); auto_lay.addWidget(self.le_auto_dr); auto_lay.addWidget(QLabel("Azimuth:"))
        h_az = QHBoxLayout(); self.le_auto_az = TouchLineEdit("0.0"); self.le_auto_az.setValidator(val_coord); h_az.addWidget(self.le_auto_az); b_pAz = QPushButton("📐 Pick"); b_pAz.clicked.connect(lambda: self.start_pick("pick_azimuth")); h_az.addWidget(b_pAz); auto_lay.addLayout(h_az); b_gen = QPushButton("⚙️ GENERATE"); b_gen.setStyleSheet("background:#c8e6c9; font-weight:bold"); b_gen.clicked.connect(self.auto_gen); auto_lay.addWidget(b_gen); auto_lay.addStretch(); tabs.addTab(tab_auto, "Auto")
        proj_lay.addWidget(tabs); gb_proj.setLayout(proj_lay); ctrl_layout.addWidget(gb_proj)

        gb_vis = QGroupBox("Visualization Options"); vis_lay = QVBoxLayout(); h_zx = QHBoxLayout(); h_zx.addWidget(QLabel("Z Exag:")); self.le_zx = TouchLineEdit("5.0"); self.le_zx.setValidator(val_z); h_zx.addWidget(self.le_zx); vis_lay.addLayout(h_zx)
        h_cam = QHBoxLayout(); b_rot_l = QPushButton("↺ 90°"); b_rot_l.clicked.connect(lambda: self.rotate_view(-90.0)); h_cam.addWidget(b_rot_l); b_top = QPushButton("Top"); b_top.clicked.connect(lambda: self.set_view_angle('top')); b_iso = QPushButton("Iso"); b_iso.clicked.connect(lambda: self.set_view_angle('iso')); b_front = QPushButton("South"); b_front.clicked.connect(lambda: self.set_view_angle('south')); b_side = QPushButton("East"); b_side.clicked.connect(lambda: self.set_view_angle('east')); h_cam.addWidget(b_top); h_cam.addWidget(b_iso); h_cam.addWidget(b_front); h_cam.addWidget(b_side); b_rot_r = QPushButton("↻ 90°"); b_rot_r.clicked.connect(lambda: self.rotate_view(90.0)); h_cam.addWidget(b_rot_r); vis_lay.addLayout(h_cam)
        self.chk_flags = QCheckBox("Show Flags"); self.chk_flags.setChecked(True); self.chk_flags.stateChanged.connect(lambda: self.update_plot(full=False)); vis_lay.addWidget(self.chk_flags); self.chk_compass = QCheckBox("Show Compass"); self.chk_compass.setChecked(True); self.chk_compass.stateChanged.connect(lambda: self.update_plot(full=False)); vis_lay.addWidget(self.chk_compass); b_upd = QPushButton("🔄 Refresh"); b_upd.clicked.connect(lambda: self.update_plot(full=True)); vis_lay.addWidget(b_upd); gb_vis.setLayout(vis_lay); ctrl_layout.addWidget(gb_vis)
        
        ctrl_layout.addStretch()
        export_row = QHBoxLayout()
        self.btn_export = QPushButton("💾 Exp AGD"); self.btn_export.setStyleSheet("background:#fff9c4; font-weight:bold; height: 35px;"); self.btn_export.clicked.connect(self.export_csv)
        self.btn_report = QPushButton("📄 Report"); self.btn_report.setStyleSheet("background:#e1f5fe; font-weight:bold; height: 35px;"); self.btn_report.clicked.connect(self.generate_pdf_report)
        self.btn_view_agd = QPushButton("👀 View AGD"); self.btn_view_agd.setStyleSheet("background:#e0f2f1; font-weight:bold; height: 35px;"); self.btn_view_agd.clicked.connect(self.view_agd_file)
        export_row.addWidget(self.btn_export); export_row.addWidget(self.btn_report); export_row.addWidget(self.btn_view_agd); ctrl_layout.addLayout(export_row)
        
        scroll_area.setWidget(ctrl); self.plotter = QtInteractor(parent=self); splitter.addWidget(scroll_area); splitter.addWidget(self.plotter); splitter.setSizes([480, 1000]); main_layout.addWidget(splitter)

    # --- NEW CALCULATION AND TRANSFER METHODS ---
    def calculate_natural_slope(self, axis='X'):
        if self.plot_data_df.empty:
            QMessageBox.warning(self, "Attention", "Load a survey first.")
            return
        coords = self.plot_data_df['X_m' if axis == 'X' else 'Y_m'].values
        elevations = self.plot_data_df['Elevation'].values
        if len(coords) < 2: return
        slope, intercept = np.polyfit(coords, elevations, 1)
        slope_cm_m = slope * 100.0 
        if axis == 'X': self.le_nat_slope_x.setText(f"{slope_cm_m:.3f}")
        else: self.le_nat_slope_y.setText(f"{slope_cm_m:.3f}")

    def apply_nat_to_active(self, target='cross'):
        try:
            if target == 'cross':
                val = self.le_nat_slope_x.text()
                self.le_params['cross_slope'].setText(val)
                self.le_params['cross_slope'].editingFinished.emit()
            else:
                val = self.le_nat_slope_y.text()
                self.le_long_slope.setText(val)
                self.le_long_slope.editingFinished.emit()
            QMessageBox.information(self, "Applied", f"Value {val} transferred to project.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not transfer: {e}")

    def apply_laser_leveling(self):
        """ Generates a single plane using calculated natural X and Y slopes """
        if self.plot_data_df.empty:
            QMessageBox.warning(self, "Attention", "Load a survey first.")
            return

        try:
            # 1. Get slopes from inputs (cm/m) and convert to m/m
            s_x = float(self.le_nat_slope_x.text().replace(',', '.')) / 100.0
            s_y = float(self.le_nat_slope_y.text().replace(',', '.')) / 100.0
            
            # 2. Calc geometric center and average elevation
            xm, xM = self.plot_data_df['X_m'].min(), self.plot_data_df['X_m'].max()
            ym, yM = self.plot_data_df['Y_m'].min(), self.plot_data_df['Y_m'].max()
            cx, cy = (xm + xM) / 2.0, (ym + yM) / 2.0
            z_avg = self.plot_data_df['Elevation'].mean()

            # 3. Define master line oriented North (Y-axis)
            # Pt A at center, Pt B 100m North
            dist_ref = 100.0
            x1, y1, z1 = cx, cy, z_avg
            x2, y2 = cx, cy + dist_ref
            z2 = z1 + (dist_ref * s_y) # Apply Y slope as longitudinal
            cross_slope = s_x            # Apply X slope as cross-slope

            laser_line = {
                "x1": x1, "y1": y1, "z1": z1,
                "x2": x2, "y2": y2, "z2": z2,
                "cross_slope": cross_slope
            }

            # 4. Replace project with this single plane
            self.project_lines = [laser_line]
            self.current_line_index = 0
            self.temp_line_params = laser_line.copy()

            # 5. Update UI and Plot
            self.update_list()
            self.refresh_le()
            self.update_plot(full=True)
            
            QMessageBox.information(self, "Laser Leveling", 
                f"Configured single plane based on natural slopes:\n"
                f"Longitudinal (Y): {s_y*100:.3f} cm/m\n"
                f"Cross Slope (X): {s_x*100:.3f} cm/m")

        except ValueError:
            QMessageBox.warning(self, "Error", "Calculate natural slopes first.")

    def set_view_angle(self, mode):
        if mode == 'top': self.plotter.view_xy()
        elif mode == 'iso': self.plotter.view_isometric(); self.plotter.camera.elevation = 15 
        elif mode == 'south': self.plotter.view_xz(); self.plotter.camera.elevation = 50 
        elif mode == 'east': self.plotter.view_yz(); self.plotter.camera.elevation = 50 
        self.plotter.reset_camera()

    def rotate_view(self, angle):
        pos = np.array(self.plotter.camera.position); foc = np.array(self.plotter.camera.focal_point); vec = (foc - pos) / np.linalg.norm(foc - pos)
        if abs(vec[2]) > 0.9: self.plotter.camera.roll -= angle
        else: self.plotter.camera.azimuth += angle
        self.plotter.update()

    def process_coordinates(self):
        if self.gnss_df.empty: self.le_center_coords.clear(); return
        lat_mean = self.gnss_df['Latitude'].mean(); lon_mean = self.gnss_df['Longitude'].mean(); ele_mean = self.gnss_df['Elevation'].mean()
        self.z_mean = ele_mean; self.le_center_coords.setText(f"{lat_mean:.8f}, {lon_mean:.8f}, {ele_mean:.3f}")
        self.plot_data_df = self.gnss_df.copy()
        if 'Easting' in self.plot_data_df.columns and 'Northing' in self.plot_data_df.columns: self.plot_data_df['X_m'] = self.plot_data_df['Easting']; self.plot_data_df['Y_m'] = self.plot_data_df['Northing']
        else:
            min_lon = self.plot_data_df['Longitude'].min(); min_lat = self.plot_data_df['Latitude'].min(); mlr = np.deg2rad(lat_mean); mpdl = 111132.954 - 559.822 * np.cos(2*mlr) + 1.175 * np.cos(4*mlr); mpdlo = 111412.84 * np.cos(mlr) - 93.5 * np.cos(3*mlr); self.plot_data_df['X_m'] = (self.plot_data_df['Longitude'] - min_lon) * mpdlo; self.plot_data_df['Y_m'] = (self.plot_data_df['Latitude'] - min_lat) * mpdl
        self.plot_data_df['Z_Rel'] = self.plot_data_df['Elevation'] - self.z_mean
        if not self.flags_df.empty:
            self.flags_plot_df = self.flags_df.copy()
            if 'Easting' in self.flags_plot_df.columns: self.flags_plot_df['X_m'] = self.flags_plot_df['Easting']; self.flags_plot_df['Y_m'] = self.flags_plot_df['Northing']
            else: self.flags_plot_df['X_m'] = (self.flags_plot_df['Longitude'] - min_lon) * mpdlo; self.flags_plot_df['Y_m'] = (self.flags_plot_df['Latitude'] - min_lat) * mpdl
            self.flags_plot_df['Z_Rel'] = self.flags_plot_df['Elevation'] - self.z_mean
        else: self.flags_plot_df = pd.DataFrame()
            
    def open_gnss(self):
        last_path = load_last_csv_path(); d = os.path.dirname(last_path) if last_path else ""; fp, _ = QFileDialog.getOpenFileName(self, "Open Survey", d, "CSV/TXT (*.csv *.txt)")
        if fp:
            save_last_csv_path(fp); self.gnss_df = load_gnss_data(fp)
            if self.gnss_df.empty: QMessageBox.warning(self,"Error","Empty data"); return
            self.project_lines = []; self.process_coordinates(); self.new_line(); self.update_plot(full=True)

    def open_flags(self):
        d = os.path.dirname(load_last_csv_path()); fp, _ = QFileDialog.getOpenFileName(self, "Open Flags", d, "TXT (*.txt);;All (*.*)")
        if fp: self.flags_df = load_flags_data(fp); self.process_coordinates(); self.update_plot(full=False)

    def new_line(self):
        self.list_w.setCurrentRow(-1); self.current_line_index = -1; self.plotter.remove_actor("marker_start"); self.plotter.remove_actor("marker_end")
        if not self.plot_data_df.empty:
            xm, xM = self.plot_data_df['X_m'].min(), self.plot_data_df['X_m'].max(); ym, yM = self.plot_data_df['Y_m'].min(), self.plot_data_df['Y_m'].max(); zm = self.plot_data_df['Elevation'].mean(); cx = (xm+xM)/2.0; cy = (ym+yM)/2.0; self.temp_line_params = {"x1":cx, "y1":cy, "z1":zm, "x2":cx, "y2":cy, "z2":zm, "cross_slope":-0.002}
        else: self.temp_line_params = {k:0.0 for k in self.line_keys}
        self.refresh_le(); self.draw_lines()

    def del_line(self):
        row = self.list_w.currentRow()
        if 0 <= row < len(self.project_lines): del self.project_lines[row]; self.update_list(); self.new_line(); self.update_plot(full=True)

    def del_all_lines(self):
        if not self.project_lines: return 
        reply = QMessageBox.question(self, 'Confirm Deletion', 'Are you sure you want to delete ALL lines in the project?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes: self.project_lines = []; self.update_list(); self.new_line(); self.update_plot(full=True)

    def sel_line(self, curr, prev):
        row = self.list_w.currentRow(); self.plotter.remove_actor("marker_start"); self.plotter.remove_actor("marker_end")
        if curr and 0 <= row < len(self.project_lines): self.current_line_index = row; self.temp_line_params = self.project_lines[row].copy(); self.refresh_le(); self.draw_lines()
        elif curr is None: self.new_line()
            
    def update_from_le(self):
        try:
            vals = {}
            for k, le in self.le_params.items():
                raw_val = float(le.text().replace(',', '.')); vals[k] = raw_val / 100.0 if k == "cross_slope" else raw_val
        except ValueError: return
        self.temp_line_params.update(vals) 
        if self.current_line_index == -1:
            if (vals["x2"]-vals["x1"])**2 + (vals["y2"]-vals["y1"])**2 > 1e-4: self.project_lines.append(vals.copy()); self.current_line_index = len(self.project_lines) - 1; self.update_list()
        elif 0 <= self.current_line_index < len(self.project_lines): self.project_lines[self.current_line_index].update(vals); self.update_list()
        self.calc_line_stats(); self.draw_lines()

    def update_z2_from_slope(self):
        try:
            slope_cm = float(self.le_long_slope.text().replace(',', '.')); slope_val = slope_cm / 100.0; p = self.temp_line_params; dx = p['x2'] - p['x1']; dy = p['y2'] - p['y1']; dist = math.sqrt(dx*dx + dy*dy); new_z2 = p['z1'] + (dist * slope_val); self.le_params['z2'].setText(f"{new_z2:.3f}"); self.update_from_le()
        except ValueError: pass

    def calc_line_stats(self):
        p = self.temp_line_params; dx = p['x2'] - p['x1']; dy = p['y2'] - p['y1']; dz = p['z2'] - p['z1']; length = math.sqrt(dx*dx + dy*dy)
        if length > 0.001:
            slope_cm = (dz / length) * 100.0
            if not self.le_long_slope.hasFocus(): self.le_long_slope.setText(f"{slope_cm:.2f}")
        self.lbl_stats_len.setText(f"Length: {length:.2f} m")

    def calculate_volumes(self):
        if self.plot_data_df.empty: return
        points = self.plot_data_df[['X_m', 'Y_m']].values
        try: hull = ConvexHull(points); total_area = hull.volume
        except: total_area = 0.0
        n_points = len(self.plot_data_df); point_area = total_area / n_points if n_points > 0 else 0
        if not self.project_lines: delta_z = np.zeros(n_points)
        else:
            z_prop = calculate_z_from_lines(self.plot_data_df['X_m'].values, self.plot_data_df['Y_m'].values, self.project_lines); z_prop = np.nan_to_num(z_prop, nan=self.plot_data_df['Elevation'].values); delta_z = self.plot_data_df['Elevation'].values - z_prop
        cut_vol = np.sum(delta_z[delta_z > 0]) * point_area; fill_vol = np.abs(np.sum(delta_z[delta_z < 0])) * point_area; net_vol = cut_vol - fill_vol
        for a in self.max_min_actors: self.plotter.remove_actor(a, render=False)
        self.max_min_actors = []
        if len(delta_z) > 0:
            max_cut_depth = np.max(delta_z); max_fill_depth = np.min(delta_z)
            try: zx = float(self.le_zx.text())
            except: zx = 1.0
            if zx < 0.1: zx = 1.0
            if max_cut_depth > 0:
                idx_cut = np.argmax(delta_z); pos_cut = [self.plot_data_df.iloc[idx_cut]['X_m'], self.plot_data_df.iloc[idx_cut]['Y_m'], self.plot_data_df.iloc[idx_cut]['Z_Rel']]; s_cut = pv.Sphere(radius=1.5, center=pos_cut); s_cut.scale([1.0, 1.0, 1.0/zx], inplace=True); s_cut.translate([0, 0, 1.0], inplace=True); self.max_min_actors.append(self.plotter.add_mesh(s_cut, color='red')); pos_lbl = [pos_cut[0], pos_cut[1], pos_cut[2] + 3.0]; self.max_min_actors.append(self.plotter.add_point_labels([pos_lbl], [f"Max Cut\n{max_cut_depth:.2f}m"], text_color='white', font_size=16, shape='rounded_rect', shape_color='red', shape_opacity=1.0, always_visible=True))
            if max_fill_depth < 0:
                idx_fill = np.argmin(delta_z); pos_fill = [self.plot_data_df.iloc[idx_fill]['X_m'], self.plot_data_df.iloc[idx_fill]['Y_m'], self.plot_data_df.iloc[idx_fill]['Z_Rel']]; s_fill = pv.Sphere(radius=1.5, center=pos_fill); s_fill.scale([1.0, 1.0, 1.0/zx], inplace=True); s_fill.translate([0, 0, 1.0], inplace=True); self.max_min_actors.append(self.plotter.add_mesh(s_fill, color='blue')); pos_lbl = [pos_fill[0], pos_fill[1], pos_fill[2] + 3.0]; self.max_min_actors.append(self.plotter.add_point_labels([pos_lbl], [f"Max Fill\n{abs(max_fill_depth):.2f}m"], text_color='white', font_size=16, shape='rounded_rect', shape_color='blue', shape_opacity=1.0, always_visible=True))
        else: max_cut_depth = 0; max_fill_depth = 0
        tolerance = 0.05; in_tolerance = np.sum(np.abs(delta_z) < tolerance); similarity_pct = (in_tolerance / n_points) * 100.0 if n_points > 0 else 0
        self.stats_cache = {"area_ha": total_area/10000.0, "cut_m3": cut_vol, "fill_m3": fill_vol, "net_m3": net_vol, "max_cut_m": max_cut_depth, "max_fill_m": abs(max_fill_depth), "sim_pct": similarity_pct}
        self.lbl_area.setText(f"Area: {self.stats_cache['area_ha']:.2f} ha"); self.lbl_cut.setText(f"Cut (Excavation): {cut_vol:.0f} m³"); self.lbl_fill.setText(f"Fill (Embankment): {fill_vol:.0f} m³"); self.lbl_net.setText(f"Net: {net_vol:+.0f} m³"); self.lbl_max_cut.setText(f"Max Cut: {max_cut_depth:.2f} m"); self.lbl_max_fill.setText(f"Max Fill: {abs(max_fill_depth):.2f} m"); self.lbl_sim.setText(f"Similarity (±5cm): {similarity_pct:.1f}%")

    def update_list(self):
        idx = self.current_line_index; self.list_w.blockSignals(True); self.list_w.clear()
        for i, l in enumerate(self.project_lines): slope_cm = l['cross_slope'] * 100.0; self.list_w.addItem(f"L{i+1}: X:{l['x1']:.0f} Y:{l['y1']:.0f} S:{slope_cm:.2f} cm/m")
        if 0 <= idx < len(self.project_lines): self.list_w.setCurrentRow(idx)
        else: self.list_w.setCurrentRow(-1)
        self.list_w.blockSignals(False)
        
    def refresh_le(self):
        for k, le in self.le_params.items():
            val = self.temp_line_params.get(k, 0.0); le.setText(f"{val * 100.0:.2f}" if k == "cross_slope" else f"{val:.3f}")
        self.calc_line_stats() 
                
    def auto_gen(self):
        if self.plot_data_df.empty: return
        try: sp = float(self.le_auto_sp.text()); dr = float(self.le_auto_dr.text()); az = float(self.le_auto_az.text())
        except: return
        pts_x = self.plot_data_df['X_m'].values; pts_y = self.plot_data_df['Y_m'].values; z_avg = self.plot_data_df['Elevation'].mean(); theta = np.deg2rad(-az); c, s = np.cos(theta), np.sin(theta); rx = pts_x * c - pts_y * s; ry = pts_x * s + pts_y * c; min_x, max_x = rx.min(), rx.max(); min_y, max_y = ry.min(), ry.max(); slope = -(abs(dr)/(sp/2.0)); lines = []; cy = (min_y + max_y)/2.0; ylocs = [cy]; curr = cy + sp
        while curr < max_y + 50: ylocs.append(curr); curr += sp
        curr = cy - sp
        while curr > min_y - 50: ylocs.append(curr); curr -= sp
        cx = (min_x + max_x)/2.0; length = (max_x - min_x) + 100; c_inv, s_inv = np.cos(-theta), np.sin(-theta)
        for y in ylocs:
            px1, py1 = cx - length, y; px2, py2 = cx + length, y; x1 = px1 * c_inv - py1 * s_inv; y1 = px1 * s_inv + py1 * c_inv; x2 = px2 * c_inv - py2 * s_inv; y2 = px2 * s_inv + py2 * c_inv; lines.append({"x1":x1, "y1":y1, "z1":z_avg, "x2":x2, "y2":y2, "z2":z_avg, "cross_slope":slope})
        self.project_lines = lines; self.update_list(); self.update_plot(full=True)

    def auto_best_fit(self):
        if self.plot_data_df.empty: return
        p = self.temp_line_params; x1, y1, z1 = p['x1'], p['y1'], p['z1']; x2, y2, z2 = p['x2'], p['y2'], p['z2']; dx_line = x2 - x1; dy_line = y2 - y1; len_sq = dx_line**2 + dy_line**2
        if len_sq < 1e-6: QMessageBox.warning(self, "Attention", "Position the A-B line in 3D first!"); return
        X = self.plot_data_df['X_m'].values; Y = self.plot_data_df['Y_m'].values; Z = self.plot_data_df['Elevation'].values; PX = X - x1; PY = Y - y1; t = (PX * dx_line + PY * dy_line) / len_sq; Z_line = z1 + t * (z2 - z1); length = np.sqrt(len_sq); signed_dist = (PX * dy_line - PY * dx_line) / length; dZ = Z - Z_line; sum_dist2 = np.sum(signed_dist**2); opt_cross_slope = 0.0 if sum_dist2 < 1e-6 else np.sum(signed_dist * dZ) / sum_dist2
        max_cut_cm, ok = QInputDialog.getDouble(self, "Stone/Cut Limit", "Enter MAX allowed cut in cm (e.g. 3.0):", 3.0, 0.0, 1000.0, 1)
        if not ok: return
        max_cut_m = max_cut_cm / 100.0; Z_prop = Z_line + (signed_dist * opt_cross_slope); cuts = Z - Z_prop; current_max_cut = np.max(cuts); msg_extra = ""
        if current_max_cut > max_cut_m: shift_up = current_max_cut - max_cut_m; z1 += shift_up; z2 += shift_up; msg_extra = f"\n\n⚠️ Line raised {(shift_up*100):.1f} cm to respect limit."
        else: msg_extra = f"\n\n✅ Height unchanged. Max cut ({(current_max_cut*100):.1f} cm) is within limits."
        self.temp_line_params.update({"z1": z1, "z2": z2, "cross_slope": opt_cross_slope})
        if self.current_line_index == -1: self.project_lines.append(self.temp_line_params.copy()); self.current_line_index = len(self.project_lines) - 1
        else: self.project_lines[self.current_line_index].update(self.temp_line_params)
        self.update_list(); self.refresh_le(); self.calc_line_stats(); self.update_plot(full=True); QMessageBox.information(self, "Best Fit", f"Process complete.\nCross Slope: {(opt_cross_slope*100):.3f} cm/m{msg_extra}")

    def auto_multi_plane(self):
        if self.plot_data_df.empty: return
        p = self.temp_line_params; x1, y1, z1 = p['x1'], p['y1'], p['z1']; x2, y2, z2 = p['x2'], p['y2'], p['z2']; V_x = x2 - x1; V_y = y2 - y1; len_sq = V_x**2 + V_y**2
        if len_sq < 1e-6: QMessageBox.warning(self, "Attention", "Position points A and B to set flow direction!"); return
        length = np.sqrt(len_sq); step_m, ok1 = QInputDialog.getDouble(self, "Multi-Plane Smoothing", "Distance between sections (hinges) (e.g. 30 m):", 30.0, 5.0, 200.0, 1)
        if not ok1: return
        max_cut_cm, ok2 = QInputDialog.getDouble(self, "Stone/Cut Limit", "MAX allowed cut in cm (e.g. 3.0):", 3.0, 0.0, 1000.0, 1)
        if not ok2: return
        max_cut_m = max_cut_cm / 100.0; X = self.plot_data_df['X_m'].values; Y = self.plot_data_df['Y_m'].values; Z = self.plot_data_df['Elevation'].values; PX = X - x1; PY = Y - y1; sd = (PX * V_y - PY * V_x) / length; max_pos = np.max(sd); min_neg = np.min(sd); flow_sign = 1.0 if abs(max_pos) >= abs(min_neg) else -1.0; max_dist = max_pos if flow_sign > 0 else abs(min_neg); num_lines = int(max_dist / step_m) + 2; lines = []; curr_x1, curr_y1, curr_z1 = x1, y1, z1; curr_x2, curr_y2, curr_z2 = x2, y2, z2; N_x = V_y / length; N_y = -V_x / length
        for i in range(num_lines):
            d_start = i * step_m * flow_sign; d_end = (i + 1) * step_m * flow_sign; mask = (sd >= min(d_start, d_end)) & (sd <= max(d_start, d_end))
            if not np.any(mask): cs = 0.0
            else: strip_Z = Z[mask]; strip_sd = sd[mask]; strip_PX = X[mask] - curr_x1; strip_PY = Y[mask] - curr_y1; t = (strip_PX * V_x + strip_PY * V_y) / len_sq; Z_line_points = curr_z1 + t * (curr_z2 - curr_z1); local_sd = strip_sd - d_start; dZ = strip_Z - Z_line_points; sum_sq = np.sum(local_sd**2); cs = 0.0 if sum_sq < 1e-6 else np.sum(local_sd * dZ) / sum_sq
            lines.append({"x1": curr_x1, "y1": curr_y1, "z1": curr_z1, "x2": curr_x2, "y2": curr_y2, "z2": curr_z2, "cross_slope": cs}); curr_x1 += N_x * step_m * flow_sign; curr_y1 += N_y * step_m * flow_sign; curr_x2 += N_x * step_m * flow_sign; curr_y2 += N_y * step_m * flow_sign; delta_z = (step_m * flow_sign) * cs; curr_z1 += delta_z; curr_z2 += delta_z
        Z_prop = calculate_z_from_lines(X, Y, lines); cuts = Z - Z_prop; current_max_cut = np.max(cuts); msg_extra = ""
        if current_max_cut > max_cut_m:
            shift_up = current_max_cut - max_cut_m
            for l in lines: l["z1"] += shift_up; l["z2"] += shift_up
            msg_extra = f"\n\n⚠️ Structure raised {(shift_up*100):.1f} cm due to stone limit."
        else: msg_extra = f"\n\n✅ No stone risk. Max cut: {(current_max_cut*100):.1f} cm."
        self.project_lines = lines; self.current_line_index = 0
        if len(self.project_lines) > 0: self.temp_line_params = self.project_lines[0].copy()
        self.update_list(); self.refresh_le(); self.calc_line_stats(); self.update_plot(full=True); QMessageBox.information(self, "Multi-Plane Generated", f"Generated {len(lines)} parallel lines every {step_m} meters.{msg_extra}")
    
    def start_pick(self, mode):
        if self.terrain_actor is None: return
        self.picking_mode = mode
        if "pick" in mode: self.pick_azimuth_pts = []
        self.plotter.enable_point_picking(callback=self.picked, show_message="Click on the terrain", show_point=False)

    def picked(self, point):
        p = np.array(point); xm, ym = p[0], p[1]
        try: zx = float(self.le_zx.text())
        except: zx = 1.0
        zm_rel = p[2]; zm_real = zm_rel + self.z_mean; snapped = False
        if not self.flags_plot_df.empty:
            dists = np.sqrt((self.flags_plot_df['X_m']-xm)**2 + (self.flags_plot_df['Y_m']-ym)**2); min_idx = dists.idxmin()
            if dists[min_idx] < 5.0: flag = self.flags_plot_df.loc[min_idx]; xm, ym = flag['X_m'], flag['Y_m']; zm_real, zm_rel = flag['Elevation'], flag['Z_Rel']; snapped = True
        if not snapped and len(self.project_lines) > 0: z_design_arr = calculate_z_from_lines(np.array([xm]), np.array([ym]), self.project_lines); zm_real = z_design_arr[0]; zm_rel = zm_real - self.z_mean
        if self.picking_mode == "line_start": actor_name = "marker_start"; color_sphere = "lime"
        elif self.picking_mode == "line_end": actor_name = "marker_end"; color_sphere = "red"
        else: actor_name = f"p_{np.random.rand()}"; color_sphere = "orange"
        sphere = pv.Sphere(radius=1.0); sphere.scale([1.0, 1.0, 1.0 / zx], inplace=True); sphere.translate([xm, ym, zm_rel], inplace=True); self.plotter.add_mesh(sphere, color=color_sphere, name=actor_name)
        if self.picking_mode == "line_start":
            self.le_params["x1"].setText(f"{xm:.3f}"); self.le_params["y1"].setText(f"{ym:.3f}"); self.le_params["z1"].setText(f"{zm_real:.3f}")
            if self.current_line_index == -1: self.le_params["x2"].setText(f"{xm:.3f}"); self.le_params["y2"].setText(f"{ym:.3f}"); self.le_params["z2"].setText(f"{zm_real:.3f}")
            self.update_from_le(); self.plotter.disable_picking()
        elif self.picking_mode == "line_end": self.le_params["x2"].setText(f"{xm:.3f}"); self.le_params["y2"].setText(f"{ym:.3f}"); self.le_params["z2"].setText(f"{zm_real:.3f}"); self.update_from_le(); self.plotter.disable_picking()
        elif self.picking_mode == "pick_azimuth":
             self.pick_azimuth_pts.append((xm, ym))
             if len(self.pick_azimuth_pts) == 2:
                 p1, p2 = self.pick_azimuth_pts; dx, dy = p2[0]-p1[0], p2[1]-p1[1]; self.le_auto_az.setText(f"{(-np.degrees(np.arctan2(dy, dx)))%360:.2f}"); self.plotter.disable_picking()

    def init_plot_view(self):
        self.plotter.clear_actors(); self.plotter.set_background('lightgray'); self.plotter.show_grid(xtitle="X", ytitle="Y", ztitle="Z"); self.plotter.add_axes(interactive=True); self.plotter.camera_position = 'iso'
    
    def draw_lines(self):
        for a in self.ridge_actors + self.label_actors: self.plotter.remove_actor(a, render=False)
        self.ridge_actors = []; self.label_actors = []
        try: zx = float(self.le_zx.text())
        except: zx = 1.0
        for i, l in enumerate(self.project_lines):
            z_s, z_e = l["z1"] - self.z_mean, l["z2"] - self.z_mean; pts = np.array([[l["x1"], l["y1"], z_s], [l["x2"], l["y2"], z_e]]); col = 'green' if i == self.current_line_index else 'blue'; self.ridge_actors.append(self.plotter.add_lines(pts, color=col, width=6 if i == self.current_line_index else 3))
            if i == self.current_line_index: self.label_actors.append(self.plotter.add_point_labels([[l["x1"], l["y1"], z_s + 1.0]], ["A"], text_color='green', font_size=30, shape_opacity=0.5)); self.label_actors.append(self.plotter.add_point_labels([[l["x2"], l["y2"], z_e + 1.0]], ["B"], text_color='red', font_size=30, shape_opacity=0.5))
        self.calculate_volumes(); self.plotter.render()

    def update_plot(self, full=False):
        try: zx = float(self.le_zx.text())
        except: zx = 1.0
        if self.plot_data_df.empty: return
        self.plotter.set_scale(xscale=1.0, yscale=1.0, zscale=zx)
        if full or self.terrain_actor is None:
            self.plotter.clear_actors(); self.terrain_actor = None; self.design_actor = None; self.init_plot_view(); pts = self.plot_data_df[['X_m', 'Y_m', 'Z_Rel']].values; cloud = pv.PolyData(pts); cloud['Elevation'] = self.plot_data_df['Elevation'].values; surf = cloud.delaunay_2d(); min_x, max_x = self.plot_data_df['X_m'].min(), self.plot_data_df['X_m'].max(); min_y, max_y = self.plot_data_df['Y_m'].min(), self.plot_data_df['Y_m'].max(); min_z_rel, max_z_rel = self.plot_data_df['Z_Rel'].min(), self.plot_data_df['Z_Rel'].max(); self.plotter.show_grid(bounds=[min_x, max_x, min_y, max_y, min_z_rel-0.1, max_z_rel], xtitle="X", ytitle="Y", ztitle=f"Rel. Elevation (0 = {self.z_mean:.1f}m)", fmt="%.2f"); sargs = dict(fmt="%.2f", title="True Elev (m)"); self.terrain_actor = self.plotter.add_mesh(surf, cmap=self.current_colormap, scalars='Elevation', scalar_bar_args=sargs); self.plotter.reset_camera()
        for a in self.flag_actors: self.plotter.remove_actor(a, render=False)
        self.flag_actors = []
        if self.chk_flags.isChecked() and not self.flags_plot_df.empty:
            base_geom = pv.Sphere(radius=1.5); base_geom.scale([1.0, 1.0, 1.0 / zx], inplace=True); pts_data = self.flags_plot_df[['X_m', 'Y_m', 'Z_Rel']].values.copy(); cloud = pv.PolyData(pts_data); glyphs = cloud.glyph(geom=base_geom, scale=False); self.flag_actors.append(self.plotter.add_mesh(glyphs, color='#ff9800')); self.flag_actors.append(self.plotter.add_point_labels(pts_data, self.flags_plot_df['Label'].tolist(), font_size=14))
        for a in self.compass_actors: self.plotter.remove_actor(a, render=False)
        self.compass_actors = []
        if self.chk_compass.isChecked():
            min_x, max_x = self.plot_data_df['X_m'].min(), self.plot_data_df['X_m'].max(); min_y, max_y = self.plot_data_df['Y_m'].min(), self.plot_data_df['Y_m'].max(); z_max_rel = self.plot_data_df['Z_Rel'].max(); cx, cy = (min_x + max_x)/2, (min_y + max_y)/2; z_arrow = z_max_rel + 1.0; scale = np.sqrt((max_x-min_x)**2 + (max_y-min_y)**2) * 0.15; arrow_head = scale * 0.2; lines_n = np.array([ [cx, cy, z_arrow], [cx, cy + scale, z_arrow], [cx, cy + scale, z_arrow], [cx - arrow_head, cy + scale - arrow_head, z_arrow], [cx, cy + scale, z_arrow], [cx + arrow_head, cy + scale - arrow_head, z_arrow] ]); act_n = self.plotter.add_lines(lines_n, color='red', width=5); lbl_n = self.plotter.add_point_labels([[cx, cy + scale*1.1, z_arrow]], ["N"], text_color='red', font_size=25); lines_e = np.array([ [cx, cy, z_arrow], [cx + scale, cy, z_arrow], [cx + scale, cy, z_arrow], [cx + scale - arrow_head, cy + arrow_head, z_arrow], [cx + scale, cy, z_arrow], [cx + scale - arrow_head, cy - arrow_head, z_arrow] ]); act_e = self.plotter.add_lines(lines_e, color='blue', width=5); lbl_e = self.plotter.add_point_labels([[cx + scale*1.1, cy, z_arrow]], ["E"], text_color='blue', font_size=25); self.compass_actors.extend([act_n, lbl_n, act_e, lbl_e])
        self.draw_lines()
        if self.design_actor: self.plotter.remove_actor(self.design_actor, render=False)
        if self.project_lines: margine = 3.0; xm, xM = self.plot_data_df['X_m'].min() - margine, self.plot_data_df['X_m'].max() + margine; ym, yM = self.plot_data_df['Y_m'].min() - margine, self.plot_data_df['Y_m'].max() + margine; vx, vy = np.meshgrid(np.linspace(xm, xM, 50), np.linspace(ym, yM, 50)); vz_real = calculate_z_from_lines(vx.flatten(), vy.flatten(), self.project_lines); grid = pv.StructuredGrid(vx, vy, (vz_real - self.z_mean).reshape(vx.shape)); self.design_actor = self.plotter.add_mesh(grid, color='magenta', opacity=0.65, show_edges=True)
        self.plotter.render()

    def generate_pdf_report(self):
        if self.plot_data_df.empty: QMessageBox.warning(self, "Error", "No data loaded."); return
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", f"Grading_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "PDF (*.pdf)")
        if not filename: return
        try:
            img_path = "temp_map_capture.png"; self.plotter.screenshot(img_path); doc = SimpleDocTemplate(filename, pagesize=letter); styles = getSampleStyleSheet(); elements = []; title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, spaceAfter=20); elements.append(Paragraph(f"TECHNICAL GRADING REPORT", title_style)); elements.append(Paragraph(f"Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}", styles['Normal'])); elements.append(Spacer(1, 12)); coords_text = self.le_center_coords.text(); elements.append(Paragraph(f"<b>Field Center (Lat, Lon, Alt):</b> {coords_text}", styles['Normal'])); elements.append(Spacer(1, 12))
            if os.path.exists(img_path): img = Image(img_path, width=390, height=340); elements.append(img); elements.append(Spacer(1, 12))
            elements.append(Paragraph("<b>Soil Movement Summary</b>", styles['Heading3'])); s = self.stats_cache; data_stats = [["Concept", "Value"], ["Total Surface", f"{s.get('area_ha', 0):.2f} ha"], ["Cut Volume", f"{s.get('cut_m3', 0):.0f} m³"], ["Fill Volume", f"{s.get('fill_m3', 0):.0f} m³"], ["Net Balance", f"{s.get('net_m3', 0):+.0f} m³"], ["Max Cut Depth", f"{s.get('max_cut_m', 0):.2f} m"], ["Max Fill Depth", f"{s.get('max_fill_m', 0):.2f} m"], ["Project Similarity (±5cm)", f"{s.get('sim_pct', 0):.1f} %"]]; t_stats = Table(data_stats, colWidths=[200, 150]); t_stats.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.cadetblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('GRID', (0, 0), (-1, -1), 1, colors.grey)])); elements.append(t_stats); elements.append(Spacer(1, 20))
            if self.project_lines:
                elements.append(Paragraph("<b>Slope Detail by Line</b>", styles['Heading3'])); line_data = [["Line", "Cross Slope (cm/m)", "Long. Slope (cm/m)"]]; i = 1
                for l in self.project_lines: dx = l['x2'] - l['x1']; dy = l['y2'] - l['y1']; dz = l['z2'] - l['z1']; dist = math.sqrt(dx**2 + dy**2); slope_long = (dz / dist * 100) if dist > 0 else 0; line_data.append([f"L{i}", f"{l['cross_slope']*100:.2f}", f"{slope_long:.2f}"]); i += 1
                t_lines = Table(line_data, colWidths=[80, 135, 135]); t_lines.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), ('GRID', (0, 0), (-1, -1), 1, colors.grey), ('ALIGN', (0, 0), (-1, -1), 'CENTER')])); elements.append(t_lines)
            doc.build(elements)
            if os.path.exists(img_path): os.remove(img_path)
            QMessageBox.information(self, "Success", f"Report generated: {filename}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def view_agd_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select AGD File", "", "AGD Files (*.agd);;CSV Files (*.csv)")
        if not fp: return
        try:
            df = pd.read_csv(fp)
            if 'CutFill(m)' not in df.columns:
                if 'Elevation Existing(m)' in df.columns and 'Elevation Proposed(m)' in df.columns: df['CutFill(m)'] = df['Elevation Existing(m)'] - df['Elevation Proposed(m)']
                else: QMessageBox.warning(self, "Error", "File does not contain Cut/Fill data."); return
            
            if self.z_mean == 0: self.z_mean = df['Elevation Proposed(m)'].mean()
            pts_rel = df[['Easting', 'Northing', 'Elevation Proposed(m)']].values.copy(); pts_rel[:, 2] -= self.z_mean
            cloud = pv.PolyData(pts_rel); cloud['Cut_Fill'] = df['CutFill(m)'].values; surf = cloud.delaunay_2d()
            
            limit = max(abs(df['CutFill(m)'].min()), abs(df['CutFill(m)'].max())); sargs = dict(title="Cut (+) / Fill (-) [m]", fmt="%.2f")
            self.plotter.clear_actors(); self.plotter.add_mesh(surf, scalars='Cut_Fill', cmap='RdBu_r', clim=[-limit, limit], scalar_bar_args=sargs, show_edges=True)
            self.plotter.add_text(f"File: {os.path.basename(fp)}", position='upper_left', font_size=10); self.plotter.reset_camera()
        except Exception as e: QMessageBox.critical(self, "Error", f"Could not load: {e}")

    def export_csv(self):
        if self.plot_data_df.empty: QMessageBox.warning(self, "Error", "No data available."); return
        df_e = self.gnss_df.copy(); df_e['Easting'] = self.plot_data_df['X_m'].values; df_e['Northing'] = self.plot_data_df['Y_m'].values
        if 'Elevation' in df_e.columns: df_e.rename(columns={'Elevation': 'Elevation Existing(m)'}, inplace=True)
        df_e['Elevation Proposed(m)'] = calculate_z_from_lines(df_e['Easting'].values, df_e['Northing'].values, self.project_lines) if self.project_lines else df_e['Elevation Existing(m)']
        df_e['Elevation Proposed(m)'].fillna(df_e['Elevation Existing(m)'], inplace=True); df_e['CutFill(m)'] = df_e['Elevation Existing(m)'] - df_e['Elevation Proposed(m)']; df_e['Code'] = "3GRD"; df_e['Comments'] = ""; df_e['Quality'] = 4; df_e['Heading'] = 0.0; df_e['Roll'] = 0.0
        cols = ['Latitude', 'Longitude', 'Elevation Existing(m)', 'Elevation Proposed(m)', 'CutFill(m)', 'Code', 'Comments', 'Quality', 'Easting', 'Northing', 'Heading', 'Roll']
        fp, _ = QFileDialog.getSaveFileName(self, "Export AGD", "grading_export.agd", "AGD (*.agd);;CSV (*.csv)")
        if fp: df_e[cols].to_csv(fp, index=False, float_format='%.8f'); QMessageBox.information(self, "OK", f"Saved: {fp}")

if __name__ == '__main__':
    app = QApplication(sys.argv); w = TerrainApp(); w.show(); sys.exit(app.exec_())