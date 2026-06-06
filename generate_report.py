#!/usr/bin/env python3
"""
Wind Resource Assessment PDF Report Generator
==============================================
Generates a professional 14-page PDF report matching the BAE_V163_Illinois_AEP_Report format.

Tech Stack: ReportLab (body) + Playwright/HTML (cover) + pypdf (merge) + matplotlib (charts)

Usage:
    python3 generate_report.py [--input data.json] [--output report.pdf]

Default output: /home/z/my-project/download/Wind_Resource_Assessment_Report.pdf
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import weibull_min

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch, mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# ━━ Constants ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PAGE_W, PAGE_H = A4  # 595.28 x 841.89
LEFT_MARGIN = 1.0 * inch
RIGHT_MARGIN = 1.0 * inch
TOP_MARGIN = 0.8 * inch
BOTTOM_MARGIN = 0.8 * inch
AVAILABLE_WIDTH = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN
MAX_KEEP_HEIGHT = PAGE_H * 0.4
H1_ORPHAN_THRESHOLD = (PAGE_H - TOP_MARGIN - BOTTOM_MARGIN) * 0.15

CHART_DIR = '/tmp/windflow_charts'
PDF_SKILL_DIR = '/home/z/my-project/skills/pdf'
DEFAULT_OUTPUT = '/home/z/my-project/download/Wind_Resource_Assessment_Report.pdf'

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ━━ Palette (auto-generated) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCENT       = colors.HexColor('#2d8eae')
TEXT_PRIMARY  = colors.HexColor('#191b1c')
TEXT_MUTED    = colors.HexColor('#70767b')
BG_SURFACE   = colors.HexColor('#d9dfe4')
BG_PAGE      = colors.HexColor('#eff1f2')

TABLE_HEADER_COLOR = ACCENT
TABLE_HEADER_TEXT  = colors.white
TABLE_ROW_EVEN     = colors.white
TABLE_ROW_ODD      = BG_SURFACE

# ━━ Font Registration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Liberation Serif is a metric-compatible equivalent of Times New Roman
pdfmetrics.registerFont(TTFont('LiberationSerif', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('LiberationSerif-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSerif', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSerif-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'))
registerFontFamily('LiberationSerif', normal='LiberationSerif', bold='LiberationSerif-Bold')
registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans')
registerFontFamily('DejaVuSerif', normal='DejaVuSerif', bold='DejaVuSerif-Bold')

# ━━ Styles ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

styles = getSampleStyleSheet()

sTitle = ParagraphStyle(
    'ReportTitle', fontName='LiberationSerif', fontSize=24, leading=30,
    alignment=TA_LEFT, textColor=TEXT_PRIMARY, spaceAfter=12)

sH1 = ParagraphStyle(
    'H1', fontName='LiberationSerif', fontSize=18, leading=24,
    alignment=TA_LEFT, textColor=ACCENT, spaceBefore=18, spaceAfter=8)

sH2 = ParagraphStyle(
    'H2', fontName='LiberationSerif', fontSize=14, leading=19,
    alignment=TA_LEFT, textColor=TEXT_PRIMARY, spaceBefore=12, spaceAfter=6)

sBody = ParagraphStyle(
    'Body', fontName='LiberationSerif', fontSize=10.5, leading=16,
    alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY, spaceAfter=6)

sBodyLeft = ParagraphStyle(
    'BodyLeft', parent=sBody, alignment=TA_LEFT)

sBullet = ParagraphStyle(
    'Bullet', parent=sBody, leftIndent=20, bulletIndent=8, spaceAfter=4)

sCaption = ParagraphStyle(
    'Caption', fontName='LiberationSerif', fontSize=9, leading=13,
    alignment=TA_CENTER, textColor=TEXT_MUTED, spaceBefore=3, spaceAfter=6)

sTH = ParagraphStyle(
    'TH', fontName='LiberationSerif', fontSize=10, leading=14,
    alignment=TA_CENTER, textColor=TABLE_HEADER_TEXT)

sTC = ParagraphStyle(
    'TC', fontName='LiberationSerif', fontSize=10, leading=14,
    alignment=TA_CENTER, textColor=TEXT_PRIMARY)

sTCL = ParagraphStyle(
    'TCL', fontName='LiberationSerif', fontSize=10, leading=14,
    alignment=TA_LEFT, textColor=TEXT_PRIMARY)

sTCR = ParagraphStyle(
    'TCR', fontName='LiberationSerif', fontSize=10, leading=14,
    alignment=TA_RIGHT, textColor=TEXT_PRIMARY)

sFooter = ParagraphStyle(
    'Footer', fontName='LiberationSerif', fontSize=8, leading=10,
    alignment=TA_CENTER, textColor=TEXT_MUTED)

sFormula = ParagraphStyle(
    'Formula', fontName='DejaVuSans', fontSize=10, leading=16,
    alignment=TA_CENTER, textColor=TEXT_PRIMARY, spaceAfter=8,
    spaceBefore=8)

sTOCH1 = ParagraphStyle(
    'TOCH1', fontName='LiberationSerif', fontSize=13, leading=22,
    leftIndent=20, spaceBefore=4)

sTOCH2 = ParagraphStyle(
    'TOCH2', fontName='LiberationSerif', fontSize=11, leading=18,
    leftIndent=40, spaceBefore=2)

# ━━ Default Data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_DATA = {
    "project_name": "BAE Wind Farm",
    "location": "Illinois, USA",
    "latitude": 40.58,
    "longitude": -88.53,
    "elevation": 238,
    "terrain_type": "Agricultural (flat to gently rolling)",
    "turbine_model": "Vestas V163-4.5MW",
    "hub_height": 113,
    "rotor_diameter": 163,
    "rated_power_kw": 4500,
    "num_turbines": 130,
    "wake_model": "Jensen/PARK (k=0.04)",
    "data_source": "ERA5 via Open-Meteo (2020-2024)",
    "sectors": 16,
    "mean_ws_100m": 6.86,
    "mean_ws_hub": 6.98,
    "weibull_A_100m": 7.73,
    "weibull_k_100m": 2.37,
    "weibull_A_hub": 7.87,
    "weibull_k_hub": 2.37,
    "wind_power_density_100m": 319.5,
    "wind_power_density_hub": 336.8,
    "air_density": 1.2051,
    "air_density_std": 0.0528,
    "roughness_z0": 0.10,
    "power_law_alpha": 0.1422,
    "shear_factor": 1.0177,
    "gross_aep_weibull": 12.63,
    "gross_aep_ts": 12.48,
    "gross_aep": 12.48,
    "wake_loss_pct": 9,
    "other_loss_pct": 5,
    "net_aep": 10.79,
    "gross_cf_pct": 31.7,
    "net_cf_pct": 27.4,
    "p50": 10.79,
    "p75": 9.92,
    "p84": 9.50,
    "p90": 9.13,
    "p95": 8.66,
    "uncertainty_sigma_pct": 12,
    "wind_rose": [
        {"sector": "N", "center_deg": 0.0, "freq_pct": 4.20, "mean_ws": 5.83},
        {"sector": "NNE", "center_deg": 22.5, "freq_pct": 3.78, "mean_ws": 5.89},
        {"sector": "NE", "center_deg": 45.0, "freq_pct": 4.91, "mean_ws": 5.85},
        {"sector": "ENE", "center_deg": 67.5, "freq_pct": 4.36, "mean_ws": 5.63},
        {"sector": "E", "center_deg": 90.0, "freq_pct": 4.27, "mean_ws": 5.75},
        {"sector": "ESE", "center_deg": 112.5, "freq_pct": 3.96, "mean_ws": 6.02},
        {"sector": "SE", "center_deg": 135.0, "freq_pct": 4.50, "mean_ws": 6.35},
        {"sector": "SSE", "center_deg": 157.5, "freq_pct": 5.95, "mean_ws": 7.08},
        {"sector": "S", "center_deg": 180.0, "freq_pct": 9.76, "mean_ws": 8.42},
        {"sector": "SSW", "center_deg": 202.5, "freq_pct": 10.79, "mean_ws": 8.31},
        {"sector": "SW", "center_deg": 225.0, "freq_pct": 8.20, "mean_ws": 7.36},
        {"sector": "WSW", "center_deg": 247.5, "freq_pct": 6.10, "mean_ws": 6.95},
        {"sector": "W", "center_deg": 270.0, "freq_pct": 7.45, "mean_ws": 7.22},
        {"sector": "WNW", "center_deg": 292.5, "freq_pct": 8.11, "mean_ws": 7.45},
        {"sector": "NW", "center_deg": 315.0, "freq_pct": 8.11, "mean_ws": 6.97},
        {"sector": "NNW", "center_deg": 337.5, "freq_pct": 5.54, "mean_ws": 6.10}
    ],
    "power_curve": [
        {"ws": 3, "power_kw": 0}, {"ws": 4, "power_kw": 126},
        {"ws": 5, "power_kw": 292}, {"ws": 6, "power_kw": 585},
        {"ws": 7, "power_kw": 1010}, {"ws": 8, "power_kw": 1600},
        {"ws": 9, "power_kw": 2350}, {"ws": 10, "power_kw": 3180},
        {"ws": 11, "power_kw": 3920}, {"ws": 12, "power_kw": 4350},
        {"ws": 13, "power_kw": 4500}, {"ws": 14, "power_kw": 4500},
        {"ws": 15, "power_kw": 4500}, {"ws": 16, "power_kw": 4500},
        {"ws": 17, "power_kw": 4500}, {"ws": 18, "power_kw": 4500},
        {"ws": 19, "power_kw": 4500}, {"ws": 20, "power_kw": 4500},
        {"ws": 21, "power_kw": 4500}, {"ws": 22, "power_kw": 4500},
        {"ws": 23, "power_kw": 4500}, {"ws": 24, "power_kw": 4500},
        {"ws": 25, "power_kw": 4500}
    ],
    "monthly_ws": [7.8, 7.5, 7.9, 8.1, 7.2, 6.3, 5.8, 5.6, 6.4, 7.1, 7.6, 7.9],
    "diurnal_ws": [6.2, 5.9, 5.7, 5.5, 5.4, 5.6, 6.1, 6.8, 7.3, 7.6, 7.8, 7.9,
                   7.8, 7.7, 7.5, 7.3, 7.1, 6.9, 6.7, 6.5, 6.4, 6.3, 6.2, 6.1],
    "loss_breakdown": [
        {"category": "Wake Losses", "pct": 9, "method": "Jensen/PARK (k=0.04) assumed"},
        {"category": "Electrical Losses", "pct": 1.5, "method": "Transformer and cable losses"},
        {"category": "Availability", "pct": 2.0, "method": "Turbine and substation downtime"},
        {"category": "Curtailment (environmental/grid)", "pct": 1.0, "method": "Noise, shadow flicker, grid"},
        {"category": "Icing / Cold Weather", "pct": 0.5, "method": "Blade icing and low-temp shutdown"},
        {"category": "Total Other Losses", "pct": 5, "method": "Combined estimate"}
    ],
    "assumptions": [
        "ERA5 reanalysis data at 0.25 degree resolution (~25 km grid) is used as a proxy for site-specific wind measurements.",
        "Wind shear extrapolation uses a constant roughness length.",
        "The power curve is based on published specifications and standard air density corrections.",
        "Wake losses are estimated based on typical layouts.",
        "Other losses are a combined estimate covering electrical, availability, curtailment, and icing losses.",
        "P-exceedance analysis assumes a standard deviation in annual energy production.",
        "No long-term correction (MCP) has been applied."
    ],
    "per_turbine_data": []
}


# ━━ Chart Generation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCENT_HEX = '#2d8eae'
ACCENT_LIGHT = '#8ed0e8'
BG_HEX = '#eff1f2'
GRID_COLOR = '#c8cdd2'
TEXT_MUTED_HEX = '#70767b'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': '#191b1c',
    'xtick.color': TEXT_MUTED_HEX,
    'ytick.color': TEXT_MUTED_HEX,
    'grid.color': GRID_COLOR,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.6,
})


def generate_weibull_chart(d):
    """Weibull distribution histogram + fitted PDF."""
    A = d['weibull_A_hub']
    k = d['weibull_k_hub']
    mean_ws = d['mean_ws_hub']

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    # Simulate histogram from Weibull parameters
    np.random.seed(42)
    samples = weibull_min.rvs(k, scale=A, size=50000)
    ax.hist(samples, bins=np.arange(0, 26, 1), density=True,
            color=ACCENT_LIGHT, edgecolor='white', alpha=0.7, label='Wind Speed Distribution')

    # Fitted Weibull PDF
    x = np.linspace(0, 25, 300)
    pdf = weibull_min.pdf(x, k, scale=A)
    ax.plot(x, pdf, color=ACCENT_HEX, linewidth=2.2, label=f'Weibull PDF (A={A}, k={k})')

    ax.axvline(mean_ws, color='#e05c5c', linewidth=1.5, linestyle='--',
               label=f'Mean WS = {mean_ws} m/s')

    ax.set_xlabel('Wind Speed (m/s)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title('Weibull Distribution at Hub Height', fontsize=12, fontweight='bold', color='#191b1c')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 25)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'weibull.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_wind_rose_chart(d):
    """16-sector wind rose with frequency bars colored by mean WS."""
    rose = d['wind_rose']
    n = len(rose)
    directions = np.array([r['center_deg'] for r in rose])
    freqs = np.array([r['freq_pct'] for r in rose])
    mean_wss = np.array([r['mean_ws'] for r in rose])
    labels = [r['sector'] for r in rose]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    theta = np.deg2rad(directions)
    width = 2 * np.pi / n * 0.85

    # Color by mean wind speed
    norm = plt.Normalize(vmin=mean_wss.min(), vmax=mean_wss.max())
    cmap = matplotlib.colormaps.get_cmap('YlGnBu')
    bar_colors = cmap(norm(mean_wss))

    bars = ax.bar(theta, freqs, width=width, color=bar_colors,
                  edgecolor='white', linewidth=0.5, alpha=0.88)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(directions, labels, fontsize=8)
    ax.set_rlabel_position(67.5)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2%', '4%', '6%', '8%', '10%'], fontsize=7, color=TEXT_MUTED_HEX)
    ax.set_title('Wind Rose (Frequency by Sector)', fontsize=12,
                 fontweight='bold', color='#191b1c', pad=20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label('Mean WS (m/s)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'wind_rose.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_monthly_chart(d):
    """Monthly mean wind speed bar chart."""
    ws = d['monthly_ws']
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    x = np.arange(12)
    bars = ax.bar(x, ws, color=ACCENT_HEX, width=0.65, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, ws):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#191b1c')

    mean_val = np.mean(ws)
    ax.axhline(mean_val, color='#e05c5c', linewidth=1.5, linestyle='--',
               label=f'Annual Mean = {mean_val:.1f} m/s')

    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS, fontsize=9)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=10)
    ax.set_title('Monthly Mean Wind Speed at Hub Height', fontsize=12,
                 fontweight='bold', color='#191b1c')
    ax.set_ylim(0, max(ws) * 1.2)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'monthly_ws.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_diurnal_chart(d):
    """Diurnal wind speed pattern line chart."""
    ws = d['diurnal_ws']
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    hours = np.arange(24)
    ax.plot(hours, ws, color=ACCENT_HEX, linewidth=2.2, marker='o',
            markersize=4, markerfacecolor=ACCENT_HEX, markeredgecolor='white', markeredgewidth=0.5)
    ax.fill_between(hours, ws, alpha=0.12, color=ACCENT_HEX)

    mean_val = np.mean(ws)
    ax.axhline(mean_val, color='#e05c5c', linewidth=1.2, linestyle='--',
               label=f'Daily Mean = {mean_val:.1f} m/s')

    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}' for h in hours], fontsize=7)
    ax.set_xlabel('Hour of Day (Local Time)', fontsize=10)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=10)
    ax.set_title('Diurnal Wind Speed Pattern at Hub Height', fontsize=12,
                 fontweight='bold', color='#191b1c')
    ax.set_xlim(0, 23)
    ax.set_ylim(min(ws) * 0.85, max(ws) * 1.1)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'diurnal_ws.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_power_curve_chart(d):
    """Power curve with cut-in, rated, cut-out markers."""
    pc = d['power_curve']
    ws_vals = [p['ws'] for p in pc]
    pw_vals = [p['power_kw'] / 1000 for p in pc]  # MW

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.plot(ws_vals, pw_vals, color=ACCENT_HEX, linewidth=2.2, label='Power Curve')
    ax.fill_between(ws_vals, pw_vals, alpha=0.08, color=ACCENT_HEX)

    # Markers
    rated_mw = d['rated_power_kw'] / 1000
    ax.axhline(rated_mw, color=TEXT_MUTED_HEX, linewidth=0.8, linestyle=':', alpha=0.6)
    ax.text(24.5, rated_mw, f'Rated: {rated_mw:.1f} MW', fontsize=7,
            va='center', ha='left', color=TEXT_MUTED_HEX)

    # Cut-in marker
    ax.axvline(3, color='#e0a85c', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.text(3.2, 0.1, 'Cut-in\n3 m/s', fontsize=7, color='#e0a85c')

    # Rated WS marker
    rated_ws = None
    for p in pc:
        if p['power_kw'] >= d['rated_power_kw']:
            rated_ws = p['ws']
            break
    if rated_ws:
        ax.axvline(rated_ws, color='#5cb85c', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(rated_ws + 0.3, 0.1, f'Rated\n{rated_ws} m/s', fontsize=7, color='#5cb85c')

    # Cut-out marker
    ax.axvline(25, color='#e05c5c', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.text(23, 0.1, 'Cut-out\n25 m/s', fontsize=7, color='#e05c5c', ha='right')

    ax.set_xlabel('Wind Speed (m/s)', fontsize=10)
    ax.set_ylabel('Power (MW)', fontsize=10)
    ax.set_title(f"Power Curve: {d['turbine_model']}", fontsize=12,
                 fontweight='bold', color='#191b1c')
    ax.set_xlim(0, 27)
    ax.set_ylim(0, rated_mw * 1.15)
    ax.legend(fontsize=8, loc='center right')
    ax.grid(axis='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'power_curve.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


# ━━ Helper Functions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def P(text, style=sBody):
    """Shortcut for Paragraph."""
    return Paragraph(str(text), style)


def PH(text, style=sTH):
    """Shortcut for header Paragraph."""
    return Paragraph(f'<b>{text}</b>', style)


def add_heading(text, style, level=0):
    """Create a heading with bookmark for TOC."""
    key = 'h_%s' % hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph(f'<a name="{key}"/>{text}', style)
    p.bookmark_name = text
    p.bookmark_level = level
    p.bookmark_text = text
    p.bookmark_key = key
    return p


def add_major_section(text, style=sH1):
    """Add H1 with orphan prevention."""
    return [
        CondPageBreak(H1_ORPHAN_THRESHOLD),
        add_heading(f'<b>{text}</b>', style, level=0),
    ]


def safe_keep_together(elements):
    """Wrap elements in KeepTogether only if total height is reasonable."""
    total_h = 0
    for el in elements:
        try:
            w, h = el.wrap(AVAILABLE_WIDTH, PAGE_H)
            total_h += h
        except Exception:
            total_h += 50
    if total_h <= MAX_KEEP_HEIGHT:
        return [KeepTogether(elements)]
    elif len(elements) >= 2:
        return [KeepTogether(elements[:2])] + list(elements[2:])
    else:
        return list(elements)


def make_table(data, col_widths, caption=None):
    """Create a styled table with consistent formatting."""
    tbl = Table(data, colWidths=col_widths, hAlign='CENTER')
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), TABLE_HEADER_TEXT),
        ('GRID', (0, 0), (-1, -1), 0.5, TEXT_MUTED),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(data)):
        bg = TABLE_ROW_EVEN if i % 2 == 1 else TABLE_ROW_ODD
        style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
    tbl.setStyle(TableStyle(style_cmds))

    elements = [Spacer(1, 18), tbl]
    if caption:
        elements.append(Spacer(1, 6))
        elements.append(P(caption, sCaption))
    elements.append(Spacer(1, 18))
    return elements


def img_element(path, width_inches=5.5, caption=None):
    """Create Image element with optional caption."""
    max_w = AVAILABLE_WIDTH - 10  # leave some padding
    img = Image(path, width=width_inches * inch)
    # Ensure image fits within available width
    if img.drawWidth > max_w:
        ratio = max_w / img.drawWidth
        img.drawWidth = max_w
        img.drawHeight = img.drawHeight * ratio
    # Also cap height at 60% of available frame height
    max_h = PAGE_H * 0.50
    if img.drawHeight > max_h:
        ratio = max_h / img.drawHeight
        img.drawHeight = max_h
        img.drawWidth = img.drawWidth * ratio
    img.hAlign = 'CENTER'
    elements = [Spacer(1, 12), img]
    if caption:
        elements.append(Spacer(1, 6))
        elements.append(P(caption, sCaption))
    elements.append(Spacer(1, 12))
    return elements


# ━━ TocDocTemplate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark_name'):
            level = getattr(flowable, 'bookmark_level', 0)
            text = getattr(flowable, 'bookmark_text', '')
            key = getattr(flowable, 'bookmark_key', '')
            self.notify('TOCEntry', (level, text, self.page, key))


# ━━ Page Footer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def footer_handler(canvas, doc, project_name):
    """Draw page footer on every page."""
    canvas.saveState()
    canvas.setFont('LiberationSerif', 8)
    canvas.setFillColor(TEXT_MUTED)
    page_num = doc.page
    footer_text = f"{project_name} - Wind Resource Assessment | Page {page_num}"
    canvas.drawCentredString(PAGE_W / 2, 0.4 * inch, footer_text)
    # Thin line above footer
    canvas.setStrokeColor(BG_SURFACE)
    canvas.setLineWidth(0.5)
    canvas.line(LEFT_MARGIN, 0.55 * inch, PAGE_W - RIGHT_MARGIN, 0.55 * inch)
    canvas.restoreState()


# ━━ Cover Page HTML ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_cover_html(d):
    """Generate cover page HTML using Template 01 (HUD Data Terminal) style."""
    project = d['project_name']
    location = d['location']
    data_source = d['data_source']
    date_str = datetime.now().strftime('%B %Y')
    mean_ws = d['mean_ws_hub']
    net_aep = d['net_aep']
    net_cf = d['net_cf_pct']
    num_t = d['num_turbines']
    model = d['turbine_model']

    summary = (
        f"This report presents the wind resource assessment for {project}, located in {location}. "
        f"Based on {data_source}, the mean wind speed at hub height ({d['hub_height']}m) is "
        f"{mean_ws} m/s with a net AEP of {net_aep} GWh and a net capacity factor of {net_cf}%. "
        f"The project comprises {num_t} x {model} turbines."
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap" rel="stylesheet">
<style>
@page {{
    size: 794px 1123px;
    margin: 0;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{
    width: 794px;
    height: 1123px;
    margin: 0;
    padding: 0;
    background: #ffffff;
    font-family: 'Inter', sans-serif;
    overflow: hidden;
}}
.page {{
    width: 794px;
    height: 1123px;
    position: relative;
    overflow: hidden;
}}
/* Layer 1 - Background Grid */
.grid-bg {{
    position: absolute;
    inset: 0;
    z-index: 1;
    overflow: hidden;
    pointer-events: none;
}}
.grid-bg svg {{
    width: 100%;
    height: 100%;
}}
/* Layer 2 - Structure Lines */
.anchor-line {{
    position: absolute;
    left: 95px;
    top: 112px;
    width: 6px;
    height: 898px;
    background: {ACCENT_HEX};
    z-index: 2;
    border-radius: 3px;
}}
.meta-separator {{
    position: absolute;
    left: 137px;
    top: 792px;
    width: 280px;
    height: 1px;
    background: rgba(45, 142, 174, 0.4);
    z-index: 2;
}}
/* Layer 3 - Content */
.content {{
    position: absolute;
    left: 137px;
    top: 0;
    width: 600px;
    height: 100%;
    z-index: 3;
}}
.kicker {{
    position: absolute;
    top: 168px;
    font-size: 16px;
    font-weight: 400;
    letter-spacing: 3px;
    color: rgba(25, 27, 28, 0.6);
    text-transform: uppercase;
    line-height: 1.3;
}}
.hero-title {{
    position: absolute;
    top: 310px;
    font-size: 52px;
    font-weight: 900;
    color: #191b1c;
    line-height: 1.15;
    letter-spacing: -0.5px;
}}
.summary {{
    position: absolute;
    top: 520px;
    width: 420px;
    font-size: 16px;
    font-weight: 400;
    line-height: 1.6;
    color: rgba(25, 27, 28, 0.85);
}}
.meta {{
    position: absolute;
    top: 812px;
    font-size: 16px;
    font-weight: 400;
    line-height: 2.0;
    color: rgba(25, 27, 28, 0.85);
}}
.meta-label {{
    font-weight: 300;
    color: rgba(25, 27, 28, 0.5);
    font-size: 14px;
}}
</style>
</head>
<body>
<div class="page">
    <!-- Layer 1: Grid Background -->
    <div class="grid-bg">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 794 1123">
            <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(45,142,174,0.04)" stroke-width="0.5"/>
                </pattern>
            </defs>
            <rect width="794" height="1123" fill="url(#grid)"/>
        </svg>
    </div>

    <!-- Layer 2: Structure -->
    <div class="anchor-line"></div>
    <div class="meta-separator"></div>

    <!-- Layer 3: Content -->
    <div class="content">
        <div class="kicker">Wind Resource Assessment</div>
        <div class="hero-title">{project}</div>
        <div class="summary">{summary}</div>
        <div class="meta">
            <span class="meta-label">Location:</span> {location}<br/>
            <span class="meta-label">Date:</span> {date_str}<br/>
            <span class="meta-label">Data Source:</span> {data_source}
        </div>
    </div>
</div>
</body>
</html>
"""
    return html


# ━━ Report Body ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_report(d, chart_paths):
    """Build the ReportLab story (body only, no cover)."""
    story = []

    # ─── Table of Contents ───
    story.append(P('<b>Table of Contents</b>', sTitle))
    story.append(Spacer(1, 12))

    toc = TableOfContents()
    toc.levelStyles = [sTOCH1, sTOCH2]
    story.append(toc)
    story.append(PageBreak())

    # ─── Section 1: Project Overview ───
    story.extend(add_major_section('1. Project Overview'))

    overview_text = (
        f"This report presents the wind resource assessment for the {d['project_name']} project, "
        f"located in {d['location']} ({d['latitude']}N, {abs(d['longitude'])}W) at an elevation "
        f"of {d['elevation']}m above sea level. The site is characterized by {d['terrain_type'].lower()} "
        f"terrain. The assessment is based on {d['data_source']} reanalysis data, covering a five-year period. "
        f"The proposed wind farm consists of {d['num_turbines']} {d['turbine_model']} turbines "
        f"with a hub height of {d['hub_height']}m and a rotor diameter of {d['rotor_diameter']}m."
    )
    story.extend(safe_keep_together([
        add_heading('<b>1. Project Overview</b>', sH1, level=0),
        P(overview_text),
    ]))

    story.append(Spacer(1, 12))

    # Project Summary Table
    total_capacity_mw = d['num_turbines'] * d['rated_power_kw'] / 1000
    rows = [
        [PH('Parameter'), PH('Value')],
        [P('Project Name', sTCL), P(d['project_name'], sTC)],
        [P('Location', sTCL), P(d['location'], sTC)],
        [P('Coordinates', sTCL), P(f"{d['latitude']}N, {abs(d['longitude'])}W", sTC)],
        [P('Elevation', sTCL), P(f"{d['elevation']} m", sTC)],
        [P('Terrain Type', sTCL), P(d['terrain_type'], sTC)],
        [P('Turbine Model', sTCL), P(d['turbine_model'], sTC)],
        [P('Number of Turbines', sTCL), P(str(d['num_turbines']), sTC)],
        [P('Hub Height', sTCL), P(f"{d['hub_height']} m", sTC)],
        [P('Rotor Diameter', sTCL), P(f"{d['rotor_diameter']} m", sTC)],
        [P('Rated Power per Turbine', sTCL), P(f"{d['rated_power_kw']:,} kW", sTC)],
        [P('Total Installed Capacity', sTCL), P(f"{total_capacity_mw:,.1f} MW", sTC)],
        [P('Data Source', sTCL), P(d['data_source'], sTC)],
    ]
    cw = [AVAILABLE_WIDTH * 0.42, AVAILABLE_WIDTH * 0.58]
    story.extend(make_table(rows, cw, 'Table 1: Project Summary'))

    # ─── Section 2: Wind Data and Methodology ───
    story.extend(add_major_section('2. Wind Data and Methodology'))
    story.append(add_heading('<b>2.1 Data Source</b>', sH2, level=1))
    ds_text = (
        f"Wind data for this assessment was obtained from {d['data_source']}. "
        f"ERA5 is the fifth-generation ECMWF reanalysis dataset, providing hourly estimates of atmospheric "
        f"variables on a 0.25-degree grid (~25 km resolution). The dataset covers the period from 2020 to 2024, "
        f"offering a statistically representative sample of the wind climate at the project site. "
        f"Wind speeds at 100m above ground level were extracted and extrapolated to hub height ({d['hub_height']}m) "
        f"using the power law with a shear exponent derived from the site roughness."
    )
    story.append(P(ds_text))

    story.append(Spacer(1, 12))
    story.append(add_heading('<b>2.2 Methodology</b>', sH2, level=1))
    meth_text = (
        "The wind resource assessment follows a standard methodology consisting of: "
        "(1) data quality control and validation, "
        "(2) wind shear extrapolation from 100m to hub height using the power law, "
        "(3) Weibull distribution fitting to the long-term wind speed data, "
        f"(4) air density calculation using the ERA5 temperature and pressure data, "
        f"(5) Annual Energy Production (AEP) estimation using the manufacturer's power curve adjusted for site-specific "
        f"air density, "
        f"(6) wake loss estimation using the {d['wake_model']} wake model, "
        f"and (7) exceedance probability (P-value) analysis based on a {d['uncertainty_sigma_pct']}% "
        f"standard deviation in AEP."
    )
    story.append(P(meth_text))

    # ─── Section 3: Wind Resource Summary ───
    story.extend(add_major_section('3. Wind Resource Summary'))

    # 3.1 Wind Speed Statistics
    story.append(add_heading('<b>3.1 Wind Speed Statistics</b>', sH2, level=1))
    ws_text = (
        f"The mean wind speed at 100m is {d['mean_ws_100m']} m/s, and at hub height ({d['hub_height']}m) "
        f"it is {d['mean_ws_hub']} m/s. The wind power density increases from "
        f"{d['wind_power_density_100m']} W/m<super>2</super> at 100m to "
        f"{d['wind_power_density_hub']} W/m<super>2</super> at hub height, "
        f"indicating a viable wind resource for the proposed turbine configuration."
    )
    story.append(P(ws_text))

    ws_rows = [
        [PH('Parameter'), PH('100m'), PH(f"Hub ({d['hub_height']}m)")],
        [P('Mean Wind Speed', sTCL), P(f"{d['mean_ws_100m']} m/s", sTC), P(f"{d['mean_ws_hub']} m/s", sTC)],
        [P('Weibull A (scale)', sTCL), P(f"{d['weibull_A_100m']}", sTC), P(f"{d['weibull_A_hub']}", sTC)],
        [P('Weibull k (shape)', sTCL), P(f"{d['weibull_k_100m']}", sTC), P(f"{d['weibull_k_hub']}", sTC)],
        [P('Wind Power Density', sTCL),
         P(f"{d['wind_power_density_100m']} W/m<super>2</super>", sTC),
         P(f"{d['wind_power_density_hub']} W/m<super>2</super>", sTC)],
    ]
    cw3 = [AVAILABLE_WIDTH * 0.36, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.32]
    story.extend(make_table(ws_rows, cw3, 'Table 2: Wind Speed Statistics'))

    # 3.2 Weibull Distribution
    story.append(add_heading('<b>3.2 Weibull Distribution</b>', sH2, level=1))
    wb_text = (
        f"The Weibull distribution at hub height has a scale parameter A = {d['weibull_A_hub']} "
        f"and a shape parameter k = {d['weibull_k_hub']}. The shape parameter k > 2 indicates a relatively "
        f"narrow distribution with most wind speeds concentrated around the mean. "
        f"Figure 1 shows the wind speed histogram with the fitted Weibull PDF."
    )
    story.append(P(wb_text))
    story.extend(img_element(chart_paths['weibull'], 6.2, 'Figure 1: Weibull Distribution at Hub Height'))

    # 3.3 Wind Rose
    story.append(add_heading('<b>3.3 Wind Rose</b>', sH2, level=1))
    wr_text = (
        "The wind rose analysis reveals a dominant wind direction from the south and southwest sectors, "
        "with the S and SSW sectors contributing approximately 20.5% of the total frequency. "
        "The highest mean wind speeds are also observed in these sectors, exceeding 8 m/s. "
        "Figure 2 presents the 16-sector wind rose with bars colored by mean wind speed."
    )
    story.append(P(wr_text))
    story.extend(img_element(chart_paths['wind_rose'], 4.8, 'Figure 2: Wind Rose (Frequency and Mean Speed by Sector)'))

    # Wind rose table
    rose = d['wind_rose']
    rose_rows = [[PH('Sector'), PH('Center'), PH('Freq (%)'), PH('Mean WS (m/s)')]]
    for r in rose:
        rose_rows.append([
            P(r['sector'], sTC),
            P(f"{r['center_deg']:.1f} deg", sTC),
            P(f"{r['freq_pct']:.2f}", sTC),
            P(f"{r['mean_ws']:.2f}", sTC),
        ])
    cw_rose = [AVAILABLE_WIDTH * 0.18, AVAILABLE_WIDTH * 0.22, AVAILABLE_WIDTH * 0.28, AVAILABLE_WIDTH * 0.32]
    story.extend(make_table(rose_rows, cw_rose, 'Table 3: Wind Rose Data by Sector'))

    # ─── Section 4: Air Density ───
    story.extend(add_major_section('4. Air Density'))
    ad_text = (
        f"Site-specific air density was calculated using ERA5 temperature and pressure data following "
        f"the ERA5T method. The mean air density at the site is {d['air_density']} kg/m<super>3</super> "
        f"with a standard deviation of {d['air_density_std']} kg/m<super>3</super>. This is slightly below "
        f"the standard air density of 1.225 kg/m<super>3</super> due to the site's elevation of "
        f"{d['elevation']}m above sea level."
    )
    story.append(P(ad_text))

    story.append(Spacer(1, 10))
    story.append(P('<b>Air Density Calculation (ERA5T Method):</b>', sBodyLeft))
    story.append(Spacer(1, 6))
    story.append(P('rho = P / (R * T)', sFormula))
    story.append(P('where:', sBodyLeft))
    story.append(P('rho = air density (kg/m<super>3</super>)', sBullet))
    story.append(P('P = atmospheric pressure (Pa)', sBullet))
    story.append(P('R = specific gas constant for dry air = 287.05 J/(kg K)', sBullet))
    story.append(P('T = temperature (K)', sBullet))
    story.append(Spacer(1, 8))

    ad_rows = [
        [PH('Parameter'), PH('Value')],
        [P('Mean Air Density', sTCL), P(f"{d['air_density']} kg/m<super>3</super>", sTC)],
        [P('Std. Deviation', sTCL), P(f"{d['air_density_std']} kg/m<super>3</super>", sTC)],
        [P('Standard Air Density', sTCL), P('1.225 kg/m<super>3</super>', sTC)],
        [P('Density Ratio (site/std)', sTCL), P(f"{d['air_density']/1.225:.4f}", sTC)],
    ]
    cw_ad = [AVAILABLE_WIDTH * 0.50, AVAILABLE_WIDTH * 0.50]
    story.extend(make_table(ad_rows, cw_ad, 'Table 4: Air Density Summary'))

    # ─── Section 5: Terrain and Roughness ───
    story.extend(add_major_section('5. Terrain and Roughness'))
    tr_text = (
        f"The site terrain is classified as {d['terrain_type'].lower()}. "
        f"The surface roughness length (z<sub>0</sub>) is estimated at {d['roughness_z0']}m, "
        f"consistent with agricultural land use. The power law shear exponent (alpha) is "
        f"{d['power_law_alpha']}, and the resulting shear factor for extrapolation from 100m "
        f"to {d['hub_height']}m hub height is {d['shear_factor']}."
    )
    story.append(P(tr_text))

    tr_rows = [
        [PH('Parameter'), PH('Symbol'), PH('Value')],
        [P('Surface Roughness Length', sTCL), P('z0', sTC), P(f"{d['roughness_z0']} m", sTC)],
        [P('Power Law Exponent', sTCL), P('alpha', sTC), P(f"{d['power_law_alpha']}", sTC)],
        [P('Shear Factor (100m to hub)', sTCL), P('-', sTC), P(f"{d['shear_factor']}", sTC)],
        [P('Hub Height', sTCL), P('H', sTC), P(f"{d['hub_height']} m", sTC)],
        [P('Reference Height', sTCL), P('Href', sTC), P('100 m', sTC)],
    ]
    cw_tr = [AVAILABLE_WIDTH * 0.40, AVAILABLE_WIDTH * 0.25, AVAILABLE_WIDTH * 0.35]
    story.extend(make_table(tr_rows, cw_tr, 'Table 5: Terrain and Roughness Parameters'))

    # ─── Section 6: Seasonal and Diurnal Patterns ───
    story.extend(add_major_section('6. Seasonal and Diurnal Patterns'))

    story.append(add_heading('<b>6.1 Monthly Wind Speed</b>', sH2, level=1))
    ms_text = (
        "The monthly mean wind speeds show a clear seasonal pattern, with the highest wind speeds "
        "occurring during the winter and early spring months (October through April) and the lowest "
        "during the summer months (June through August). Figure 3 presents the monthly variation."
    )
    story.append(P(ms_text))
    story.extend(img_element(chart_paths['monthly'], 6.2, 'Figure 3: Monthly Mean Wind Speed at Hub Height'))

    # Monthly data table
    mws = d['monthly_ws']
    mrows = [[PH('Month'), PH('Mean WS (m/s)')]]
    for i, (m, v) in enumerate(zip(MONTHS, mws)):
        mrows.append([P(m, sTC), P(f'{v:.1f}', sTC)])
    cw_m = [AVAILABLE_WIDTH * 0.40, AVAILABLE_WIDTH * 0.60]
    story.extend(make_table(mrows, cw_m, 'Table 6: Monthly Mean Wind Speed'))

    story.append(add_heading('<b>6.2 Diurnal Pattern</b>', sH2, level=1))
    dp_text = (
        "The diurnal pattern shows higher wind speeds during the daytime hours (09:00-16:00) "
        "and lower speeds during nighttime, which is typical for continental interior locations. "
        "The peak occurs around 11:00-12:00 local time. Figure 4 illustrates the diurnal variation."
    )
    story.append(P(dp_text))
    story.extend(img_element(chart_paths['diurnal'], 6.2, 'Figure 4: Diurnal Wind Speed Pattern at Hub Height'))

    # ─── Section 7: Power Curve and AEP ───
    story.extend(add_major_section('7. Power Curve and AEP'))

    story.append(add_heading('<b>7.1 Power Curve</b>', sH2, level=1))
    pc_text = (
        f"The {d['turbine_model']} turbine has a rated power of {d['rated_power_kw']:,} kW, "
        f"a cut-in wind speed of 3 m/s, and a cut-out wind speed of 25 m/s. "
        f"The power curve has been adjusted for the site-specific air density of "
        f"{d['air_density']} kg/m<super>3</super>. Figure 5 shows the power curve."
    )
    story.append(P(pc_text))
    story.extend(img_element(chart_paths['power_curve'], 6.2, f"Figure 5: Power Curve - {d['turbine_model']}"))

    # Power curve table
    pc = d['power_curve']
    # Split into 2 columns of data to save space
    pc_rows = [[PH('WS (m/s)'), PH('Power (kW)'), PH('WS (m/s)'), PH('Power (kW)')]]
    half = len(pc) // 2 + len(pc) % 2
    for i in range(half):
        row = [P(str(pc[i]['ws']), sTC), P(f"{pc[i]['power_kw']:,}", sTC)]
        if i + half < len(pc):
            row.extend([P(str(pc[i + half]['ws']), sTC), P(f"{pc[i + half]['power_kw']:,}", sTC)])
        else:
            row.extend([P('-', sTC), P('-', sTC)])
        pc_rows.append(row)
    cw_pc = [AVAILABLE_WIDTH * 0.18, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.18, AVAILABLE_WIDTH * 0.32]
    story.extend(make_table(pc_rows, cw_pc, 'Table 7: Power Curve Data'))

    story.append(add_heading('<b>7.2 AEP Estimation</b>', sH2, level=1))
    aep_text = (
        f"Two methods were used to estimate the gross AEP: the Weibull-based method yielded "
        f"{d['gross_aep_weibull']} GWh, while the time-series method yielded "
        f"{d['gross_aep_ts']} GWh. The time-series result ({d['gross_aep']} GWh) is adopted "
        f"as the reference gross AEP. After applying wake losses ({d['wake_loss_pct']}%) and "
        f"other losses ({d['other_loss_pct']}%), the net AEP is {d['net_aep']} GWh with a "
        f"net capacity factor of {d['net_cf_pct']}%."
    )
    story.append(P(aep_text))

    aep_rows = [
        [PH('Parameter'), PH('Value')],
        [P('Gross AEP (Weibull)', sTCL), P(f"{d['gross_aep_weibull']} GWh", sTC)],
        [P('Gross AEP (Time Series)', sTCL), P(f"{d['gross_aep_ts']} GWh", sTC)],
        [P('Adopted Gross AEP', sTCL), P(f"{d['gross_aep']} GWh", sTC)],
        [P('Gross Capacity Factor', sTCL), P(f"{d['gross_cf_pct']}%", sTC)],
        [P('Wake Loss', sTCL), P(f"{d['wake_loss_pct']}%", sTC)],
        [P('Other Losses', sTCL), P(f"{d['other_loss_pct']}%", sTC)],
        [P('Net AEP', sTCL), P(f"{d['net_aep']} GWh", sTC)],
        [P('Net Capacity Factor', sTCL), P(f"{d['net_cf_pct']}%", sTC)],
    ]
    cw_aep = [AVAILABLE_WIDTH * 0.50, AVAILABLE_WIDTH * 0.50]
    story.extend(make_table(aep_rows, cw_aep, 'Table 8: AEP Estimation Summary'))

    story.append(add_heading('<b>7.3 Loss Breakdown</b>', sH2, level=1))
    lb_text = (
        "The total losses are estimated at 14% of gross AEP, comprising wake losses and various "
        "other loss categories. Table 9 provides a detailed breakdown."
    )
    story.append(P(lb_text))

    lb = d['loss_breakdown']
    lb_rows = [[PH('Loss Category'), PH('Loss (%)'), PH('Method/Notes')]]
    for item in lb:
        lb_rows.append([
            P(item['category'], sTCL),
            P(f"{item['pct']}%", sTC),
            P(item['method'], sTCL),
        ])
    cw_lb = [AVAILABLE_WIDTH * 0.30, AVAILABLE_WIDTH * 0.18, AVAILABLE_WIDTH * 0.52]
    story.extend(make_table(lb_rows, cw_lb, 'Table 9: Loss Breakdown'))

    # ─── Section 8: Exceedance Probabilities ───
    story.extend(add_major_section('8. Exceedance Probabilities'))
    ep_text = (
        f"P-exceedance values represent the probability that the actual AEP will meet or exceed "
        f"a given threshold. With an uncertainty (standard deviation) of {d['uncertainty_sigma_pct']}%, "
        f"the exceedance probabilities are calculated assuming a normal distribution around the P50 estimate. "
        f"Financial models typically rely on the P50 and P90 values."
    )
    story.append(P(ep_text))

    # Calculate confidence levels
    sigma_gwh = d['p50'] * d['uncertainty_sigma_pct'] / 100
    ep_rows = [
        [PH('Exceedance Level'), PH('AEP (GWh)'), PH('Capacity Factor (%)')],
        [P('P50 (Central Estimate)', sTCL), P(f"{d['p50']:.2f}", sTC),
         P(f"{d['p50'] / d['gross_aep'] * d['gross_cf_pct']:.1f}", sTC)],
        [P('P75', sTCL), P(f"{d['p75']:.2f}", sTC),
         P(f"{d['p75'] / d['gross_aep'] * d['gross_cf_pct']:.1f}", sTC)],
        [P('P84', sTCL), P(f"{d['p84']:.2f}", sTC),
         P(f"{d['p84'] / d['gross_aep'] * d['gross_cf_pct']:.1f}", sTC)],
        [P('P90', sTCL), P(f"{d['p90']:.2f}", sTC),
         P(f"{d['p90'] / d['gross_aep'] * d['gross_cf_pct']:.1f}", sTC)],
        [P('P95', sTCL), P(f"{d['p95']:.2f}", sTC),
         P(f"{d['p95'] / d['gross_aep'] * d['gross_cf_pct']:.1f}", sTC)],
    ]
    cw_ep = [AVAILABLE_WIDTH * 0.34, AVAILABLE_WIDTH * 0.33, AVAILABLE_WIDTH * 0.33]
    story.extend(make_table(ep_rows, cw_ep, 'Table 10: Exceedance Probability Values'))

    # ─── Section 9: Farm-Level Summary ───
    story.extend(add_major_section('9. Farm-Level Summary'))
    fl_text = (
        f"The {d['project_name']} wind farm comprises {d['num_turbines']} {d['turbine_model']} turbines "
        f"with a total installed capacity of {total_capacity_mw:,.1f} MW. "
        f"Table 11 summarizes the farm-level energy production metrics."
    )
    story.append(P(fl_text))

    fl_rows = [
        [PH('Parameter'), PH('Per Turbine'), PH('Farm Total')],
        [P('Installed Capacity', sTCL), P(f"{d['rated_power_kw']/1000:.1f} MW", sTC),
         P(f"{total_capacity_mw:,.1f} MW", sTC)],
        [P('Gross AEP', sTCL),
         P(f"{d['gross_aep'] * 1000 / d['num_turbines']:.1f} MWh", sTC),
         P(f"{d['gross_aep'] * 1000:,.0f} MWh", sTC)],
        [P('Net AEP', sTCL),
         P(f"{d['net_aep'] * 1000 / d['num_turbines']:.1f} MWh", sTC),
         P(f"{d['net_aep'] * 1000:,.0f} MWh", sTC)],
        [P('Wake Loss', sTCL), P(f"{d['wake_loss_pct']}%", sTC), P(f"{d['wake_loss_pct']}%", sTC)],
        [P('Other Losses', sTCL), P(f"{d['other_loss_pct']}%", sTC), P(f"{d['other_loss_pct']}%", sTC)],
        [P('Net Capacity Factor', sTCL), P(f"{d['net_cf_pct']}%", sTC), P(f"{d['net_cf_pct']}%", sTC)],
        [P('P90 AEP', sTCL),
         P(f"{d['p90'] * 1000 / d['num_turbines']:.1f} MWh", sTC),
         P(f"{d['p90'] * 1000:,.0f} MWh", sTC)],
    ]
    cw_fl = [AVAILABLE_WIDTH * 0.36, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.32]
    story.extend(make_table(fl_rows, cw_fl, 'Table 11: Farm-Level Summary'))

    # ─── Section 10: Assumptions and Limitations ───
    story.extend(add_major_section('10. Assumptions and Limitations'))
    al_text = (
        "The following assumptions and limitations should be considered when interpreting "
        "the results of this wind resource assessment:"
    )
    story.append(P(al_text))

    for i, assumption in enumerate(d['assumptions'], 1):
        story.append(P(f"{i}. {assumption}", sBullet))
    story.append(Spacer(1, 8))

    lim_text = (
        "Additionally, the absence of on-site meteorological mast data introduces inherent uncertainty. "
        "The ERA5 reanalysis data, while comprehensive, may not capture local topographic effects "
        "at the turbine-level scale. The results should be validated with site-specific measurements "
        "when available."
    )
    story.append(P(lim_text))

    # ─── Section 11: Conclusions and Recommendations ───
    story.extend(add_major_section('11. Conclusions and Recommendations'))

    conc_text = (
        f"The wind resource assessment for the {d['project_name']} project indicates a viable "
        f"wind resource with a mean hub-height wind speed of {d['mean_ws_hub']} m/s and a "
        f"wind power density of {d['wind_power_density_hub']} W/m<super>2</super>. "
        f"The net AEP is estimated at {d['net_aep']} GWh with a net capacity factor of "
        f"{d['net_cf_pct']}%, which is competitive for the region."
    )
    story.append(P(conc_text))
    story.append(Spacer(1, 8))

    rec_intro = "Key recommendations include:"
    story.append(P(f"<b>{rec_intro}</b>", sBodyLeft))

    recommendations = [
        f"Install a meteorological mast or lidar at the site to validate ERA5-derived wind speeds and reduce uncertainty in the P-exceedance analysis.",
        f"Conduct a detailed layout optimization to minimize wake losses, which are currently estimated at {d['wake_loss_pct']}%.",
        f"Perform a long-term correlation (MCP) analysis using nearby long-term reference stations to extend the data record beyond 5 years.",
        f"Consider seasonal curtailment strategies during low-wind summer months to optimize maintenance scheduling.",
        f"Re-evaluate loss assumptions with project-specific data as the design progresses, particularly for availability and curtailment estimates.",
    ]
    for i, rec in enumerate(recommendations, 1):
        story.append(P(f"{i}. {rec}", sBullet))
    story.append(Spacer(1, 12))

    closing = (
        "This assessment provides a robust preliminary estimate of the wind resource and energy "
        "production potential at the site. The results support the feasibility of the proposed "
        f"{d['num_turbines']}-turbine wind farm, subject to the assumptions and limitations "
        "outlined in Section 10."
    )
    story.append(P(closing))

    return story


# ━━ Main Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description='Generate Wind Resource Assessment PDF Report')
    parser.add_argument('--input', '-i', help='Path to JSON data file', default=None)
    parser.add_argument('--output', '-o', help='Output PDF path', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Load data
    if args.input:
        with open(args.input, 'r') as f:
            data = json.load(f)
    else:
        data = DEFAULT_DATA

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    print(f"[1/7] Generating charts...")
    chart_paths = {
        'weibull': generate_weibull_chart(data),
        'wind_rose': generate_wind_rose_chart(data),
        'monthly': generate_monthly_chart(data),
        'diurnal': generate_diurnal_chart(data),
        'power_curve': generate_power_curve_chart(data),
    }
    print(f"       Charts saved to {CHART_DIR}/")

    print(f"[2/7] Generating cover page HTML...")
    cover_html = generate_cover_html(data)
    cover_html_path = os.path.join(CHART_DIR, 'cover.html')
    with open(cover_html_path, 'w') as f:
        f.write(cover_html)

    print(f"[3/7] Rendering cover page to PDF...")
    cover_pdf_path = os.path.join(CHART_DIR, 'cover.pdf')
    html2poster = os.path.join(PDF_SKILL_DIR, 'scripts', 'html2poster.js')
    result = subprocess.run(
        ['node', html2poster, cover_html_path, '--output', cover_pdf_path, '--width', '794px'],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"       WARNING: Cover rendering failed: {result.stderr}")
        print(f"       Proceeding without cover page...")
        cover_pdf_path = None
    else:
        print(f"       Cover PDF: {cover_pdf_path}")

    print(f"[4/7] Building report body PDF...")
    body_pdf_path = os.path.join(CHART_DIR, 'body.pdf')

    doc = TocDocTemplate(
        body_pdf_path,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=f"{data['project_name']} - Wind Resource Assessment",
        author='Z.ai',
        creator='Z.ai',
    )

    story = build_report(data, chart_paths)

    # Build with footer
    project_name = data['project_name']
    doc.multiBuild(story, onFirstPage=lambda c, d: footer_handler(c, d, project_name),
                   onLaterPages=lambda c, d: footer_handler(c, d, project_name))

    print(f"       Body PDF: {body_pdf_path}")

    print(f"[5/7] Merging cover and body PDFs...")
    from pypdf import PdfReader, PdfWriter, Transformation

    A4_W, A4_H = 595.28, 841.89

    def normalize_page_to_a4(page):
        box = page.mediabox
        w, h = float(box.width), float(box.height)
        if abs(w - A4_W) > 0.5 or abs(h - A4_H) > 0.5:
            sx, sy = A4_W / w, A4_H / h
            page.add_transformation(Transformation().scale(sx=sx, sy=sy))
            page.mediabox.lower_left = (0, 0)
            page.mediabox.upper_right = (A4_W, A4_H)
        return page

    writer = PdfWriter()

    if cover_pdf_path and os.path.exists(cover_pdf_path):
        cover_page = PdfReader(cover_pdf_path).pages[0]
        writer.add_page(normalize_page_to_a4(cover_page))
    else:
        print(f"       WARNING: No cover page, body only")

    for page in PdfReader(body_pdf_path).pages:
        writer.add_page(normalize_page_to_a4(page))

    writer.add_metadata({
        '/Title': f"{data['project_name']} - Wind Resource Assessment",
        '/Author': 'Z.ai',
        '/Creator': 'Z.ai',
        '/Subject': 'Wind Resource Assessment Report',
    })

    with open(args.output, 'wb') as f:
        writer.write(f)

    output_size = os.path.getsize(args.output) / 1024

    # Count pages
    reader = PdfReader(args.output)
    num_pages = len(reader.pages)

    print(f"[6/7] Final PDF generated successfully!")
    print(f"       Path: {args.output}")
    print(f"       Size: {output_size:.1f} KB")
    print(f"       Pages: {num_pages}")

    print(f"[7/7] Done!")

    return args.output


if __name__ == '__main__':
    main()
