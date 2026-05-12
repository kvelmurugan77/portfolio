#!/usr/bin/env python3
"""
PyInstaller Build Script for WindFarm Designer Pro
===================================================
Packages the Python application into a standalone .exe (Windows) or
App (macOS) or binary (Linux).

Usage:
    python build_exe.py              # Build with default settings
    python build_exe.py --debug      # Build with debug symbols
    python build_exe.py --onefile    # Build as single .exe file
    python build_exe.py --onedir     # Build as directory (recommended)
    python build_exe.py --clean      # Clean build artifacts first

Prerequisites:
    pip install pyinstaller

Output:
    dist/WindFarm_Designer_Pro/      # Application directory
    dist/WindFarm_Designer_Pro.exe   # Or single executable
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

APP_NAME = "WindFarm_Designer_Pro"
APP_VERSION = "1.0.0"
APP_AUTHOR = "WindFarm Designer Pro Team"
APP_DESCRIPTION = "Wind Farm Design and Energy Assessment Tool"
APP_ICON = None  # Path to .ico file (set if available)

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MAIN_SCRIPT = SCRIPT_DIR / "main.py"
SRC_DIR = SCRIPT_DIR / "src"

# PyInstaller options
ONEDIR = True          # True = directory, False = single file
CONSOLE = False        # True = console window, False = GUI-only
DEBUG = False           # True = include debug symbols
UPX = True             # True = compress with UPX (if installed)
CLEAN = False          # True = clean build artifacts first


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("ERROR: PyInstaller is not installed.")
        print("Install with: pip install pyinstaller")
        return False


def clean_build():
    """Clean previous build artifacts."""
    build_dir = SCRIPT_DIR / "build"
    dist_dir = SCRIPT_DIR / "dist"

    for d in [build_dir, dist_dir]:
        if d.exists():
            print(f"Cleaning: {d}")
            shutil.rmtree(d, ignore_errors=True)

    spec_file = SCRIPT_DIR / f"{APP_NAME}.spec"
    if spec_file.exists():
        print(f"Cleaning: {spec_file}")
        spec_file.unlink()


def build_executable():
    """Run PyInstaller to build the executable."""
    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--version-file", "version.txt",
    ]

    # One-file vs one-directory
    if ONEDIR:
        cmd.append("--onedir")
    else:
        cmd.append("--onefile")

    # Console vs windowed
    if CONSOLE or DEBUG:
        cmd.append("--console")
    else:
        cmd.append("--windowed" if platform.system() == "Darwin" else "--noconsole")

    # Debug mode
    if DEBUG:
        cmd.append("--debug")
    else:
        cmd.append("--strip")

    # UPX compression
    if UPX:
        cmd.append("--upx-dir")
        cmd.append("")  # Use system UPX

    # Icon
    if APP_ICON and Path(APP_ICON).exists():
        cmd.extend(["--icon", APP_ICON])

    # Data files (include any resource files)
    # cmd.extend(["--add-data", "resources:resources"])

    # Include the entire src package
    cmd.extend(["--paths", str(SCRIPT_DIR)])

    # Hidden imports (PyInstaller sometimes misses these)
    hidden_imports = [
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "numpy",
        "scipy",
        "scipy.special",
        "scipy.integrate",
        "scipy.ndimage",
        "scipy.interpolate",
        "matplotlib",
        "matplotlib.backends",
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_pdf",
        "requests",
        "shapely",
        "shapely.geometry",
        "shapely.ops",
        "rasterio",
        "pyproj",
        "src",
        "src.core",
        "src.core.srtm_downloader",
        "src.core.roughness_downloader",
        "src.core.gwa_downloader",
        "src.core.layout_optimizer",
        "src.core.wake_model",
        "src.core.aep_calculator",
        "src.core.wind_flow_model",
        "src.gui",
        "src.utils",
        "src.utils.geo_utils",
        "src.utils.data_utils",
    ]

    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    # Collect submodules
    cmd.extend(["--collect-submodules", "matplotlib"])
    cmd.extend(["--collect-submodules", "scipy"])
    cmd.extend(["--collect-submodules", "PyQt5"])

    # Main script
    cmd.append(str(MAIN_SCRIPT))

    # Set working directory
    cwd = os.getcwd()

    print("=" * 60)
    print("Building WindFarm Designer Pro")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python:   {sys.version}")
    print(f"Mode:     {'Directory' if ONEDIR else 'Single File'}")
    print(f"Console:  {'Yes' if (CONSOLE or DEBUG) else 'No'}")
    print(f"Debug:    {'Yes' if DEBUG else 'No'}")
    print("=" * 60)
    print()

    print("Running PyInstaller...")
    print(f"Command: {' '.join(cmd[:10])}...")

    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: Build failed: {e}")
        return False


def post_build():
    """Post-build steps: copy additional files, create shortcuts, etc."""
    dist_dir = SCRIPT_DIR / "dist" / APP_NAME

    if not dist_dir.exists():
        print("WARNING: Build output not found.")
        return

    # Create a README in the dist directory
    readme = dist_dir / "README.txt"
    with open(readme, 'w') as f:
        f.write(f"WindFarm Designer Pro v{APP_VERSION}\n")
        f.write("=" * 50 + "\n\n")
        f.write("To run the application, execute:\n")
        if platform.system() == "Windows":
            f.write("  WindFarm_Designer_Pro.exe\n")
        else:
            f.write("  ./WindFarm_Designer_Pro\n")
        f.write("\nFor the first launch, it may take a moment to initialize.\n")

    print(f"\nBuild complete!")
    print(f"Output location: {dist_dir}")

    if platform.system() == "Windows":
        exe_path = dist_dir / f"{APP_NAME}.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"Executable: {exe_path} ({size_mb:.1f} MB)")
    elif platform.system() == "Darwin":
        app_path = dist_dir / APP_NAME
        print(f"Application: {app_path}")
    else:
        bin_path = dist_dir / APP_NAME
        if bin_path.exists():
            size_mb = bin_path.stat().st_size / (1024 * 1024)
            print(f"Binary: {bin_path} ({size_mb:.1f} MB)")


def create_version_file():
    """Create version.txt for PyInstaller."""
    version_file = SCRIPT_DIR / "version.txt"
    with open(version_file, 'w') as f:
        f.write(f"VSVersionInfo(\n")
        f.write(f"  ffi=FixedFileInfo(\n")
        f.write(f"    filevers=({APP_VERSION}, 0),\n")
        f.write(f"    prodvers=({APP_VERSION}, 0),\n")
        f.write(f"    mask=0x3f,\n")
        f.write(f"    flags=0x0,\n")
        f.write(f"    OS=0x40004,\n")
        f.write(f"    fileType=0x1,\n")
        f.write(f"    subtype=0x0,\n")
        f.write(f"    date=(0, 0)\n")
        f.write(f"  ),\n")
        f.write(f"  kids=[\n")
        f.write(f"    StringFileInfo(\n")
        f.write(f"      [StringTable(\n")
        f.write(f'        u\'040904B0\',\n')
        f.write(f'        [StringStruct(u\'CompanyName\', u\'{APP_AUTHOR}\'),\n')
        f.write(f'         StringStruct(u\'FileDescription\', u\'{APP_DESCRIPTION}\'),\n')
        f.write(f"         StringStruct(u'FileVersion', u'{APP_VERSION}'),\n")
        f.write(f'         StringStruct(u\'InternalName\', u\'{APP_NAME}\'),\n')
        f.write(f'         StringStruct(u\'OriginalFilename\', u\'{APP_NAME}.exe\'),\n')
        f.write(f'         StringStruct(u\'ProductName\', u\'WindFarm Designer Pro\'),\n')
        f.write(f'         StringStruct(u\'ProductVersion\', u\'{APP_VERSION}\')])]),\n')
        f.write(f"    VarFileInfo([VarStruct(u\'Translation\', [1033, 1200])])\n")
        f.write(f"  ]\n")
        f.write(f")\n")


def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description='Build WindFarm Designer Pro executable')
    parser.add_argument('--onefile', action='store_true', help='Build as single file')
    parser.add_argument('--onedir', action='store_true', help='Build as directory (default)')
    parser.add_argument('--console', action='store_true', help='Include console window')
    parser.add_argument('--debug', action='store_true', help='Debug build')
    parser.add_argument('--no-upx', action='store_true', help='Disable UPX compression')
    parser.add_argument('--clean', action='store_true', help='Clean build first')

    args = parser.parse_args()

    # Apply arguments
    global ONEDIR, CONSOLE, DEBUG, UPX, CLEAN
    if args.onedir:
        ONEDIR = True
    if args.onefile:
        ONEDIR = False
    if args.console:
        CONSOLE = True
    if args.debug:
        DEBUG = True
    if args.no_upx:
        UPX = False
    if args.clean:
        CLEAN = True

    # Step 1: Clean
    if CLEAN:
        print("Cleaning previous build...")
        clean_build()

    # Step 2: Check PyInstaller
    if not check_pyinstaller():
        sys.exit(1)

    # Step 3: Create version file
    create_version_file()

    # Step 4: Build
    success = build_executable()

    # Step 5: Post-build
    if success:
        post_build()
    else:
        print("\nERROR: Build failed. Check the output above for errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()
