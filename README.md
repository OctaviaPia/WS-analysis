# Wafer Surface Analysis

A collection of Python tools for wafer surface topography analysis, flatness metrology, and WS (Wafer Specification) metric computation on 300 mm wafers.

## Files

### `ws_analysis.py`

Full WS metric detection script covering the 9 major sections of the WS Calculation Formulas.

Features:
- Parse XYZ CSV (x, y, z) and map CSV (2D grid) input formats
- Grid conversion from scattered XYZ points to regular map matrix
- Center estimation and polar coordinate frame (R, Theta)
- Thickness and global flatness metrics: THKMean, THKCTR, GBIR, GFLR, GFLD
- Shape metrics: Warp-BF, Bow-BF, Warp-3P, Bow-3P
- Site flatness (26 x 8 mm): SBIR, SFQR, SFLR with corner and center placements
- Edge sectors: ESBIR, ESFQR over 72 sectors
- Diagnostic tests: clipped pixels, texture smoothing, shape decomposition, missing bowl test, texture spectrum
- Nanotopography emulation (SQMM): shrinking filter, window PV, cleanup variants
- Fringe height step detection over 12.8 mm patches
- JSON result export

Usage:

```
python ws_analysis.py --front Fmap.csv --thickness Tmap.csv --ee 3
python ws_analysis.py --front Fxyz.csv --thickness Txyz.csv --ee 2
python ws_analysis.py --thickness Tmap.csv
python ws_analysis.py --front Fmap.csv
```

### `main_tkinter.py`

Tkinter GUI for the WS wafer analysis workflow. Does not require PySide6 or PyQt.

Features:
- File pickers for Front (F) and Thickness (T) data
- Adjustable assumption parameters: EE, site width/height, sector width
- Background compute thread with progress bar and cancel button
- Left panel: heatmap or contour view of the loaded wafer map
- Right panel: tabbed metric tables for sections 2 through 6
- Custom blue-green-yellow-red colormap

Run:

```
python main_tkinter.py
```

### `plane_analysis.py`

Flatness analysis for the central 50 x 50 mm^2 square region of a 300 mm wafer.

Metrics:
- TTV: total thickness variation against the base (lowest point) plane
- FP: least squares fitted plane coefficients
- FPD_POS / FPD_NEG: max positive / negative deviation from the fitted plane
- SFQR: peak-to-valley relative to the fitted plane
- SBIR: peak-to-valley relative to the base plane

Modes:
- fx: Fxyz.csv (Front, xyz)
- fm: Fmap.csv (Front, map, default)
- tx: Txyz.csv (Thickness, xyz)
- tm: Tmap.csv (Thickness, map)

Usage:

```
python plane_analysis.py -d ./SUBLOT_ID -m fm
python plane_analysis.py -d ./SUBLOT_ID -m tx
```

## Requirements

- Python 3.8 or later
- numpy
- matplotlib (for `main_tkinter.py`)
- scipy (optional, improves filtering and distance transform performance)

## Input Data Format

XYZ format: three comma-separated columns per line, x (mm), y (mm), z (um).

Map format: two-dimensional table of comma-separated values. Empty cells are treated as NaN. Rows correspond to Y, columns to X.

## Assumptions

Several numeric conventions are marked as `[ASSUMED]` in the source and printed at the end of `ws_analysis.py`. Key ones include:

- Center mode for boundary fitting: `fit`
- Shrinking filter cutoffs: 1 mm to 20 mm, 11 geometric steps
- SQMM internal cutoff: 20 mm
- 3-point reference angles: 90, 210, 330 degrees
- Site grid origin at floor(xmin/w)*w, floor(ymin/h)*h
- Texture band: 0.3 to 1.0 cycles/mm
- Six-term quadratic basis: [1, x, y, x^2, y^2, xy]
- Despike threshold: 6 nm with a 5x5 median filter

## Notes

- All spatial units are millimeters unless stated otherwise.
- All height units are micrometers unless stated otherwise.
- Grid axes follow the convention: rows are Y, columns are X.
- The GUI uses a simplified shape metric path. Full 3-point and nanotopography metrics are available through `ws_analysis.py`.
