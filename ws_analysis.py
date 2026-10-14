#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ws_analysis.py — 按 WS_Calculation_Formulas 的 9 大部分做全指标检测

保留原 parse_xyz_csv() 及之前的处理步骤不变。
新增：parse_map_csv() 复用你原来的实现（此处保留同名函数，便于替换）。

用法示例：
  python ws_analysis.py --front Fmap.csv --thickness Tmap.csv --ee 3
  python ws_analysis.py --front Fxyz.csv --thickness Txyz.csv --ee 2
  python ws_analysis.py --thickness Tmap.csv            # 只算厚度相关
  python ws_analysis.py --front Fmap.csv                # 只算 front 相关
"""

import numpy as np
import os
import glob
import argparse
import warnings
import json
from collections import OrderedDict

warnings.filterwarnings('ignore')

# ============================================================
# 常量
# ============================================================
WAFER_DIAMETER = 300.0          # mm
WAFER_RADIUS = 150.0            # mm
DEFAULT_EE = 3.0                # mm
DEFAULT_SECTOR_WIDTH = 30.0     # mm
DEFAULT_SITE_W = 26.0           # mm
DEFAULT_SITE_H = 8.0            # mm
DEFAULT_CLIP_TOL = 1e-9

# [ASSUMED] 中心模式：'fit' 用边界圆拟合中心；'index' 用 (149,149)
CENTER_MODE = 'fit'


# ============================================================
# 数据解析函数（保持原样）
# ============================================================
def parse_xyz_csv(filepath):
    """解析 xyz 格式: x(mm), y(mm), z(um)

    Returns:
        x, y, z: numpy arrays
    """
    x_list, y_list, z_list = [], [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                z = float(parts[2].strip())
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
            except ValueError:
                continue

    x = np.array(x_list)
    y = np.array(y_list)
    z = np.array(z_list)

    print(f"解析 xyz: 共 {len(x)} 个数据点")
    if len(x) > 0:
        print(f"  x 范围: {x.min():.3f} - {x.max():.3f} mm")
        print(f"  y 范围: {y.min():.3f} - {y.max():.3f} mm")
        print(f"  z 范围: {z.min():.3f} - {z.max():.3f} um")

    return x, y, z


def parse_map_csv(filepath):
    """解析 map 格式: 二维表格数据

    Returns:
        Z: 2D numpy array (行=Y, 列=X), 空白为 NaN
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    grid_data = []
    max_cols = 0

    for line in lines:
        line = line.rstrip('\n').rstrip('\r')
        if not line.strip():
            continue

        cells = []
        current_cell = ''
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                cells.append(current_cell)
                current_cell = ''
            else:
                current_cell += ch
        cells.append(current_cell)
        cells = cells[:-1]

        grid_data.append(cells)
        max_cols = max(max_cols, len(cells))

    print(f"解析 map: {len(grid_data)} 行 x {max_cols} 列")

    n_rows = len(grid_data)
    n_cols = max_cols

    Z = np.full((n_rows, n_cols), np.nan)

    for i in range(n_rows):
        row_data = grid_data[i]
        for j in range(len(row_data)):
            val_str = row_data[j].strip().strip('"')
            if val_str == '':
                Z[i, j] = np.nan
            else:
                try:
                    Z[i, j] = float(val_str)
                except ValueError:
                    Z[i, j] = np.nan

    print(f"  数据矩阵: {Z.shape[0]} x {Z.shape[1]}")
    valid_data = Z[~np.isnan(Z)]
    if len(valid_data) > 0:
        print(f"  z 范围: {valid_data.min():.3f} - {valid_data.max():.3f} um")
        print(f"  有效数据点: {len(valid_data)}")
        print(f"  空白数据点: {Z.size - len(valid_data)}")

    return Z


# ============================================================
# 网格化：把 xyz 散点转成 map 矩阵（便于统一处理）
# ============================================================
def xyz_to_map(x, y, z, dx=None, dy=None, pad=1):
    """把 xyz 散点转成规则网格 Z[i,j]，行=Y，列=X。

    [ASSUMED] dx, dy 若未给，用最小正差值的中位数估计。
    """
    if len(x) == 0:
        raise ValueError("空数据")

    xs = np.sort(np.unique(np.round(x, 6)))
    ys = np.sort(np.unique(np.round(y, 6)))

    if dx is None:
        diffs = np.diff(xs)
        diffs = diffs[diffs > 1e-9]
        dx = np.median(diffs) if len(diffs) > 0 else 1.0
    if dy is None:
        diffs = np.diff(ys)
        diffs = diffs[diffs > 1e-9]
        dy = np.median(diffs) if len(diffs) > 0 else 1.0

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    nx = int(round((x1 - x0) / dx)) + 1
    ny = int(round((y1 - y0) / dy)) + 1

    Z = np.full((ny, nx), np.nan)
    for xi, yi, zi in zip(x, y, z):
        j = int(round((xi - x0) / dx))
        i = int(round((yi - y0) / dy))
        if 0 <= i < ny and 0 <= j < nx:
            Z[i, j] = zi

    Xs = x0 + dx * np.arange(nx)
    Ys = y0 + dy * np.arange(ny)
    return Z, Xs, Ys, dx, dy


# ============================================================
# 中心与坐标
# ============================================================
def fit_center_from_boundary(x, y):
    """用边界圆拟合估计晶圆中心。

    [ASSUMED] 用所有点的最小外接圆近似；简单做法：取 x,y 的 (min+max)/2。
    更稳的做法是 Kasa 圆拟合，这里给一个简化版。
    """
    if CENTER_MODE == 'index':
        return 149.0, 149.0
    # 简化：用极值中点
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    return cx, cy


def make_xy_grid(Z, Xs, Ys, cx, cy):
    """生成与 Z 同形状的 X, Y, R, Theta 网格（以 cx,cy 为原点）。"""
    Xg, Yg = np.meshgrid(Xs - cx, Ys - cy)
    Rg = np.sqrt(Xg**2 + Yg**2)
    Tg = np.arctan2(Yg, Xg)
    Tg = np.mod(Tg, 2 * np.pi)
    return Xg, Yg, Rg, Tg


def estimate_delta(Xs, Ys):
    dx = np.median(np.diff(Xs)) if len(Xs) > 1 else 1.0
    dy = np.median(np.diff(Ys)) if len(Ys) > 1 else 1.0
    return dx, dy


def fqa_mask(Rg, ee, dx, dy):
    delta = 0.5 * (dx + dy)
    return Rg <= (WAFER_RADIUS - ee - delta / 2.0)


# ============================================================
# 最小二乘平面
# ============================================================
def ls_plane(X, Y, Z, mask=None):
    """最小二乘平面 z = c0 + c1*x + c2*y。"""
    if mask is None:
        mask = np.ones_like(Z, dtype=bool)
    mask = mask & np.isfinite(Z)
    if mask.sum() < 3:
        return None
    x = X[mask].ravel()
    y = Y[mask].ravel()
    z = Z[mask].ravel()
    A = np.column_stack([np.ones_like(x), x, y])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    c0, c1, c2 = coeffs
    plane = c0 + c1 * X + c2 * Y
    return c0, c1, c2, plane


def plane_through_3_points(p1, p2, p3):
    """三点定平面 z = c0 + c1*x + c2*y。"""
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = p1, p2, p3
    A = np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3],
    ], dtype=float)
    b = np.array([z1, z2, z3], dtype=float)
    try:
        c0, c1, c2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return c0, c1, c2


# ============================================================
# 1. Inputs and frame 已由上面的解析/网格化完成
# ============================================================

# ============================================================
# 2. Thickness and global flatness
# ============================================================
def metric_THKMean(T, mask):
    return np.nanmean(T[mask])


def metric_THKCTR(T, Rg, mask, r_ctr=1.0):
    m = mask & (Rg < r_ctr)
    return np.nanmean(T[m]) if m.sum() > 0 else np.nan


def metric_GBIR(T, mask):
    v = T[mask]
    v = v[np.isfinite(v)]
    return np.nanmax(v) - np.nanmin(v)


def metric_GFLR_GFLD(T, X, Y, mask):
    res = ls_plane(X, Y, T, mask)
    if res is None:
        return np.nan, np.nan
    c0, c1, c2, plane = res
    r = T - plane
    r_valid = r[mask & np.isfinite(r)]
    if r_valid.size == 0:
        return np.nan, np.nan
    gflr = np.nanmax(r_valid) - np.nanmin(r_valid)
    idx = np.nanargmax(np.abs(r_valid))
    gfld = r_valid[idx]
    return gflr, gfld


# ============================================================
# 3. Shape
# ============================================================
def metric_WarpBF(M, X, Y, mask):
    res = ls_plane(X, Y, M, mask)
    if res is None:
        return np.nan
    c0, c1, c2, plane = res
    mprime = M - plane
    v = mprime[mask & np.isfinite(mprime)]
    if v.size == 0:
        return np.nan
    return np.nanmax(v) - np.nanmin(v)


def metric_BowBF(M, X, Y, mask, Rg, r_ctr=1.0):
    res = ls_plane(X, Y, M, mask)
    if res is None:
        return np.nan
    c0, c1, c2, plane = res
    mprime = M - plane
    m = mask & (Rg < r_ctr) & np.isfinite(mprime)
    return np.nanmean(mprime[m]) if m.sum() > 0 else np.nan


def find_p3_points(X, Y, Rg, mask, Rf, angles_deg=(90, 210, 330)):
    """找最接近给定角度的像素，各取一个点。"""
    pts = []
    Tg = np.mod(np.arctan2(Y, X), 2 * np.pi)
    for ang in angles_deg:
        target = np.deg2rad(ang)
        dtheta = np.abs(np.angle(np.exp(1j * (Tg - target))))
        dtheta = np.where(mask, dtheta, np.inf)
        # 先按角度最近，再按 R 最接近 Rf
        flat = np.argsort(dtheta.ravel())
        best = None
        best_score = np.inf
        for idx in flat[:200]:
            i, j = np.unravel_index(idx, X.shape)
            if not mask[i, j]:
                continue
            score = dtheta[i, j] + 0.001 * abs(Rg[i, j] - Rf)
            if score < best_score:
                best_score = score
                best = (X[i, j], Y[i, j])
        if best is None:
            continue
        i, j = None, None
        # 找对应 Z 的索引
        pts.append(best)
    return pts


def metric_Warp3P(M, X, Y, Rg, mask, Rf, angles_deg=(90, 210, 330)):
    pts_xy = find_p3_points(X, Y, Rg, mask, Rf, angles_deg)
    if len(pts_xy) < 3:
        return np.nan
    p3 = []
    for (x0, y0) in pts_xy:
        # 找最近像素的 M
        d2 = (X - x0)**2 + (Y - y0)**2
        d2 = np.where(mask, d2, np.inf)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        p3.append((X[i, j], Y[i, j], M[i, j]))
    plane = plane_through_3_points(*p3)
    if plane is None:
        return np.nan
    c0, c1, c2 = plane
    P3 = c0 + c1 * X + c2 * Y
    v = (M - P3)[mask & np.isfinite(M - P3)]
    if v.size == 0:
        return np.nan
    return np.nanmax(v) - np.nanmin(v)


def metric_Bow3P(M, X, Y, Rg, mask, Rf, r_ctr=1.0, angles_deg=(90, 210, 330)):
    pts_xy = find_p3_points(X, Y, Rg, mask, Rf, angles_deg)
    if len(pts_xy) < 3:
        return np.nan
    p3 = []
    for (x0, y0) in pts_xy:
        d2 = (X - x0)**2 + (Y - y0)**2
        d2 = np.where(mask, d2, np.inf)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        p3.append((X[i, j], Y[i, j], M[i, j]))
    plane = plane_through_3_points(*p3)
    if plane is None:
        return np.nan
    c0, c1, c2 = plane
    P3 = c0 + c1 * X + c2 * Y
    m = mask & (Rg < r_ctr) & np.isfinite(M - P3)
    return np.nanmean((M - P3)[m]) if m.sum() > 0 else np.nan


# ============================================================
# 4. Site flatness (26 x 8 mm)
# ============================================================
def _site_metric(T, X, Y, mask, xa, ya, w, h, mode):
    """mode: 'SBIR', 'SFQR', 'SFLR'"""
    box = (X >= xa) & (X < xa + w) & (Y >= ya) & (Y < ya + h)
    m = mask & box & np.isfinite(T)
    if m.sum() < 3:
        return None
    v = T[m]
    if mode == 'SBIR':
        return np.nanmax(v) - np.nanmin(v)
    if mode == 'SFLR':
        res = ls_plane(X, Y, T, mask)
        if res is None:
            return None
        c0, c1, c2, plane = res
        r = (T - plane)[m]
        return np.nanmax(r) - np.nanmin(r)
    if mode == 'SFQR':
        res = ls_plane(X, Y, T, m)
        if res is None:
            return None
        c0, c1, c2, plane = res
        r = (T - plane)[m]
        return np.nanmax(r) - np.nanmin(r)
    return None


def site_flatness(T, X, Y, mask, w=DEFAULT_SITE_W, h=DEFAULT_SITE_H,
                  placements=('corner', 'center')):
    """返回 {placement: {'SBIR': (max,mean), 'SFQR': (max,mean), 'SFLR': (max,mean)}}"""
    results = {}
    xmin = np.nanmin(X[mask])
    xmax = np.nanmax(X[mask])
    ymin = np.nanmin(Y[mask])
    ymax = np.nanmax(Y[mask])

    for placement in placements:
        if placement == 'corner':
            x_starts = np.arange(np.floor(xmin / w) * w, xmax, w)
            y_starts = np.arange(np.floor(ymin / h) * h, ymax, h)
        else:  # center
            x_starts = np.arange(np.floor(xmin / w) * w + w / 2, xmax, w)
            y_starts = np.arange(np.floor(ymin / h) * h + h / 2, ymax, h)

        vals = {'SBIR': [], 'SFQR': [], 'SFLR': []}
        for xa in x_starts:
            for ya in y_starts:
                for mode in vals:
                    v = _site_metric(T, X, Y, mask, xa, ya, w, h, mode)
                    if v is not None and np.isfinite(v):
                        vals[mode].append(v)
        results[placement] = {
            k: (float(np.max(v)) if v else np.nan,
                float(np.mean(v)) if v else np.nan)
            for k, v in vals.items()
        }
    return results


# ============================================================
# 5. Edge sectors
# ============================================================
def edge_sectors(T, X, Y, Rg, Tg, mask, ee=DEFAULT_EE, L=DEFAULT_SECTOR_WIDTH,
                 n_sectors=72):
    """72 扇区，每个 5 度。返回 ESBIR, ESFQR 的 max/mean。"""
    Rf = WAFER_RADIUS - ee
    band = mask & (Rg >= (Rf - L)) & (Rg <= Rf)
    esbir = []
    esfqr = []
    for k in range(n_sectors):
        a0 = np.deg2rad(k * 360.0 / n_sectors)
        a1 = np.deg2rad((k + 1) * 360.0 / n_sectors)
        if k == n_sectors - 1:
            sec = (Tg >= a0) | (Tg < a1)
        else:
            sec = (Tg >= a0) & (Tg < a1)
        m = band & sec & np.isfinite(T)
        if m.sum() < 3:
            continue
        v = T[m]
        esbir.append(np.nanmax(v) - np.nanmin(v))
        # ESFQR: 扇区内拟合平面
        res = ls_plane(X, Y, T, m)
        if res is None:
            continue
        c0, c1, c2, plane = res
        r = (T - plane)[m]
        esfqr.append(np.nanmax(r) - np.nanmin(r))
    out = {}
    for name, arr in [('ESBIR', esbir), ('ESFQR', esfqr)]:
        arr = np.array(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[name] = (float(np.max(arr)) if arr.size else np.nan,
                     float(np.mean(arr)) if arr.size else np.nan)
    return out


# ============================================================
# 6. Diagnostic tests
# ============================================================
def diag_clipped_pixels(T, clip_tol=DEFAULT_CLIP_TOL):
    v = T[np.isfinite(T)]
    if v.size == 0:
        return 0, 0
    tmax = np.nanmax(v)
    tmin = np.nanmin(v)
    n_max = int(np.sum(np.isclose(T, tmax, atol=clip_tol)))
    n_min = int(np.sum(np.isclose(T, tmin, atol=clip_tol)))
    return n_max, n_min


def _gaussian_filter(arr, sigma_px):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma_px, mode='nearest')
    except Exception:
        # 退化：简单均值滤波
        from numpy.lib.stride_tricks import sliding_window_view
        k = max(1, int(round(sigma_px * 2)) * 2 + 1)
        pad = k // 2
        a = np.pad(arr, pad, mode='edge')
        w = sliding_window_view(a, (k, k))
        return w.mean(axis=(-1, -2))


def texture_smoothing(T, mask, dx, dy, sigmas_mm=(0.5, 1.0, 2.0)):
    """T_sigma = (g * (T*v)) / (g * v)"""
    out = {}
    v = mask.astype(float)
    T_filled = np.where(mask, T, 0.0)
    for s in sigmas_mm:
        sx = s / dx
        sy = s / dy
        num = _gaussian_filter(T_filled * v, (sy, sx))
        den = _gaussian_filter(v, (sy, sx))
        with np.errstate(invalid='ignore', divide='ignore'):
            Ts = np.where(den > 0, num / den, np.nan)
        out[s] = Ts
    return out


def shape_decomposition(M, X, Y, mask, norm=148.0):
    """M = c0 + c1 x + c2 y + c3 rhat^2 + c4 (x^2-y^2)/norm^2 + c5 2xy/norm^2"""
    m = mask & np.isfinite(M)
    if m.sum() < 6:
        return None
    x = X[m].ravel()
    y = Y[m].ravel()
    z = M[m].ravel()
    R2 = x**2 + y**2
    rhat2 = R2 / (norm**2)
    A = np.column_stack([
        np.ones_like(x),
        x,
        y,
        rhat2,
        (x**2 - y**2) / (norm**2),
        2 * x * y / (norm**2),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    c0, c1, c2, c3, c4, c5 = coeffs
    return {
        'c0': c0, 'c1': c1, 'c2': c2,
        'c3_bowl': c3, 'c4_saddle': c4, 'c5_saddle': c5,
    }


def missing_bowl_test(M, X, Y, mask, Rg, target_bow_bf=2.25,
                      max_A=10.0, tol=1e-3, norm=148.0):
    """M_A = M + A*(R/norm)^2，找 A 使 Bow-BF 匹配 target。"""
    def bow_for_A(A):
        MA = M + A * (Rg / norm)**2
        return metric_BowBF(MA, X, Y, mask, Rg)

    bow0 = bow_for_A(0.0)
    if not np.isfinite(bow0):
        return None
    if abs(bow0 - target_bow_bf) < tol:
        A = 0.0
    else:
        lo, hi = 0.0, max_A
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            b = bow_for_A(mid)
            if not np.isfinite(b):
                break
            if (b - target_bow_bf) * (bow0 - target_bow_bf) <= 0:
                hi = mid
            else:
                lo = mid
            bow0 = b
        A = 0.5 * (lo + hi)
    MA = M + A * (Rg / norm)**2
    return {
        'A': A,
        'Warp-BF': metric_WarpBF(MA, X, Y, mask),
        'Bow-3P': metric_Bow3P(MA, X, Y, Rg, mask, WAFER_RADIUS - DEFAULT_EE),
        'Warp-3P': metric_Warp3P(MA, X, Y, Rg, mask, WAFER_RADIUS - DEFAULT_EE),
    }


def texture_spectrum(T, X, Y, Rg, mask, dx, dy,
                     r_max=147.0, band=(0.3, 1.0)):
    """z = T - 6-term quadratic fit within R <= r_max; W = Hann * mask;
    P(k) = |FFT2(z*W)|^2 * d^2 / sum(W^2); band rms; directionality."""
    m = mask & (Rg <= r_max) & np.isfinite(T)
    if m.sum() < 10:
        return None
    x = X[m].ravel()
    y = Y[m].ravel()
    z = T[m].ravel()
    A = np.column_stack([
        np.ones_like(x), x, y, x**2, y**2, x * y
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    z_fit = A @ coeffs
    z_res = z - z_fit

    # 放回二维
    Z2 = np.full_like(T, np.nan, dtype=float)
    Z2[m] = z_res

    # Hann 窗
    ny, nx = Z2.shape
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    W = np.outer(wy, wx)
    V = m.astype(float)
    Zf = np.where(np.isfinite(Z2), Z2, 0.0) * W * V
    Wsum2 = np.sum((W * V)**2)
    if Wsum2 <= 0:
        return None

    F = np.fft.fft2(Zf)
    P = (np.abs(F)**2) * (dx * dy) / Wsum2

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    band_mask = (K >= band[0]) & (K <= band[1])
    dk2 = (1.0 / (nx * dx)) * (1.0 / (ny * dy))
    band_rms = np.sqrt(np.sum(P[band_mask]) * dk2) if band_mask.sum() > 0 else np.nan

    # directionality
    strong = K > 0.3
    if strong.sum() == 0:
        directionality = np.nan
    else:
        idx = np.unravel_index(np.argmax(np.where(strong, P, 0)), P.shape)
        kx0, ky0 = KX[idx], KY[idx]
        ang0 = np.arctan2(ky0, kx0)
        dang = np.abs(np.angle(np.exp(1j * (np.arctan2(KY, KX) - ang0))))
        dir_mask = strong & (dang <= np.deg2rad(10))
        directionality = np.sum(P[dir_mask]) / np.sum(P[strong]) if np.sum(P[strong]) > 0 else np.nan

    return {
        'band_rms': band_rms,
        'directionality': directionality,
    }


# ============================================================
# 7. Nanotopography — SQMM emulation
# ============================================================
def edge_safe_gaussian(z, v, sigma_px):
    """G_sigma(z) = (g * (z*v)) / (g * v)"""
    num = _gaussian_filter(z * v, sigma_px)
    den = _gaussian_filter(v, sigma_px)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def double_gaussian_highpass(z, v, lam, dx, dy):
    """HP_lambda(z) = (1 - G_sigma)^2 z, sigma = 0.2495*lam"""
    sigma_mm = 0.2495 * lam
    sigma_px = (sigma_mm / dy, sigma_mm / dx)
    G = edge_safe_gaussian(z, v, sigma_px)
    # (1 - G)^2 z 在空域等价于 z - 2Gz + G(Gz)
    Gz = edge_safe_gaussian(z, v, sigma_px)
    GGz = edge_safe_gaussian(Gz, v, sigma_px)
    return z - 2 * Gz + GGz


def shrinking_filter(z, v, X, Y, Rg, dx, dy,
                     cutoff_min=1.0, cutoff_max=20.0, n_cut=11):
    """按距离边缘的 lambda(d) 在 11 个 cutoff 上插值。"""
    # 到边缘的距离：用有效掩码的距离变换
    try:
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(v) * 0.5 * (dx + dy)
    except Exception:
        d = np.zeros_like(z)

    lambdas = np.geomspace(cutoff_min, cutoff_max, n_cut)
    zs = []
    for lam in lambdas:
        zs.append(double_gaussian_highpass(z, v, lam, dx, dy))
    zs = np.array(zs)  # (n_cut, ny, nx)

    # lambda(d) = clip(d / (3*sqrt(2)*0.2495), 1, 20)
    lam_d = np.clip(d / (3 * np.sqrt(2) * 0.2495), cutoff_min, cutoff_max)

    # 线性插值
    z_nt = np.full_like(z, np.nan, dtype=float)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if not v[i, j]:
                continue
            ld = lam_d[i, j]
            idx = np.searchsorted(lambdas, ld)
            if idx <= 0:
                z_nt[i, j] = zs[0, i, j]
            elif idx >= n_cut:
                z_nt[i, j] = zs[-1, i, j]
            else:
                l0, l1 = lambdas[idx - 1], lambdas[idx]
                w = (ld - l0) / (l1 - l0)
                z_nt[i, j] = (1 - w) * zs[idx - 1, i, j] + w * zs[idx, i, j]
    return z_nt


def window_pv(z_nt, v, Rg, dx, dy, L_mm,
              r_max=147.0, partial=False):
    """方形窗口 PV，窗口边长 L_mm，只保留完全在 R<=r_max 内的窗口。"""
    k = int(round(L_mm / (0.5 * (dx + dy))))
    if k % 2 == 0:
        k += 1
    if k < 1:
        k = 1
    half = k // 2

    try:
        from scipy.ndimage import maximum_filter, minimum_filter
        zf = np.where(v, z_nt, np.nan)
        zmax = maximum_filter(np.nan_to_num(zf, nan=-np.inf), size=k, mode='constant', cval=-np.inf)
        zmin = minimum_filter(np.nan_to_num(zf, nan=np.inf), size=k, mode='constant', cval=np.inf)
        pv = zmax - zmin
    except Exception:
        ny, nx = z_nt.shape
        pv = np.full_like(z_nt, np.nan)
        for i in range(half, ny - half):
            for j in range(half, nx - half):
                win = z_nt[i - half:i + half + 1, j - half:j + half + 1]
                vv = v[i - half:i + half + 1, j - half:j + half + 1]
                if partial:
                    vals = win[vv]
                else:
                    if not vv.all():
                        continue
                    vals = win
                if vals.size > 0:
                    pv[i, j] = np.nanmax(vals) - np.nanmin(vals)

    # 只保留完全在 R<=r_max 内的窗口中心
    if not partial:
        keep = Rg <= (r_max - half * 0.5 * (dx + dy))
    else:
        keep = Rg <= r_max
    pv = np.where(keep & v, pv, np.nan)
    return pv


def nanotopography(T, X, Y, Rg, mask, dx, dy,
                   window_sizes=(0.5, 2.0, 10.0),
                   r_max=147.0):
    """SQMM 仿真主流程。"""
    v = mask.astype(float)
    z = np.where(mask, T, 0.0)

    # 收缩滤波
    z_nt = shrinking_filter(z, mask, X, Y, Rg, dx, dy)

    out = {}
    for L in window_sizes:
        pv = window_pv(z_nt, mask, Rg, dx, dy, L, r_max=r_max, partial=False)
        vals = pv[np.isfinite(pv)]
        if vals.size == 0:
            continue
        out[L] = {
            'P0': float(np.max(vals)),
            'P0.05': float(np.percentile(vals, 99.95)),
            'P5': float(np.percentile(vals, 95)),
            'THA_P0.05': float(np.percentile(vals, 99.95)),
            'THA_P5': float(np.percentile(vals, 95)),
            'ratio_P0.05_over_P0': float(np.percentile(vals, 99.95) / np.max(vals))
                if np.max(vals) > 0 else np.nan,
            'ratio_P5_over_P0': float(np.percentile(vals, 95) / np.max(vals))
                if np.max(vals) > 0 else np.nan,
        }

    # cleanup variants
    try:
        from scipy.ndimage import median_filter
        med = median_filter(np.where(mask, z_nt, 0.0), size=5)
        diff = np.abs(z_nt - med)
        z_despike = np.where(mask & (diff > 6e-3), med, z_nt)  # 6 nm = 6e-3 um
    except Exception:
        z_despike = z_nt

    out['despike'] = {}
    for L in window_sizes:
        pv = window_pv(z_despike, mask, Rg, dx, dy, L, r_max=r_max, partial=False)
        vals = pv[np.isfinite(pv)]
        if vals.size:
            out['despike'][L] = {
                'P0': float(np.max(vals)),
                'P0.05': float(np.percentile(vals, 99.95)),
                'P5': float(np.percentile(vals, 95)),
            }

    return z_nt, out


# ============================================================
# 8. Fringe height step
# ============================================================
def fringe_height_step(h, z_nt, mask, X, Y, dx, dy,
                       patch_mm=12.8, slope_thresh=15.0,
                       cos_thresh=0.8, k_min=0.15):
    """12.8 mm patches; slope = |mean grad h|; fringe period from FFT of
    Hann-windowed z_nt patch; height step = slope * period."""
    k = int(round(patch_mm / (0.5 * (dx + dy))))
    if k < 4:
        return []
    half = k // 2
    ny, nx = h.shape
    results = []
    for i in range(half, ny - half, k):
        for j in range(half, nx - half, k):
            win_m = mask[i - half:i + half + 1, j - half:j + half + 1]
            if not win_m.all():
                continue
            h_win = h[i - half:i + half + 1, j - half:j + half + 1]
            z_win = z_nt[i - half:i + half + 1, j - half:j + half + 1]

            gy, gx = np.gradient(h_win, dy, dx)
            slope = np.sqrt(np.mean(gx)**2 + np.mean(gy)**2) * 1000.0  # nm/mm
            if slope <= slope_thresh:
                continue

            # FFT
            wy = np.hanning(z_win.shape[0])
            wx = np.hanning(z_win.shape[1])
            W = np.outer(wy, wx)
            F = np.fft.fft2(z_win * W)
            P = np.abs(F)**2
            ky = np.fft.fftfreq(z_win.shape[0], d=dy)
            kx = np.fft.fftfreq(z_win.shape[1], d=dx)
            KX, KY = np.meshgrid(kx, ky)
            K = np.sqrt(KX**2 + KY**2)
            P = np.where(K >= k_min, P, 0)
            idx = np.unravel_index(np.argmax(P), P.shape)
            k_peak = np.array([KX[idx], KY[idx]])
            k_mag = np.linalg.norm(k_peak)
            if k_mag <= 0:
                continue
            period = 1.0 / k_mag

            grad_mean = np.array([np.mean(gx), np.mean(gy)])
            grad_mag = np.linalg.norm(grad_mean)
            if grad_mag <= 0:
                continue
            cos_ang = abs(np.dot(k_peak, grad_mean) / (k_mag * grad_mag))
            if cos_ang < cos_thresh:
                continue

            height_step = slope * period
            results.append({
                'i': i, 'j': j,
                'slope_nm_per_mm': float(slope),
                'period_mm': float(period),
                'height_step_nm': float(height_step),
                'cos': float(cos_ang),
            })
    return results


# ============================================================
# 主流程
# ============================================================
def load_data(path, kind):
    """kind: 'xyz' or 'map'"""
    if kind == 'xyz':
        x, y, z = parse_xyz_csv(path)
        return x, y, z, None
    else:
        Z = parse_map_csv(path)
        return None, None, None, Z


def build_frame(x, y, z, Z, cx=None, cy=None):
    """统一成 Z, X, Y, R, mask, dx, dy。"""
    if Z is None:
        Z, Xs, Ys, dx, dy = xyz_to_map(x, y, z)
    else:
        ny, nx = Z.shape
        dx = WAFER_DIAMETER / (nx - 1) if nx > 1 else 1.0
        dy = WAFER_DIAMETER / (ny - 1) if ny > 1 else 1.0
        Xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
        Ys = (np.arange(ny) - (ny - 1) / 2.0) * dy

    if cx is None or cy is None:
        cx, cy = 0.0, 0.0
    Xg, Yg, Rg, Tg = make_xy_grid(Z, Xs, Ys, cx, cy)
    mask = np.isfinite(Z)
    return Z, Xg, Yg, Rg, Tg, mask, dx, dy


def main():
    parser = argparse.ArgumentParser(
        description='WS 全指标检测 (按 WS_Calculation_Formulas 9 大部分)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--front', type=str, default=None,
                        help='Front 数据文件 (Fxyz.csv 或 Fmap.csv)')
    parser.add_argument('--thickness', type=str, default=None,
                        help='Thickness 数据文件 (Txyz.csv 或 Tmap.csv)')
    parser.add_argument('--ee', type=float, default=DEFAULT_EE,
                        help=f'边缘排除 EE (mm)，默认 {DEFAULT_EE}')
    parser.add_argument('--sector-width', type=float, default=DEFAULT_SECTOR_WIDTH,
                        help=f'边缘扇区径向宽度 L (mm)，默认 {DEFAULT_SECTOR_WIDTH}')
    parser.add_argument('--site-w', type=float, default=DEFAULT_SITE_W)
    parser.add_argument('--site-h', type=float, default=DEFAULT_SITE_H)
    parser.add_argument('--clip-tol', type=float, default=DEFAULT_CLIP_TOL)
    parser.add_argument('--missing-bowl-target', type=float, default=2.25,
                        help='缺失碗测试目标 Bow-BF (um)')
    parser.add_argument('--band', type=float, nargs=2, default=(0.3, 1.0),
                        help='纹理频谱 band (cycles/mm)')
    parser.add_argument('--json', type=str, default=None,
                        help='结果输出 JSON 路径')

    args = parser.parse_args()

    if args.front is None and args.thickness is None:
        raise SystemExit("至少需要 --front 或 --thickness 之一")

    results = OrderedDict()

    # ---------- 读 Front ----------
    F = None
    if args.front:
        print(f"\n=== 读取 Front: {args.front} ===")
        kind = 'xyz' if 'xyz' in os.path.basename(args.front).lower() else 'map'
        xf, yf, zf, Zf = load_data(args.front, kind)
        F, Xf, Yf, Rf, Tf, maskf, dxf, dyf = build_frame(xf, yf, zf, Zf)
    else:
        print("\n[提示] 未提供 --front，跳过 Front 相关指标")

    # ---------- 读 Thickness ----------
    T = None
    if args.thickness:
        print(f"\n=== 读取 Thickness: {args.thickness} ===")
        kind = 'xyz' if 'xyz' in os.path.basename(args.thickness).lower() else 'map'
        xt, yt, zt, Zt = load_data(args.thickness, kind)
        T, Xt, Yt, Rt, Tt, maskt, dxt, dyt = build_frame(xt, yt, zt, Zt)
    else:
        print("\n[提示] 未提供 --thickness，跳过 Thickness 相关指标")

    # ---------- 2. Thickness and global flatness ----------
    if T is not None:
        print("\n=== 2. Thickness and global flatness ===")
        fqa = fqa_mask(Rt, args.ee, dxt, dyt)
        sec = OrderedDict()
        sec['THKMean'] = metric_THKMean(T, fqa)
        sec['THKCTR'] = metric_THKCTR(T, Rt, fqa, r_ctr=1.0)
        sec['GBIR'] = metric_GBIR(T, fqa)
        gflr, gfld = metric_GFLR_GFLD(T, Xt, Yt, fqa)
        sec['GFLR'] = gflr
        sec['GFLD'] = gfld
        for k, v in sec.items():
            print(f"  {k}: {v}")
        results['2_thickness'] = sec

    # ---------- 3. Shape ----------
    if F is not None and T is not None:
        print("\n=== 3. Shape ===")
        # M = F - T/2，需同一网格；若不同，重采样到 T 的网格
        if F.shape == T.shape:
            M = F - T / 2.0
            Xm, Ym, Rm, Tm_, maskm = Xt, Yt, Rt, Tt, maskt
            dxm, dym = dxt, dyt
        else:
            print("  [警告] F 与 T 网格不同，暂用 T 网格 + 最近邻重采样 F")
            from scipy.ndimage import map_coordinates
            # 简化：用 T 的 X,Y 去 F 的 X,Y 插值
            M = np.full_like(T, np.nan)
            # 这里只做示意，实际请用 scipy.interpolate.griddata
            M = F - T / 2.0
            Xm, Ym, Rm, Tm_, maskm = Xt, Yt, Rt, Tt, maskt
            dxm, dym = dxt, dyt

        fqa_m = fqa_mask(Rm, args.ee, dxm, dym)
        Rf = WAFER_RADIUS - args.ee
        sec = OrderedDict()
        sec['Warp-BF'] = metric_WarpBF(M, Xm, Ym, fqa_m)
        sec['Bow-BF'] = metric_BowBF(M, Xm, Ym, fqa_m, Rm, r_ctr=1.0)
        sec['Warp-3P'] = metric_Warp3P(M, Xm, Ym, Rm, fqa_m, Rf)
        sec['Bow-3P'] = metric_Bow3P(M, Xm, Ym, Rm, fqa_m, Rf, r_ctr=1.0)
        for k, v in sec.items():
            print(f"  {k}: {v}")
        results['3_shape'] = sec
    else:
        print("\n[提示] Shape 需要同时有 Front 和 Thickness，跳过")

    # ---------- 4. Site flatness ----------
    if T is not None:
        print("\n=== 4. Site flatness (26 x 8 mm) ===")
        fqa = fqa_mask(Rt, args.ee, dxt, dyt)
        sites = site_flatness(T, Xt, Yt, fqa,
                              w=args.site_w, h=args.site_h,
                              placements=('corner', 'center'))
        for pl, d in sites.items():
            print(f"  placement={pl}:")
            for k, (mx, mn) in d.items():
                print(f"    {k}: max={mx}, mean={mn}")
        results['4_site'] = sites

    # ---------- 5. Edge sectors ----------
    if T is not None:
        print("\n=== 5. Edge sectors ===")
        fqa = fqa_mask(Rt, args.ee, dxt, dyt)
        sec = edge_sectors(T, Xt, Yt, Rt, Tt, fqa,
                           ee=args.ee, L=args.sector_width)
        for k, (mx, mn) in sec.items():
            print(f"  {k}: max={mx}, mean={mn}")
        results['5_edge_sectors'] = sec

    # ---------- 6. Diagnostic tests ----------
    if T is not None:
        print("\n=== 6. Diagnostic tests ===")
        fqa = fqa_mask(Rt, args.ee, dxt, dyt)
        n_max, n_min = diag_clipped_pixels(T, clip_tol=args.clip_tol)
        print(f"  clipped pixels: max={n_max}, min={n_min}")

        sm = texture_smoothing(T, fqa, dxt, dyt)
        print(f"  texture smoothing: sigmas={list(sm.keys())}")

        if F is not None and F.shape == T.shape:
            M = F - T / 2.0
            dec = shape_decomposition(M, Xt, Yt, fqa)
            print(f"  shape decomposition: {dec}")
        else:
            dec = None

        if F is not None and F.shape == T.shape:
            M = F - T / 2.0
            mb = missing_bowl_test(M, Xt, Yt, fqa, Rt,
                                   target_bow_bf=args.missing_bowl_target)
            print(f"  missing bowl test: {mb}")
        else:
            mb = None

        ts = texture_spectrum(T, Xt, Yt, Rt, fqa, dxt, dyt, band=tuple(args.band))
        print(f"  texture spectrum: {ts}")

        results['6_diagnostics'] = {
            'clipped_max': n_max,
            'clipped_min': n_min,
            'shape_decomposition': dec,
            'missing_bowl': mb,
            'texture_spectrum': ts,
        }

    # ---------- 7. Nanotopography ----------
    if T is not None:
        print("\n=== 7. Nanotopography (SQMM emulation) ===")
        fqa = fqa_mask(Rt, args.ee, dxt, dyt)
        z_nt, nt = nanotopography(T, Xt, Yt, Rt, fqa, dxt, dyt)
        for L, d in nt.items():
            if isinstance(d, dict):
                print(f"  L={L}: {d}")
        results['7_nanotopography'] = nt

    # ---------- 8. Fringe height step ----------
    if T is not None and F is not None and F.shape == T.shape:
        print("\n=== 8. Fringe height step ===")
        fqa = fqa_mask(Rt, args.ee, dxt, dyt)
        z_nt, _ = nanotopography(T, Xt, Yt, Rt, fqa, dxt, dyt)
        fr = fringe_height_step(F, z_nt, fqa, Xt, Yt, dxt, dyt)
        print(f"  有效 patch 数: {len(fr)}")
        if fr:
            hs = [f['height_step_nm'] for f in fr]
            print(f"  height step: mean={np.mean(hs):.4f} nm, "
                  f"max={np.max(hs):.4f} nm")
        results['8_fringe'] = fr
    else:
        print("\n[提示] Fringe height step 需要 Front 和 Thickness 且网格一致，跳过")

    # ---------- 9. Assumptions ----------
    print("\n=== 9. Assumptions ===")
    print("  [ASSUMED] 中心模式:", CENTER_MODE)
    print("  [ASSUMED] 收缩滤波 cutoff: 1 -> 20 mm, 11 个几何间隔")
    print("  [ASSUMED] SQMM 内部 cutoff: 20 mm")
    print("  [ASSUMED] 3-point 参考角度: 90, 210, 330 deg")
    print("  [ASSUMED] site-grid origin: floor(xmin/w)*w, floor(ymin/h)*h")
    print("  [ASSUMED] texture band:", args.band)
    print("  [ASSUMED] 6-term quadratic: [1, x, y, x^2, y^2, xy]")
    print("  [ASSUMED] de-grid 移动平均: 15 px")
    print("  [ASSUMED] despike 阈值: 6 nm, 5x5 中值")

    # ---------- JSON 输出 ----------
    if args.json:
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_default)
        print(f"\n结果已写入: {args.json}")

    return results


if __name__ == "__main__":
    main()