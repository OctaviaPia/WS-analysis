#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
晶圆平面度分析脚本（300mm 晶圆中心 50x50mm^2 方形区域）

指标:
  TTV      - 基础平面(最低点水平面)下的总厚度变化
  FP       - 最小二乘拟合平面 (可含斜率)
  FPD_POS  - 相对拟合平面的正最大偏移
  FPD_NEG  - 相对拟合平面的负最大偏移
  SFQR     - 相对拟合平面的峰谷值 (max - min)
  SBIR     - 相对基础平面的峰谷值 (max - min)

模式:
  fx  - Fxyz.csv  (Front,    xyz)
  fm  - Fmap.csv  (Front,    map)  默认
  tx  - Txyz.csv  (Thickness, xyz)
  tm  - Tmap.csv  (Thickness, map)
"""

import numpy as np
import os
import glob
import argparse
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 常量
# ============================================================
WAFER_DIAMETER = 300.0          # mm
CENTER_BOX_SIZE = 25.0          # mm, 中心方形区域边长
HALF_BOX = CENTER_BOX_SIZE / 2.0  # 25 mm


# ============================================================
# 数据解析函数 (从 plot_thk_new.py 移植)
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

        # 按逗号分割（处理引号）
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
        cells = cells[:-1]  # 去掉尾部空单元格

        grid_data.append(cells)
        max_cols = max(max_cols, len(cells))

    print(f"解析 map: {len(grid_data)} 行 x {max_cols} 列")

    data_start_row = 0
    data_start_col = 0

    n_rows = len(grid_data) - data_start_row
    n_cols = max_cols - data_start_col

    Z = np.full((n_rows, n_cols), np.nan)

    for i in range(data_start_row, len(grid_data)):
        row_data = grid_data[i]
        for j in range(data_start_col, len(row_data)):
            val_str = row_data[j].strip().strip('"')
            if val_str == '':
                Z[i - data_start_row, j - data_start_col] = np.nan
            else:
                try:
                    Z[i - data_start_row, j - data_start_col] = float(val_str)
                except ValueError:
                    Z[i - data_start_row, j - data_start_col] = np.nan

    print(f"  数据矩阵: {Z.shape[0]} x {Z.shape[1]}")
    valid_data = Z[~np.isnan(Z)]
    if len(valid_data) > 0:
        print(f"  z 范围: {valid_data.min():.3f} - {valid_data.max():.3f} um")
        print(f"  有效数据点: {len(valid_data)}")
        print(f"  空白数据点: {Z.size - len(valid_data)}")

    return Z


# ============================================================
# 中心区域提取
# ============================================================
def extract_center_box_xyz(x, y, z):
    """从 xyz 散点中提取中心 50x50mm^2 区域

    Returns:
        xc, yc, zc: 区域内的点
    """
    mask = (np.abs(x) <= HALF_BOX) & (np.abs(y) <= HALF_BOX)
    return x[mask], y[mask], z[mask]


def extract_center_box_map(Z, wafer_diameter=WAFER_DIAMETER,
                           box_size=CENTER_BOX_SIZE):
    """从 map 矩阵中提取中心 50x50mm^2 区域

    假设 map 矩阵覆盖整个晶圆直径 (wafer_diameter)，
    计算中心 box_size 对应的行列范围。

    Returns:
        Zc: 中心区域子矩阵
        xc, yc: 对应的物理坐标 (mm), 以晶圆中心为原点
    """
    n_rows, n_cols = Z.shape

    # 每像素对应的物理尺寸 (mm)
    # 假设矩阵宽/高对应晶圆直径，且为正方形像素
    # 如果行列数不同，分别计算
    if n_cols > 1:
        px_x = wafer_diameter / (n_cols - 1)
    else:
        px_x = wafer_diameter
    if n_rows > 1:
        px_y = wafer_diameter / (n_rows - 1)
    else:
        px_y = wafer_diameter

    # 中心像素索引
    cx = (n_cols - 1) / 2.0
    cy = (n_rows - 1) / 2.0

    # 物理坐标 (以中心为原点)
    xs = (np.arange(n_cols) - cx) * px_x
    ys = (np.arange(n_rows) - cy) * px_y

    # 中心 50x50 区域掩码
    x_mask = np.abs(xs) <= HALF_BOX
    y_mask = np.abs(ys) <= HALF_BOX

    Zc = Z[np.ix_(y_mask, x_mask)]
    xc = xs[x_mask]
    yc = ys[y_mask]

    return Zc, xc, yc


# ============================================================
# 平面拟合与指标计算
# ============================================================
def fit_plane_lsq(x, y, z):
    """最小二乘拟合平面 z = a*x + b*y + c

    Returns:
        a, b, c: 平面系数
        residual: z - (a*x + b*y + c)
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    # lstsq 求解 [a, b, c]^T
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    residual = z - (a * x + b * y + c)
    return a, b, c, residual


def compute_metrics_xyz(x, y, z):
    """对 xyz 数据计算平面度指标

    Returns:
        dict: 指标字典
    """
    if len(z) < 3:
        raise ValueError("有效数据点不足 (<3)，无法拟合平面")

    # ---- 基础平面: 厚度最低点所在水平面 ----
    z_min = np.min(z)
    z_max = np.max(z)
    base_plane = z_min  # z = z_min 水平面

    # TTV: 相对基础平面的总厚度变化
    ttv = z_max - z_min

    # ---- FP: 最小二乘拟合平面 ----
    a, b, c, residual = fit_plane_lsq(x, y, z)

    # ---- FPD_POS / FPD_NEG: 相对拟合平面的正负最大偏移 ----
    fpd_pos = np.max(residual)
    fpd_neg = np.min(residual)

    # ---- SFQR: 相对拟合平面的峰谷值 ----
    sfqr = fpd_pos - fpd_neg

    # ---- SBIR: 相对基础平面的峰谷值 ----
    sbir = z_max - z_min

    return {
        'TTV': ttv,
        'FP_a': a,
        'FP_b': b,
        'FP_c': c,
        'FPD_POS': fpd_pos,
        'FPD_NEG': fpd_neg,
        'SFQR': sfqr,
        'SBIR': sbir,
        'z_min': z_min,
        'z_max': z_max,
        'n_points': len(z),
    }


def compute_metrics_map(Zc, xc, yc):
    """对 map 中心区域矩阵计算平面度指标

    将二维矩阵展平为 (x, y, z) 点集后调用 xyz 版本。

    Returns:
        dict: 指标字典
    """
    # 构建网格坐标
    Xg, Yg = np.meshgrid(xc, yc)  # Xg, Yg 与 Zc 同形状

    # 展平并去掉 NaN
    x_flat = Xg.ravel()
    y_flat = Yg.ravel()
    z_flat = Zc.ravel()

    valid = ~np.isnan(z_flat)
    x_v = x_flat[valid]
    y_v = y_flat[valid]
    z_v = z_flat[valid]

    print(f"  中心区域有效点数: {len(z_v)}")

    if len(z_v) < 3:
        raise ValueError("中心区域有效数据点不足 (<3)，无法拟合平面")

    return compute_metrics_xyz(x_v, y_v, z_v)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='晶圆平面度分析 (300mm 晶圆中心 50x50mm^2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python plane_analysis.py -d ./SUBLOT_ID -m fm
  python plane_analysis.py -d ./SUBLOT_ID -m tx
        """
    )

    parser.add_argument('-m', '--mode', type=str, default='fm',
                        choices=['fx', 'fm', 'tx', 'tm'],
                        help='数据模式: fx=Fxyz.csv, fm=Fmap.csv, '
                             'tx=Txyz.csv, tm=Tmap.csv (默认 fm)')
    parser.add_argument('-d', '--dir', type=str, required=True,
                        help='数据文件目录 (SUBLOT ID)')

    args = parser.parse_args()

    print(f"参数设置:")
    print(f"  mode: {args.mode}")
    print(f"  dir : {args.dir}")
    print()

    # ---- 根据模式确定文件名与类型 ----
    mode_map = {
        'fx': ('Fxyz.csv', 'xyz'),
        'fm': ('Fmap.csv', 'map'),
        'tx': ('Txyz.csv', 'xyz'),
        'tm': ('Tmap.csv', 'map'),
    }
    target_file, data_type = mode_map[args.mode]

    # ---- 查找文件 ----
    csv_path = os.path.join(args.dir, target_file)
    if not os.path.exists(csv_path):
        # 回退: 模糊匹配
        pattern = os.path.join(args.dir, '*.csv')
        candidates = glob.glob(pattern)
        if data_type == 'xyz':
            candidates = [f for f in candidates
                          if 'map' not in os.path.basename(f).lower()]
        else:
            candidates = [f for f in candidates
                          if 'map' in os.path.basename(f).lower()]
        if not candidates:
            raise FileNotFoundError(
                f"在 {args.dir} 下未找到 {target_file}")
        csv_path = candidates[0]

    print(f"读取文件: {csv_path}")

    # ---- 解析 + 计算 ----
    print(f"\n--- 提取中心 {CENTER_BOX_SIZE}x{CENTER_BOX_SIZE} mm^2 区域 ---")
    if data_type == 'xyz':
        x, y, z = parse_xyz_csv(csv_path)
        xc, yc, zc = extract_center_box_xyz(x, y, z)
        print(f"  中心区域点数: {len(zc)}")
        if len(zc) < 3:
            raise ValueError("中心区域有效数据点不足 (<3)")
        metrics = compute_metrics_xyz(xc, yc, zc)
    else:
        Z = parse_map_csv(csv_path)
        Zc, xc, yc = extract_center_box_map(Z)
        print(f"  中心区域矩阵: {Zc.shape[0]} x {Zc.shape[1]}")
        metrics = compute_metrics_map(Zc, xc, yc)

    # ---- 输出结果 ----
    print("\n" + "=" * 50)
    print(f"平面度分析结果  (mode={args.mode}, file={os.path.basename(csv_path)})")
    print("=" * 50)
    print(f"  TTV       : {metrics['TTV']:.4f} um")
    print(f"  FP 平面   : z = {metrics['FP_a']:.6f}*x "
          f"+ {metrics['FP_b']:.6f}*y + {metrics['FP_c']:.4f}")
    print(f"  FPD_POS   : {metrics['FPD_POS']:.4f} um")
    print(f"  FPD_NEG   : {metrics['FPD_NEG']:.4f} um")
    print(f"  SFQR      : {metrics['SFQR']:.4f} um")
    print(f"  SBIR      : {metrics['SBIR']:.4f} um")
    print("-" * 50)
    print(f"  z_min     : {metrics['z_min']:.4f} um")
    print(f"  z_max     : {metrics['z_max']:.4f} um")
    print(f"  有效点数  : {metrics['n_points']}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    main()