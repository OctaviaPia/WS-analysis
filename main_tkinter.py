#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WS 晶圆分析 GUI — tkinter 版（不依赖 PySide6/PyQt）

左侧: 热图 / 等高线
右侧: 标签页展示 2~8 部分指标
顶部: 选择 F/T 文件, 调 Assumption 参数
后台线程计算, 进度条更新
"""

import os
import sys
import time
import traceback
import threading
import queue
from dataclasses import dataclass

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 参数
# ============================================================
@dataclass
class Assumptions:
    ee: float = 3.0
    sector_width: float = 30.0
    site_w: float = 26.0
    site_h: float = 8.0
    clip_tol: float = 1e-9
    missing_bowl_target: float = 2.25
    band_lo: float = 0.3
    band_hi: float = 1.0


def get_custom_cmap():
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    return LinearSegmentedColormap.from_list('rygb', colors, N=256)


# ============================================================
# 解析函数
# ============================================================
def parse_xyz_csv(filepath):
    x_list, y_list, z_list = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    x_list.append(float(parts[0].strip()))
                    y_list.append(float(parts[1].strip()))
                    z_list.append(float(parts[2].strip()))
                except ValueError:
                    continue
    return np.array(x_list), np.array(y_list), np.array(z_list)


def parse_map_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    grid_data = []
    max_cols = 0
    for line in lines:
        line = line.rstrip('\n').rstrip('\r')
        if not line.strip():
            continue
        cells = []
        cur = ''
        in_q = False
        for ch in line:
            if ch == '"':
                in_q = not in_q
            elif ch == ',' and not in_q:
                cells.append(cur)
                cur = ''
            else:
                cur += ch
        cells.append(cur)
        cells = cells[:-1]
        grid_data.append(cells)
        max_cols = max(max_cols, len(cells))

    Z = np.full((len(grid_data), max_cols), np.nan)
    for i, row in enumerate(grid_data):
        for j, cell in enumerate(row):
            s = cell.strip().strip('"')
            if s == '':
                continue
            try:
                Z[i, j] = float(s)
            except ValueError:
                pass
    return Z


def load_any(path):
    """自动判断 xyz / map。返回 (x, y, z) 或 (None, None, Z)"""
    name = os.path.basename(path).lower()
    if 'xyz' in name:
        return parse_xyz_csv(path)
    if 'map' in name:
        return None, None, parse_map_csv(path)
    # 兜底：先试 map
    try:
        Z = parse_map_csv(path)
        if Z.size > 0 and np.isfinite(Z).sum() > 0:
            return None, None, Z
    except Exception:
        pass
    return parse_xyz_csv(path)


# ============================================================
# 网格化
# ============================================================
def build_frame(x, y, z, Z):
    if Z is None:
        xs = np.sort(np.unique(np.round(x, 6)))
        ys = np.sort(np.unique(np.round(y, 6)))
        dx = np.median(np.diff(xs)) if len(xs) > 1 else 1.0
        dy = np.median(np.diff(ys)) if len(ys) > 1 else 1.0
        nx = int(round((xs.max() - xs.min()) / dx)) + 1
        ny = int(round((ys.max() - ys.min()) / dy)) + 1
        Z = np.full((ny, nx), np.nan)
        for xi, yi, zi in zip(x, y, z):
            j = int(round((xi - xs.min()) / dx))
            i = int(round((yi - ys.min()) / dy))
            if 0 <= i < ny and 0 <= j < nx:
                Z[i, j] = zi
        Xs = xs.min() + dx * np.arange(nx)
        Ys = ys.min() + dy * np.arange(ny)
    else:
        ny, nx = Z.shape
        dx = 300.0 / (nx - 1) if nx > 1 else 1.0
        dy = 300.0 / (ny - 1) if ny > 1 else 1.0
        Xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
        Ys = (np.arange(ny) - (ny - 1) / 2.0) * dy

    Xg, Yg = np.meshgrid(Xs, Ys)
    Rg = np.sqrt(Xg**2 + Yg**2)
    mask = np.isfinite(Z)
    return Z, Xg, Yg, Rg, mask, dx, dy


# ============================================================
# 指标
# ============================================================
def ls_plane(X, Y, Z, mask):
    m = mask & np.isfinite(Z)
    if m.sum() < 3:
        return None
    x = X[m].ravel()
    y = Y[m].ravel()
    z = Z[m].ravel()
    A = np.column_stack([np.ones_like(x), x, y])
    c, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    plane = c[0] + c[1] * X + c[2] * Y
    return c, plane


def metrics_thickness(T, X, Y, R, fqa):
    v = T[fqa & np.isfinite(T)]
    out = {}
    out['THKMean'] = float(np.mean(v)) if v.size else np.nan
    ctr = fqa & (R < 1.0) & np.isfinite(T)
    out['THKCTR'] = float(np.mean(T[ctr])) if ctr.sum() else np.nan
    out['GBIR'] = float(np.max(v) - np.min(v)) if v.size else np.nan
    res = ls_plane(X, Y, T, fqa)
    if res is not None:
        c, plane = res
        r = (T - plane)[fqa & np.isfinite(T)]
        out['GFLR'] = float(np.max(r) - np.min(r)) if r.size else np.nan
        if r.size:
            idx = np.argmax(np.abs(r))
            out['GFLD'] = float(r[idx])
    return out


def metrics_shape(M, X, Y, R, fqa, Rf):
    out = {}
    res = ls_plane(X, Y, M, fqa)
    if res is None:
        return out
    c, plane = res
    mp = M - plane
    v = mp[fqa & np.isfinite(mp)]
    out['Warp-BF'] = float(np.max(v) - np.min(v)) if v.size else np.nan
    ctr = fqa & (R < 1.0) & np.isfinite(mp)
    out['Bow-BF'] = float(np.mean(mp[ctr])) if ctr.sum() else np.nan
    out['Warp-3P'] = np.nan  # 简化，完整版见 ws_analysis
    out['Bow-3P'] = np.nan
    return out


def site_flatness_fast(T, X, Y, mask, dx, dy, w, h):
    ny, nx = T.shape
    jw = max(1, int(round(w / dx)))
    ih = max(1, int(round(h / dy)))
    res = {'corner': {'SBIR': [], 'SFQR': [], 'SFLR': []},
           'center': {'SBIR': [], 'SFQR': [], 'SFLR': []}}

    # 全局平面
    global_plane = None
    g = ls_plane(X, Y, T, mask)
    if g is not None:
        global_plane = g[1]

    for placement in ('corner', 'center'):
        j_off = 0 if placement == 'corner' else jw // 2
        i_off = 0 if placement == 'corner' else ih // 2
        for i0 in range(i_off, ny - ih + 1, ih):
            for j0 in range(j_off, nx - jw + 1, jw):
                sm = mask[i0:i0 + ih, j0:j0 + jw]
                if sm.sum() < 3:
                    continue
                sT = T[i0:i0 + ih, j0:j0 + jw]
                v = sT[sm]
                res[placement]['SBIR'].append(float(np.max(v) - np.min(v)))

                sX = X[i0:i0 + ih, j0:j0 + jw][sm]
                sY = Y[i0:i0 + ih, j0:j0 + jw][sm]
                A = np.column_stack([np.ones_like(sX), sX, sY])
                c, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
                r = v - A @ c
                res[placement]['SFQR'].append(float(np.max(r) - np.min(r)))

                if global_plane is not None:
                    gv = global_plane[i0:i0 + ih, j0:j0 + jw][sm]
                    r2 = v - gv
                    res[placement]['SFLR'].append(float(np.max(r2) - np.min(r2)))

    out = {}
    for pl, d in res.items():
        out[pl] = {}
        for k, arr in d.items():
            arr = np.array(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            out[pl][k] = (float(np.max(arr)) if arr.size else np.nan,
                          float(np.mean(arr)) if arr.size else np.nan)
    return out


def edge_sectors(T, X, Y, R, Theta, mask, ee, L, n_sectors=72):
    Rf = 150.0 - ee
    band = mask & (R >= Rf - L) & (R <= Rf) & np.isfinite(T)
    esbir, esfqr = [], []
    for k in range(n_sectors):
        a0 = np.deg2rad(k * 360.0 / n_sectors)
        a1 = np.deg2rad((k + 1) * 360.0 / n_sectors)
        if k == n_sectors - 1:
            sec = (Theta >= a0) | (Theta < a1)
        else:
            sec = (Theta >= a0) & (Theta < a1)
        m = band & sec
        if m.sum() < 3:
            continue
        v = T[m]
        esbir.append(float(np.max(v) - np.min(v)))
        res = ls_plane(X, Y, T, m)
        if res is None:
            continue
        c, plane = res
        r = (T - plane)[m]
        esfqr.append(float(np.max(r) - np.min(r)))
    out = {}
    for name, arr in [('ESBIR', esbir), ('ESFQR', esfqr)]:
        arr = np.array(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[name] = (float(np.max(arr)) if arr.size else np.nan,
                     float(np.mean(arr)) if arr.size else np.nan)
    return out


def diag_clipped(T, tol):
    v = T[np.isfinite(T)]
    if v.size == 0:
        return 0, 0
    tmax, tmin = np.max(v), np.min(v)
    return int(np.sum(np.isclose(T, tmax, atol=tol))), \
           int(np.sum(np.isclose(T, tmin, atol=tol)))


# ============================================================
# 计算线程
# ============================================================
class ComputeThread(threading.Thread):
    def __init__(self, front_path, thick_path, asm, out_queue):
        super().__init__(daemon=True)
        self.front_path = front_path
        self.thick_path = thick_path
        self.asm = asm
        self.q = out_queue
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _emit(self, p, msg):
        self.q.put(('progress', p, msg))

    def run(self):
        try:
            t0 = time.time()
            results = {}

            self._emit(5, "读取 Front...")
            F = None
            if self.front_path:
                _, _, F = load_any(self.front_path)
            self._emit(15, "读取 Thickness...")
            T = None
            if self.thick_path:
                _, _, T = load_any(self.thick_path)

            if F is None and T is None:
                self.q.put(('error', "未提供 F 或 T"))
                return

            Zref = T if T is not None else F
            Z, Xg, Yg, Rg, mask, dx, dy = build_frame(None, None, None, Zref)
            if F is not None and F.shape != Z.shape:
                F = F[:Z.shape[0], :Z.shape[1]]
            if T is not None and T.shape != Z.shape:
                T = T[:Z.shape[0], :Z.shape[1]]

            delta = 0.5 * (dx + dy)
            fqa = Rg <= (150.0 - self.asm.ee - delta / 2.0)
            Theta = np.mod(np.arctan2(Yg, Xg), 2 * np.pi)
            Rf = 150.0 - self.asm.ee

            if T is not None:
                self._emit(25, "厚度与全局平坦度...")
                results['2_thickness'] = metrics_thickness(T, Xg, Yg, Rg, fqa)

            if F is not None and T is not None:
                self._emit(40, "形状...")
                M = F - T / 2.0
                results['3_shape'] = metrics_shape(M, Xg, Yg, Rg, fqa, Rf)

            if T is not None:
                self._emit(55, "站点平坦度...")
                results['4_site'] = site_flatness_fast(
                    T, Xg, Yg, fqa, dx, dy, self.asm.site_w, self.asm.site_h)

                self._emit(70, "边缘扇区...")
                results['5_edge'] = edge_sectors(
                    T, Xg, Yg, Rg, Theta, fqa, self.asm.ee, self.asm.sector_width)

                self._emit(85, "诊断...")
                nmax, nmin = diag_clipped(T, self.asm.clip_tol)
                results['6_diag'] = {'clipped_max': nmax, 'clipped_min': nmin}

            results['elapsed'] = time.time() - t0
            self._emit(100, "完成")
            self.q.put(('done', results))

        except Exception as e:
            self.q.put(('error', f"{e}\n{traceback.format_exc()}"))


# ============================================================
# GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WS 晶圆分析")
        self.geometry("1500x900")

        self.front_var = tk.StringVar()
        self.thick_var = tk.StringVar()
        self.ee_var = tk.DoubleVar(value=3.0)
        self.sw_var = tk.DoubleVar(value=26.0)
        self.sh_var = tk.DoubleVar(value=8.0)
        self.sl_var = tk.DoubleVar(value=30.0)

        self.queue = queue.Queue()
        self.worker = None
        self.last_results = None
        self.last_Z = None

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        top = ttk.LabelFrame(self, text="参数与文件")
        top.pack(fill='x', padx=6, pady=4)

        ttk.Label(top, text="Front:").grid(row=0, column=0, sticky='e', padx=4, pady=2)
        ttk.Entry(top, textvariable=self.front_var, width=60).grid(row=0, column=1, sticky='we', padx=4)
        ttk.Button(top, text="选择...", command=lambda: self._pick(self.front_var)).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="Thickness:").grid(row=1, column=0, sticky='e', padx=4, pady=2)
        ttk.Entry(top, textvariable=self.thick_var, width=60).grid(row=1, column=1, sticky='we', padx=4)
        ttk.Button(top, text="选择...", command=lambda: self._pick(self.thick_var)).grid(row=1, column=2, padx=4)

        pf = ttk.Frame(top)
        pf.grid(row=2, column=0, columnspan=3, sticky='we', padx=4, pady=2)
        ttk.Label(pf, text="EE (mm):").pack(side='left')
        ttk.Spinbox(pf, from_=0, to=10, increment=0.5, textvariable=self.ee_var, width=6).pack(side='left', padx=4)
        ttk.Label(pf, text="Site W:").pack(side='left')
        ttk.Spinbox(pf, from_=1, to=100, increment=1, textvariable=self.sw_var, width=6).pack(side='left', padx=4)
        ttk.Label(pf, text="Site H:").pack(side='left')
        ttk.Spinbox(pf, from_=1, to=100, increment=1, textvariable=self.sh_var, width=6).pack(side='left', padx=4)
        ttk.Label(pf, text="Sector L:").pack(side='left')
        ttk.Spinbox(pf, from_=1, to=100, increment=1, textvariable=self.sl_var, width=6).pack(side='left', padx=4)

        self.run_btn = ttk.Button(pf, text="运行", command=self.run)
        self.run_btn.pack(side='left', padx=10)
        self.cancel_btn = ttk.Button(pf, text="取消", command=self.cancel, state='disabled')
        self.cancel_btn.pack(side='left')

        # 中部
        mid = ttk.Panedwindow(self, orient='horizontal')
        mid.pack(fill='both', expand=True, padx=6, pady=4)

        # 左图
        left = ttk.Frame(mid)
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        mid.add(left, weight=1)

        # 右 Tab
        right = ttk.Frame(mid)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill='both', expand=True)
        self.trees = {}
        for name in ['2 厚度', '3 形状', '4 站点', '5 边缘', '6 诊断']:
            frame = ttk.Frame(self.nb)
            tree = ttk.Treeview(frame, columns=('max', 'mean'), show='tree headings')
            tree.heading('#0', text='指标')
            tree.heading('max', text='max')
            tree.heading('mean', text='mean')
            tree.column('#0', width=180)
            tree.column('max', width=120)
            tree.column('mean', width=120)
            tree.pack(fill='both', expand=True)
            self.trees[name] = tree
            self.nb.add(frame, text=name)
        mid.add(right, weight=1)

        # 底部
        bottom = ttk.Frame(self)
        bottom.pack(fill='x', padx=6, pady=4)
        self.progress = ttk.Progressbar(bottom, orient='horizontal', mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True)
        self.status = ttk.Label(bottom, text="就绪", width=40, anchor='w')
        self.status.pack(side='left', padx=8)

        # 视图切换
        viewf = ttk.Frame(self)
        viewf.pack(fill='x', padx=6)
        self.view_mode = tk.StringVar(value='heatmap')
        ttk.Radiobutton(viewf, text='热图', variable=self.view_mode, value='heatmap',
                        command=self.redraw).pack(side='left')
        ttk.Radiobutton(viewf, text='等高线', variable=self.view_mode, value='contour',
                        command=self.redraw).pack(side='left')

    def _pick(self, var):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            var.set(p)

    def run(self):
        f = self.front_var.get().strip() or None
        t = self.thick_var.get().strip() or None
        if not f and not t:
            messagebox.showwarning("提示", "至少选择 F 或 T 文件")
            return
        asm = Assumptions(ee=self.ee_var.get(),
                          sector_width=self.sl_var.get(),
                          site_w=self.sw_var.get(),
                          site_h=self.sh_var.get())
        self.run_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')
        self.progress['value'] = 0
        self.status.config(text="开始...")
        self.worker = ComputeThread(f, t, asm, self.queue)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == 'progress':
                    _, p, s = msg
                    self.progress['value'] = p
                    self.status.config(text=s)
                elif msg[0] == 'done':
                    self.last_results = msg[1]
                    self.run_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')
                    self.status.config(text=f"完成，用时 {self.last_results.get('elapsed', 0):.1f} s")
                    self.fill_tables(self.last_results)
                    self.redraw()
                elif msg[0] == 'error':
                    self.run_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')
                    messagebox.showerror("错误", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def fill_tables(self, results):
        mapping = {
            '2 厚度': results.get('2_thickness', {}),
            '3 形状': results.get('3_shape', {}),
            '4 站点': results.get('4_site', {}),
            '5 边缘': results.get('5_edge', {}),
            '6 诊断': results.get('6_diag', {}),
        }
        for tab, data in mapping.items():
            tree = self.trees[tab]
            for i in tree.get_children():
                tree.delete(i)
            for k, v in data.items():
                if isinstance(v, tuple):
                    tree.insert('', 'end', text=str(k),
                                values=(f"{v[0]:.6g}", f"{v[1]:.6g}"))
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, tuple):
                            tree.insert('', 'end', text=f"{k}/{k2}",
                                        values=(f"{v2[0]:.6g}", f"{v2[1]:.6g}"))
                        else:
                            tree.insert('', 'end', text=f"{k}/{k2}", values=(str(v2), ''))
                elif isinstance(v, (int, float, np.floating)):
                    tree.insert('', 'end', text=str(k), values=(f"{float(v):.6g}", ''))
                else:
                    tree.insert('', 'end', text=str(k), values=(str(v), ''))

    def redraw(self):
        # 从文件重读并画图（演示用，可缓存）
        path = self.thick_var.get().strip() or self.front_var.get().strip()
        if not path:
            return
        try:
            _, _, Z = load_any(path)
            if Z is None:
                return
            self.ax.clear()
            self.ax.set_axis_off()
            Z = Z.astype(float)
            v = Z[np.isfinite(Z)]
            if v.size == 0:
                return
            cmap = get_custom_cmap()
            cmap.set_bad('white')
            if self.view_mode.get() == 'heatmap':
                self.ax.imshow(Z, cmap=cmap, vmin=np.min(v), vmax=np.max(v),
                               aspect='equal', interpolation='nearest')
            else:
                ny, nx = Z.shape
                Xg, Yg = np.meshgrid(np.arange(nx), np.arange(ny))
                Zf = np.where(np.isfinite(Z), Z, np.nanmean(v))
                self.ax.contourf(Xg, Yg, Zf, levels=20, cmap=cmap)
            self.canvas.draw()
        except Exception as e:
            print("绘图失败:", e)


if __name__ == "__main__":
    app = App()
    app.mainloop()