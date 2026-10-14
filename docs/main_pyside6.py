# main.py
import sys
import os
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTabWidget,
    QTableWidget, QTableWidgetItem, QProgressBar, QSplitter,
    QFormLayout, QDoubleSpinBox, QGroupBox, QMessageBox
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
    center_mode: str = 'fit'


# ============================================================
# 颜色映射
# ============================================================
def get_custom_cmap():
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    return LinearSegmentedColormap.from_list('rygb', colors, N=256)


# ============================================================
# 快速 Site flatness（向量化）
# ============================================================
def site_flatness_fast(T, X, Y, mask, dx, dy, w=26.0, h=8.0):
    """基于规则网格的切片，O(站点数 × 站点面积)，避免全矩阵布尔运算。"""
    ny, nx = T.shape
    x0 = X[0, 0]
    y0 = Y[0, 0]
    jw = max(1, int(round(w / dx)))
    ih = max(1, int(round(h / dy)))

    results = {'corner': {'SBIR': [], 'SFQR': [], 'SFLR': []},
               'center': {'SBIR': [], 'SFQR': [], 'SFLR': []}}

    for placement in ('corner', 'center'):
        j_off = 0 if placement == 'corner' else jw // 2
        i_off = 0 if placement == 'corner' else ih // 2
        for i0 in range(i_off, ny - ih + 1, ih):
            for j0 in range(j_off, nx - jw + 1, jw):
                sub_T = T[i0:i0 + ih, j0:j0 + jw]
                sub_m = mask[i0:i0 + ih, j0:j0 + jw]
                if sub_m.sum() < 3:
                    continue
                v = sub_T[sub_m]
                results[placement]['SBIR'].append(np.nanmax(v) - np.nanmin(v))

                sub_X = X[i0:i0 + ih, j0:j0 + jw][sub_m]
                sub_Y = Y[i0:i0 + ih, j0:j0 + jw][sub_m]
                sub_Z = v

                # 站点内 LS
                A = np.column_stack([np.ones_like(sub_X), sub_X, sub_Y])
                c, _, _, _ = np.linalg.lstsq(A, sub_Z, rcond=None)
                r = sub_Z - A @ c
                results[placement]['SFQR'].append(np.nanmax(r) - np.nanmin(r))

                # 全局 LS 的残差（这里简化：用整个 FQA 的 LS 需要预计算）
                # 实际应传入全局 plane
                results[placement]['SFLR'].append(np.nanmax(r) - np.nanmin(r))

    out = {}
    for pl, d in results.items():
        out[pl] = {k: (float(np.max(v)) if v else np.nan,
                       float(np.mean(v)) if v else np.nan)
                   for k, v in d.items()}
    return out


# ============================================================
# Worker
# ============================================================
class Worker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, front_path, thick_path, assumptions):
        super().__init__()
        self.front_path = front_path
        self.thick_path = thick_path
        self.assumptions = assumptions
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            t0 = time.time()
            results = {}

            self.progress.emit(5, "读取 Front...")
            F = self._read_map(self.front_path) if self.front_path else None
            self.progress.emit(15, "读取 Thickness...")
            T = self._read_map(self.thick_path) if self.thick_path else None

            if T is None and F is None:
                self.failed.emit("未提供 F 或 T")
                return

            # 统一网格
            if T is not None:
                Z = T
            else:
                Z = F
            ny, nx = Z.shape
            dx = 300.0 / (nx - 1)
            dy = 300.0 / (ny - 1)
            Xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
            Ys = (np.arange(ny) - (ny - 1) / 2.0) * dy
            Xg, Yg = np.meshgrid(Xs, Ys)
            Rg = np.sqrt(Xg**2 + Yg**2)
            mask = np.isfinite(Z)
            fqa = Rg <= (150.0 - self.assumptions.ee - 0.5 * (dx + dy))

            self.progress.emit(25, "厚度与全局平坦度...")
            results['2_thickness'] = self._thickness(T, Xg, Yg, fqa)

            if F is not None and T is not None:
                self.progress.emit(40, "形状...")
                M = F - T / 2.0
                results['3_shape'] = self._shape(M, Xg, Yg, Rg, fqa)

            if T is not None:
                self.progress.emit(55, "站点平坦度...")
                results['4_site'] = site_flatness_fast(
                    T, Xg, Yg, fqa, dx, dy,
                    w=self.assumptions.site_w, h=self.assumptions.site_h)

                self.progress.emit(70, "边缘扇区...")
                results['5_edge'] = self._edge(T, Xg, Yg, Rg, fqa)

                self.progress.emit(85, "诊断...")
                results['6_diag'] = self._diag(T, Xg, Yg, Rg, fqa, dx, dy)

            results['elapsed'] = time.time() - t0
            self.progress.emit(100, "完成")
            self.finished_ok.emit(results)

        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")

    def _read_map(self, path):
        # 复用你的 parse_map_csv
        from ws_analysis import parse_map_csv
        return parse_map_csv(path)

    def _thickness(self, T, X, Y, fqa):
        v = T[fqa]
        v = v[np.isfinite(v)]
        return {
            'THKMean': float(np.nanmean(v)),
            'GBIR': float(np.nanmax(v) - np.nanmin(v)),
        }

    def _shape(self, M, X, Y, R, fqa):
        # 复用你的 metric_WarpBF 等
        return {'Warp-BF': float(np.nanmax(M[fqa]) - np.nanmin(M[fqa]))}

    def _edge(self, T, X, Y, R, fqa):
        return {'ESBIR': float(np.nanmax(T[fqa]) - np.nanmin(T[fqa]))}

    def _diag(self, T, X, Y, R, fqa, dx, dy):
        return {'clipped': 0}


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WS 晶圆分析")
        self.resize(1600, 950)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- 顶部参数区 ----
        top = QGroupBox("参数与文件")
        form = QFormLayout(top)

        self.front_edit = QLineEdit()
        self.thick_edit = QLineEdit()
        btn_f = QPushButton("选择 F...")
        btn_t = QPushButton("选择 T...")
        btn_f.clicked.connect(lambda: self._pick(self.front_edit))
        btn_t.clicked.connect(lambda: self._pick(self.thick_edit))

        hf = QHBoxLayout(); hf.addWidget(self.front_edit); hf.addWidget(btn_f)
        ht = QHBoxLayout(); ht.addWidget(self.thick_edit); ht.addWidget(btn_t)
        form.addRow("Front:", hf)
        form.addRow("Thickness:", ht)

        self.ee_spin = QDoubleSpinBox(); self.ee_spin.setRange(0, 10); self.ee_spin.setValue(3.0)
        self.site_w_spin = QDoubleSpinBox(); self.site_w_spin.setRange(1, 100); self.site_w_spin.setValue(26.0)
        self.site_h_spin = QDoubleSpinBox(); self.site_h_spin.setRange(1, 100); self.site_h_spin.setValue(8.0)
        self.sector_spin = QDoubleSpinBox(); self.sector_spin.setRange(1, 100); self.sector_spin.setValue(30.0)
        form.addRow("EE (mm):", self.ee_spin)
        form.addRow("Site W (mm):", self.site_w_spin)
        form.addRow("Site H (mm):", self.site_h_spin)
        form.addRow("Sector L (mm):", self.sector_spin)

        hbtn = QHBoxLayout()
        self.run_btn = QPushButton("运行")
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn.clicked.connect(self.cancel)
        hbtn.addWidget(self.run_btn); hbtn.addWidget(self.cancel_btn)
        form.addRow(hbtn)

        root.addWidget(top)

        # ---- 中部：左图右 Tab ----
        splitter = QSplitter(Qt.Horizontal)

        # 左：图
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        splitter.addWidget(self.canvas)

        # 右：Tab
        self.tabs = QTabWidget()
        self.tables = {}
        for name in ['2 厚度', '3 形状', '4 站点', '5 边缘', '6 诊断', '7 纳米', '8 条纹']:
            tw = QTableWidget()
            tw.setColumnCount(3)
            tw.setHorizontalHeaderLabels(['指标', 'max', 'mean'])
            self.tables[name] = tw
            self.tabs.addTab(tw, name)
        splitter.addWidget(self.tabs)
        splitter.setSizes([800, 800])
        root.addWidget(splitter, 1)

        # ---- 底部进度 ----
        self.progress = QProgressBar()
        self.status = QLabel("就绪")
        root.addWidget(self.progress)
        root.addWidget(self.status)

        self.worker = None

    def _pick(self, edit):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "CSV (*.csv)")
        if path:
            edit.setText(path)

    def run(self):
        front = self.front_edit.text().strip() or None
        thick = self.thick_edit.text().strip() or None
        if not front and not thick:
            QMessageBox.warning(self, "提示", "至少选择 F 或 T 文件")
            return

        a = Assumptions(
            ee=self.ee_spin.value(),
            sector_width=self.sector_spin.value(),
            site_w=self.site_w_spin.value(),
            site_h=self.site_h_spin.value(),
        )

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = Worker(front, thick, a)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()

    @Slot(int, str)
    def _on_progress(self, p, msg):
        self.progress.setValue(p)
        self.status.setText(msg)

    @Slot(dict)
    def _on_finished(self, results):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText(f"完成，用时 {results.get('elapsed', 0):.1f} s")
        self._fill_tables(results)
        self._draw_map(results)

    @Slot(str)
    def _on_failed(self, msg):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "错误", msg)

    def _fill_tables(self, results):
        mapping = {
            '2 厚度': results.get('2_thickness', {}),
            '3 形状': results.get('3_shape', {}),
            '4 站点': results.get('4_site', {}),
            '5 边缘': results.get('5_edge', {}),
            '6 诊断': results.get('6_diag', {}),
        }
        for tab, data in mapping.items():
            tw = self.tables[tab]
            tw.setRowCount(0)
            for k, v in data.items():
                row = tw.rowCount()
                tw.insertRow(row)
                tw.setItem(row, 0, QTableWidgetItem(str(k)))
                if isinstance(v, (int, float)):
                    tw.setItem(row, 1, QTableWidgetItem(f"{v:.6g}"))
                else:
                    tw.setItem(row, 1, QTableWidgetItem(str(v)))

    def _draw_map(self, results):
        # 这里放你的热图/等高线绘制
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "热图/等高线\n(接入 plot_thk_new 逻辑)",
                     ha='center', va='center', transform=self.ax.transAxes)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()