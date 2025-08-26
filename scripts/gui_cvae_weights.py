# network_pkg/scripts/gui_cvae_weights.py
import sys, os, numpy as np, torch
from PyQt5                                          import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg             import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure                              import Figure

from network_pkg.admp_core.model.cvae_weights       import CVAEWeights
from network_pkg.admp_core.core.basis.make_rbf      import make_rbf
from network_pkg.admp_core.core.dmp.dmp2d           import fit_weights_2d, rollout_2d


# =========== Utils ===========
def resample_norm_safe(y, T, eps=1e-6):
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    z = y[idx]
    d = np.linalg.norm(z[-1] - z[0])                    # NOrmalization with end - start distance
    if d < eps:
        diffs = np.diff(z, axis=0)
        L = np.sum(np.linalg.norm(diffs, axis=1))
        d = max(L, eps)
    return (z - z[0]) / d.astype(np.float32)


def resample_linear(y, T):
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    return y[idx].astype(np.float64)


def nrmse(y, yhat, eps=1e-9):
    rmse    = np.sqrt(np.mean(np.sum((y - yhat) ** 2, axis=1)))
    d       = np.linalg.norm(y[-1] - y[0])
    if d < 1e-3:
        diffs   = np.diff(y, axis=0)
        L       = np.sum(np.linalg.norm(diffs, axis=1))
        d       = max(L, eps)
    return rmse / d, rmse


def make_context_np(y_world):
    s, g    = y_world[0], y_world[-1]
    diff    = g - s
    D       = float(np.linalg.norm(diff))
    tau     = 1.0
    return np.array([diff[0], diff[1], D, tau], dtype=np.float32)


# ============= Canvas ===============
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig     = Figure(figsize=(6,6), tight_layout=True)
        self.ax = fig.add_subplot(111) 
        super().__init__(fig)
        self.setParent(parent)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, alpha=0.3)
        self.raw_line,  = self.ax.plot([], [], '-',  lw=2, label='drawn')
        self.dmp_line,  = self.ax.plot([], [], '-',  lw=2, label='DMP')
        self.start_sc,  = self.ax.plot([], [], 'go', ms=8, label='start')
        self.goal_sc,   = self.ax.plot([], [], 'ro', ms=8, label='goal')
        self.last       = None
        self.ax.legend(loc='upper right')
        self.reset_view()

    def reset_view(self):
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.figure.canvas.draw_idle()


    def show_raw(self, y):
        if y is None or len(y)==0:
            self.raw_line.set_data([], [])
        else:
            self.raw_line.set_data(y[:,0], y[:,1])
        self.ax.figure.canvas.draw_idle()


    def show_dmp(self, yhat):
        if yhat is None or len(yhat)==0:
            self.dmp_line.set_data([], [])  
        else:
            self.dmp_line.set_data(yhat[:,0], yhat[:,1])
        self.ax.figure.canvas.draw_idle()


    def show_markers(self, start=None, goal=None):
        self.start_sc.set_data([start[0]],[start[1]]) if start is not None else self.start_sc.set_data([],[])
        self.goal_sc.set_data([goal[0]],[goal[1]])   if goal  is not None else self.goal_sc.set_data([],[])
        self.ax.figure.canvas.draw_idle()


# ============== Main Window ================
class MainWin(QtWidgets.QMainWindow):
    def __init__(self, T_in=200, T_out=600, K_fix=128, 
                 z_dim=32, model_path="artifacts/cvae_weights_K128_z32.pth"):
        super().__init__()
        self.setWindowTitle("Adaptive DMP - CVAE Weights (Fixed K)")
        self.resize(1080, 720)

        self.T_in       = T_in
        self.T_out      = T_out
        self.dt         = 1.0 / (T_out - 1)
        self.K_fix      = K_fix

        # device / model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = CVAEWeights(T_in, K_fix, C_dim=4, z_dim=z_dim).to(self.device)
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # basis (fixed K)
        self.c, self.h = make_rbf(K_fix)

        # Widgets
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        self.canvas = PlotCanvas(self)
        hbox.addWidget(self.canvas, stretch=3)

        ctrl = QtWidgets.QWidget(self)
        hbox.addWidget(ctrl, stretch=1)
        form = QtWidgets.QFormLayout(ctrl)

        # draw controls
        self.btn_set_start  = QtWidgets.QPushButton("Set Start")
        self.btn_set_goal   = QtWidgets.QPushButton("Set Goal")
        self.btn_draw       = QtWidgets.QPushButton("Draw Curve")
        self.btn_clear      = QtWidgets.QPushButton("Clear")
        bb = QtWidgets.QHBoxLayout()
        bb.addWidget(self.btn_set_start); bb.addWidget(self.btn_set_goal)

        form.addRow(bb)
        form.addRow(self.btn_draw)
        form.addRow(self.btn_clear)

        # tau
        self.s_tau = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s_tau.setMinimum(10)
        self.s_tau.setMaximum(300); self.s_tau.setValue(100)
        self.lbl_tau = QtWidgets.QLabel("tau = 1.00")
        form.addRow(self.lbl_tau, self.s_tau)

        # inference mode
        self.chk_deterministic = QtWidgets.QCheckBox("Deterministic decode (z = mu)")
        self.chk_deterministic.setChecked(True)
        form.addRow(self.chk_deterministic)

        self.btn_resample_z = QtWidgets.QPushButton("Sample new (if stochastic)")
        form.addRow(self.btn_resample_z)

        # fixed K info
        form.addRow(QtWidgets.QLabel(f"K (fixed) = {self.K}"))

        # text info
        self.txt_info = QtWidgets.QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMinimumHeight(160)
        form.addRow("weights info", self.txt_info)

        # run / save
        self.btn_run    = QtWidgets.QPushButton("Run DMP")
        self.btn_save   = QtWidgets.QPushButton("Save NPZ (K, w, c, h, y0, g)")
        form.addRow(self.btn_run)
        form.addRow(self.btn_save)

        self.lbl_nrmse = QtWidgets.QLabel("nRMSE = -")
        form.addRow(self.lbl_nrmse)

        # state
        self.mode = None; self.start = None; self.goal = None; self.raw_points=[]; self.is_drawing=False
        self.cached_z = None    # For stochastic decode

 
        # # Visuazliation Information
        # self.lbl_fixedK = QtWidgets.QLabel(f"K (fixed) = {self.K_fix}")
        # form.addRow(self.lbl_fixedK)

        # # Fitting as Normalization coordinate and reverse to world coordinate
        # self.chk_fit_norm = QtWidgets.QCheckBox("Fit in normalized coords")
        # self.chk_fit_norm.setChecked(True)
        # form.addRow(self.chk_fit_norm)

        # self.txt_info  = QtWidgets.QTextEdit()
        # self.txt_info.setReadOnly(True)
        # self.txt_info.setMinimumHeight(160)
        # form.addRow("weights info", self.txt_info)

        # self.btn_save = QtWidgets.QPushButton("Save NPZ (K, w, c, h, y0, g)")
        # form.addRow(self.btn_save)
        # self.btn_run = QtWidgets.QPushButton("Run DMP")
        # form.addRow(self.btn_run)

        # self.lbl_nrmse = QtWidgets.QLabel("nRMSE = -")
        # form.addRow(self.lbl_nrmse)

        # # state vars
        # self.mode=None; self.start=None; self.goal=None; self.raw_points=[]; self.is_drawing=False

        # connection
        self.btn_set_start.clicked.connect(lambda: self.set_mode("start"))
        self.btn_set_goal.clicked.connect(lambda: self.set_mode("goal"))
        self.btn_draw.clicked.connect(lambda: self.set_mode("draw"))
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save.clicked.connect(self.on_save_npz)
        self.btn_resample_z.clicked.connect(self.on_resample_z)
        self.s_tau.valueChanged.connect(self.on_tau_changed)
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_mpl_press)
        self.cid_move  = self.canvas.mpl_connect('motion_notify_event', self.on_mpl_move)
        self.cid_rel   = self.canvas.mpl_connect('button_release_event', self.on_mpl_release)

    # interactions
    def set_mode(self, m):
        self.mode = m
        if m == "draw":
            self.raw_points = []
            self.canvas.show_dmp(None)
            self.canvas.show_raw(None)


    def on_tau_changed(self, v):
        self.lbl_tau.setText(f"tau = {v / 100.0:.2f}")


    def on_clear(self):
        self.mode = None; self.start=None; self.goal=None; self.raw_points=[]; self.cached_z = None
        self.canvas.show_raw(None)
        self.canvas.show_dmp(None)
        self.canvas.show_markers(None, None)
        self.lbl_nrmse.setText("nRMSE = -")
        self.txt_info.setPlainText("")


    def _axes_contains(self, event):
        return (event.xdata is not None) and (event.ydata is not None)
    
    def on_mpl_press(self, event):
        if not self._axes_contains(event): return
        if self.mode == "start":
            self.start = np.array([event.xdata, event.ydata]); self.canvas.show_markers(self.start, self.goal)
        elif self.mode == "goal":
            self.goal = np.array([event.xdata, event.ydata]);  self.canvas.show_markers(self.start, self.goal)
        elif self.mode == "draw":
            self.is_drawing = True
            self.raw_points = [np.array([event.xdata, event.ydata])]

    def on_mpl_move(self, event):
        if not self._axes_contains(event): return
        if self.mode == "draw" and self.is_drawing:
            self.raw_points.append(np.array([event.xdata, event.ydata]))
            self.canvas.show_raw(np.array(self.raw_points))

    def on_mpl_release(self, event):
        if self.mode == "draw" and self.is_drawing:
            self.is_drawing = False

    # ====== z sampling control ======
    def on_resample_z(self):
        # only used if deterministic = False
        self.cached_z = torch.randn(1, self.z_dim, device=self.device)

    # ================== Core ======================
    def _prepare_curve(self):
        if len(self.raw_points) < 4:
            return None
        y = np.array(self.raw_points, dtype=np.float64)
        if self.start is not None: y[0,:]  = self.start
        if self.goal  is not None: y[-1,:] = self.goal
        return y

    def on_run(self):
        y = self._prepare_curve()
        if y is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "First draw trajectory with ==DRAW CURVE==")
            return

        # world resample (for display / eval)
        y_world = resample_linear(y, self.T_out)
        tau = self.s_tau.value()/100.0


        try:
            # Input (Normalization, T_in)
            y_in    = resample_norm_safe(y, self.T_in).astype(np.float32)                   # (T_in, 2) float 32
            c_np    = make_context_np(y_world)                                              # (4, )
            yin_t   = torch.tensor(y_in[None], dtype=torch.float32, device=self.device)     # (1, T, 2)
            c_t     = torch.tensor(c_np[None], dtype=torch.float32, device=self.device)     # (1, 4)

            # decode weights
            if self.chk_deterministic.isChecked():
                with torch.no_grad():
                    wx_t, wy_t, _, _ = self.model(yin_t, c_t, deterministic=True)
            else:
                # stochastic decode: sample Z once (cached) or fresh
                if self.cached_z is None:
                    self.cached_z = torch.randn(1, self.z_dim, device=self.device)
                # run encoder to get mu/logvar (only for shape/potential conditioning)
                with torch.no_grad():
                    mu, logvar = self.model.enc(yin_t, c_t)
                    # replace reparam with our chached z
                    wx_t, wy_t = self.model.dec(self.cached_z, c_t)

            wx = wx_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            wy = wy_t.squeeze(0).detach().cpu().numpy().astype(np.float64)

            # rollout in normalized coords then back to world
            y_fit = resample_norm_safe(y, self.T_out).astype(np.float64)
            y0 = y_fit[0]; g = y_fit[-1]
            yhat_fit = rollout_2d(self.T_out, self.dt, self.c, self.h, wx, wy, y0, g, tau=tau)

            d = np.linalg.norm(y_world[-1] - y_world[0])
            if d < 1e-3:
                diffs = np.diff(y_world, axis=0)
                L = np.sum(np.linalg.norm(diffs, axis=1))
                d = max(L, 1e-6)
            yhat_world = yhat_fit * d + y_world[0]

            nr, rm = nrmse(y_world, yhat_world)
            y_show = y_world; yhat_show = yhat_world
            y0_show, g_show = y_world[0], y_world[-1]

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "DMP error", str(e))
            return
        
        # draw
        self.canvas.show_raw(y_show)
        self.canvas.show_dmp(yhat_show)
        self.canvas.show_markers(y_show[0], y_show[-1])
        self.lbl_nrmse.setText(f"nRMSE = {nr:.4f}")

        # info
        info=[]
        info.append(f"K = {self.K_fix}, tau = {tau:.2f}, nRMSE = {nr:.4f}")
        info.append(f"||wx||2={np.linalg.norm(wx):.3f}, ||wy||2={np.linalg.norm(wy):.3f}")
        info.append(f"wx[:8] = {np.array2string(wx[:8], precision=3, separator=', ')} ...")
        info.append(f"wy[:8] = {np.array2string(wy[:8], precision=3, separator=', ')} ...")
        info.append(f"c.shape={self.c.shape}, h.shape={self.h.shape}")
        self.txt_info.setPlainText("\n".join(info))

        # cache for save
        self.canvas.last = dict(K=int(self.K_fix), wx=wx, wy=wy, c=self.c, h=self.h,
                                y0=y0_show, g=g_show, tau=float(tau),
                                dt=float(self.dt), nrmse=float(nr), rmse=float(rm))
        

    def on_save_npz(self):
        if not self.canvas.last:
            QtWidgets.QMessageBox.information(self, "Info", "First, click RUN DMP")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save NPZ", "", "NPZ files (*.npz)")
        if not path: return
        np.savez_compressed(path, **self.canvas.last)
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved: {path}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWin(T_in=200, T_out=600, K_fix=128, z_dim=32,
                  model_path="artifacts/cvae_weights_K128_z32.pth")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()