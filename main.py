import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMainWindow,
)


class DiffusionSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Pollutant Diffusion Simulation")

        self.layout = QVBoxLayout()

        # 創建輸入欄位
        self.entry_Lx = self.create_input_field(self.layout, "Lake Length (Lx) [m]:")
        self.entry_Ly = self.create_input_field(self.layout, "Lake Width (Ly) [m]:")
        self.entry_Nx = self.create_input_field(
            self.layout, "Grid Points in x direction (Nx) [count]:"
        )
        self.entry_Ny = self.create_input_field(
            self.layout, "Grid Points in y direction (Ny) [count]:"
        )
        self.entry_T = self.create_input_field(
            self.layout, "Total Simulation Time (T) [s]:"
        )
        self.entry_initial_x = self.create_input_field(
            self.layout, "Initial Release Point x-coordinate [m]:"
        )
        self.entry_initial_y = self.create_input_field(
            self.layout, "Initial Release Point y-coordinate [m]:"
        )
        self.entry_D = self.create_input_field(
            self.layout, "Diffusion Coefficient (D) [m²/s]:"
        )

        # 創建按鈕
        run_button = QPushButton("Run Simulation")
        run_button.clicked.connect(self.run_simulation)
        self.layout.addWidget(run_button)

        self.setLayout(self.layout)

    def create_input_field(self, layout, label_text):
        hbox = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        hbox.addWidget(label)
        hbox.addWidget(line_edit)
        layout.addLayout(hbox)
        return line_edit

    def run_simulation(self):
        Lx = float(self.entry_Lx.text())
        Ly = float(self.entry_Ly.text())
        Nx = int(self.entry_Nx.text())
        Ny = int(self.entry_Ny.text())
        T = float(self.entry_T.text())
        initial_x = float(self.entry_initial_x.text())
        initial_y = float(self.entry_initial_y.text())
        D = float(self.entry_D.text())

        dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
        dt = 0.1 * min(dx**2, dy**2) / D
        Nt = int(T / dt)

        # 初始化濃度場
        C = np.zeros((Nx, Ny))

        # 濃度初始點位置
        initial_i = int(initial_x / dx)
        initial_j = int(initial_y / dy)
        C[initial_i, initial_j] = 100.0

        # 建立輸出資料夾
        output_dir = "diffusion_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 使用數值方法（有限差分法）求解
        time_accumulated = 0.0
        for n in range(Nt + 1):
            C_new = C.copy()
            C_new[1:-1, 1:-1] = C[1:-1, 1:-1] + D * dt * (
                (C[2:, 1:-1] - 2 * C[1:-1, 1:-1] + C[:-2, 1:-1]) / dx**2
                + (C[1:-1, 2:] - 2 * C[1:-1, 1:-1] + C[1:-1, :-2]) / dy**2
            )

            # 邊界條件
            C_new[0, :] = C_new[-1, :] = 0
            C_new[:, 0] = C_new[:, -1] = 0

            C = C_new.copy()
            time_accumulated += dt
            if abs(time_accumulated - round(time_accumulated)) < dt:
                time_label = (
                    f"{int(time_accumulated):03d}"  # Format string to keep three digits
                )
                plt.imshow(C, extent=[0, Lx, 0, Ly], origin="lower", cmap="hot")
                plt.colorbar(label="Concentration")
                plt.xlabel("X (m)")
                plt.ylabel("Y (m)")
                plt.title(f"Pollutant Concentration Distribution at t={time_label}s")
                plt.savefig(
                    os.path.join(output_dir, f"concentration_t{time_label}.png")
                )
                plt.close()

        self.plot_window = PlotWindow(output_dir)
        self.plot_window.show()


class PlotWindow(QMainWindow):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0][1:]),
        )
        self.current_index = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Pollutant Concentration Distribution")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        button_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.show_previous)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.show_next)
        button_layout.addWidget(prev_button)
        button_layout.addWidget(next_button)
        self.layout.addLayout(button_layout)

        self.update_plot()

    def update_plot(self):
        img_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        img = plt.imread(img_path)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

    def show_next(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DiffusionSimulator()
    ex.show()
    sys.exit(app.exec_())
