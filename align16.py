import sys
import os
import json
import signal
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap

class PreviewWindow(QWidget):
    """A standard window to preview the SLM pattern on the main monitor."""
    def __init__(self, slm_driver, width=400, height=400):
        super().__init__()
        self.slm_driver = slm_driver
        self.setWindowTitle("SLM Preview (16-bit)")
        self.setFixedSize(width, height)
        self.label = QLabel(self)
        self.label.resize(width, height)

    def keyPressEvent(self, event):
        self.slm_driver.keyPressEvent(event)

class SLMDriver(QWidget):
    def __init__(self, screen_index=1, width=512, height=512, config_file="align16.json"):
        super().__init__()
        self.width = width
        self.height = height
        self.screen_index = screen_index
        self.config_file = os.path.abspath(config_file)
        
        # Default parameters (16-bit)
        self.angle = 0.0
        self.aperture_x = width // 2
        self.aperture_y = height // 2
        self.aperture_radius = 150
        self.grating_period = 30.0
        self.grating_depth = 65535.0  # Full 16-bit range
        self.checkerboard_mode = False
        self.checkerboard_size = 32
        
        self.load_config()
        self.initUI()
        
        self.preview_window = PreviewWindow(self, width=self.width, height=self.height)
        self.preview_window.show()

        self.file_watcher = QFileSystemWatcher(self)
        if not os.path.exists(self.config_file):
            self.save_config() 
        self.file_watcher.addPath(self.config_file)
        self.file_watcher.fileChanged.connect(self.on_file_changed)
        
        self.update_pattern()

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(self.width, self.height)
        self.label = QLabel(self)
        self.label.resize(self.width, self.height)

        screens = QApplication.screens()
        if self.screen_index < len(screens):
            rect = screens[self.screen_index].geometry()
            self.move(rect.left(), rect.top())
        
        print(f"16-bit SLM Active. Watching: {self.config_file}")
        print("Controls: Arrows (Move), [ ] (Period), - = (Depth), < > (Rotate), C (Checker), PgUp/Dn (Size)")

    def on_file_changed(self, path):
        self.load_config()
        self.update_pattern()
        if self.config_file not in self.file_watcher.files():
            self.file_watcher.addPath(self.config_file)

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    c = json.load(f)
                    self.angle = c.get("angle", self.angle)
                    self.aperture_x = c.get("aperture_x", self.aperture_x)
                    self.aperture_y = c.get("aperture_y", self.aperture_y)
                    self.aperture_radius = c.get("aperture_radius", self.aperture_radius)
                    self.grating_period = max(0.1, c.get("grating_period", self.grating_period))
                    self.grating_depth = c.get("grating_depth", self.grating_depth)
                    self.checkerboard_size = c.get("checkerboard_size", self.checkerboard_size)
                    self.checkerboard_mode = c.get("checkerboard_mode", self.checkerboard_mode)
            except Exception as e: print(f"Load error: {e}")

    def save_config(self):
        config = {
            "angle": self.angle, "aperture_x": self.aperture_x, "aperture_y": self.aperture_y,
            "aperture_radius": self.aperture_radius, "grating_period": self.grating_period,
            "grating_depth": self.grating_depth, "checkerboard_size": self.checkerboard_size,
            "checkerboard_mode": self.checkerboard_mode
        }
        if hasattr(self, 'file_watcher'): self.file_watcher.blockSignals(True)
        with open(self.config_file, 'w') as f: json.dump(config, f, indent=4)
        if hasattr(self, 'file_watcher'): self.file_watcher.blockSignals(False)

    def update_pattern(self):
        x = np.arange(self.width) - self.width // 2
        y = np.arange(self.height) - self.height // 2
        X, Y = np.meshgrid(x, y)
        
        theta = np.radians(self.angle)
        phase = (X * np.cos(theta) + Y * np.sin(theta)) / self.grating_period
        # Calculate 16-bit grating
        grating = (np.mod(phase, 1.0) * self.grating_depth).astype(np.uint16)

        X_abs, Y_abs = np.meshgrid(np.arange(self.width), np.arange(self.height))
        aperture_mask = (X_abs - self.aperture_x)**2 + (Y_abs - self.aperture_y)**2 <= self.aperture_radius**2

        if self.checkerboard_mode:
            check_size = max(1, self.checkerboard_size)
            checker_mask = ((X_abs // check_size) % 2) ^ ((Y_abs // check_size) % 2)
            base_pattern = np.where(checker_mask, grating, 0).astype(np.uint16)
        else:
            base_pattern = grating

        pattern = np.zeros((self.height, self.width), dtype=np.uint16)
        pattern[aperture_mask] = base_pattern[aperture_mask]
        self.display_pattern(pattern)

    def display_pattern(self, pattern_array):
        h, w = pattern_array.shape
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Bit-packing: Red = LSB (lower 8 bits), Green = MSB (upper 8 bits)
        rgb_array[:, :, 0] = (pattern_array & 0xFF).astype(np.uint8)
        rgb_array[:, :, 1] = (pattern_array >> 8).astype(np.uint8)
        # Blue is typically 0 for 16-bit HDMI SLMs
        
        self._current_array = np.ascontiguousarray(rgb_array)
        qimg = QImage(self._current_array.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)
        if hasattr(self, 'preview_window'): self.preview_window.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        # Note: step_depth is 256 for 16-bit to make adjustments visible
        step_pos, step_angle, step_rad, step_per, step_depth = 10, 5.0, 5, 1.0, 256.0
        key = event.key()
        if key == Qt.Key_Escape: self.close()
        elif key == Qt.Key_C: self.checkerboard_mode = not self.checkerboard_mode
        elif key == Qt.Key_Comma: self.angle = (self.angle - step_angle) % 360
        elif key == Qt.Key_Period: self.angle = (self.angle + step_angle) % 360
        elif key == Qt.Key_Left: self.aperture_x -= step_pos
        elif key == Qt.Key_Right: self.aperture_x += step_pos
        elif key == Qt.Key_Up: self.aperture_y -= step_pos
        elif key == Qt.Key_Down: self.aperture_y += step_pos
        elif key == Qt.Key_PageUp:
            if self.checkerboard_mode: self.checkerboard_size += step_rad
            else: self.aperture_radius += step_rad
        elif key == Qt.Key_PageDown:
            if self.checkerboard_mode: self.checkerboard_size = max(1, self.checkerboard_size - step_rad)
            else: self.aperture_radius = max(0, self.aperture_radius - step_rad)
        elif key == Qt.Key_BracketLeft: self.grating_period = max(0.5, self.grating_period - step_per)
        elif key == Qt.Key_BracketRight: self.grating_period += step_per
        elif key == Qt.Key_Minus: self.grating_depth = max(0, self.grating_depth - step_depth)
        elif key == Qt.Key_Equal: self.grating_depth = min(65535, self.grating_depth + step_depth)
        else: return
        self.save_config(); self.update_pattern()

    def closeEvent(self, event):
        if hasattr(self, 'preview_window'): self.preview_window.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    slm = SLMDriver(screen_index=1, width=512, height=512, config_file="align16.json")
    slm.show()
    sys.exit(app.exec_())