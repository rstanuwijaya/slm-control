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
        self.setWindowTitle("SLM Pattern Preview")
        self.setFixedSize(width, height)
        
        self.label = QLabel(self)
        self.label.resize(width, height)

    def keyPressEvent(self, event):
        # Forward any key presses to the main SLM driver
        self.slm_driver.keyPressEvent(event)


class SLMDriver(QWidget):
    def __init__(self, screen_index=1, width=400, height=400, config_file="align16.json"):
        super().__init__()
        self.width = width
        self.height = height
        self.screen_index = screen_index
        self.config_file = os.path.abspath(config_file)
        
        # Default parameters
        self.angle = 0.0
        self.aperture_x = width // 2
        self.aperture_y = height // 2
        self.aperture_radius = 150
        self.grating_period = 30.0
        self.grating_depth = 255.0
        self.checkerboard_mode = False
        self.checkerboard_size = 32
        
        # Initialize UI and Preview
        self.initUI()
        self.preview_window = PreviewWindow(self, width=self.width, height=self.height)
        self.preview_window.show()

        # Load initial config
        self.load_config()

        # --- Setup File Watcher ---
        self.file_watcher = QFileSystemWatcher(self)
        # Ensure the file exists before watching
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
            target_screen = screens[self.screen_index]
            rect = target_screen.geometry()
            self.move(rect.left(), rect.top())
        
        print(f"Watching config file: {self.config_file}")
        print("External edits to the JSON will update the SLM live.")

    def on_file_changed(self, path):
        """Triggered when the JSON file is modified externally."""
        # print(f"Config file changed: {path}. Reloading...")
        # Small delay or check to ensure file is finished writing can be added if needed
        self.load_config()
        self.update_pattern()
        
        # Some editors delete and recreate files on save, which breaks the watcher.
        # Re-adding the path ensures we keep watching.
        if self.config_file not in self.file_watcher.files():
            self.file_watcher.addPath(self.config_file)

    def load_config(self):
        """Loads the configuration from JSON."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.angle = config.get("angle", self.angle)
                    self.aperture_x = config.get("aperture_x", self.aperture_x)
                    self.aperture_y = config.get("aperture_y", self.aperture_y)
                    self.aperture_radius = config.get("aperture_radius", self.aperture_radius)
                    self.grating_period = config.get("grating_period", self.grating_period)
                    self.grating_depth = config.get("grating_depth", self.grating_depth)
                    self.checkerboard_size = config.get("checkerboard_size", self.checkerboard_size)
                    self.checkerboard_mode = config.get("checkerboard_mode", self.checkerboard_mode)
                    
                if self.grating_period <= 0: self.grating_period = 2.0
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self):
        """Saves current configuration. Temporarily disables watcher to avoid feedback loops."""
        config = {
            "angle": self.angle,
            "aperture_x": self.aperture_x,
            "aperture_y": self.aperture_y,
            "aperture_radius": self.aperture_radius,
            "grating_period": self.grating_period,
            "grating_depth": self.grating_depth,
            "checkerboard_size": self.checkerboard_size,
            "checkerboard_mode": self.checkerboard_mode
        }
        try:
            # Block signals so the 'fileChanged' event doesn't fire when we save internally
            if hasattr(self, 'file_watcher'):
                self.file_watcher.blockSignals(True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            if hasattr(self, 'file_watcher'):
                self.file_watcher.blockSignals(False)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def update_pattern(self):
        """Calculates the new grating/checkerboard and aperture mask."""
        x = np.arange(self.width) - self.width // 2
        y = np.arange(self.height) - self.height // 2
        X, Y = np.meshgrid(x, y)
        
        if self.grating_period <= 0:
            self.grating_period = 2.0
            
        theta = np.radians(self.angle)
        cos_theta = np.round(np.cos(theta), 10)
        sin_theta = np.round(np.sin(theta), 10)
        
        phase = (X * cos_theta + Y * sin_theta) / self.grating_period
        fractional_phase = np.mod(phase, 1.0)
        
        grating_float = fractional_phase * self.grating_depth
        grating_wrapped = np.mod(grating_float, 65536) # 16-bit depth
        grating = np.floor(grating_wrapped).astype(np.uint16)

        X_abs, Y_abs = np.meshgrid(np.arange(self.width), np.arange(self.height))
        aperture_mask = (X_abs - self.aperture_x)**2 + (Y_abs - self.aperture_y)**2 <= self.aperture_radius**2

        if self.checkerboard_mode:
            check_size = max(1, self.checkerboard_size)
            check_x = (X_abs // check_size) % 2
            check_y = (Y_abs // check_size) % 2
            checkerboard_mask = np.logical_xor(check_x, check_y)
            base_pattern = np.where(checkerboard_mask, grating, 0).astype(np.uint16)
        else:
            base_pattern = grating

        pattern = np.zeros((self.height, self.width), dtype=np.uint16)
        pattern[aperture_mask] = base_pattern[aperture_mask]

        self.display_pattern(pattern)

    def display_pattern(self, pattern_array):
        """Updates both the SLM window and the Preview window with a 16-bit pattern."""
        pattern_array = np.ascontiguousarray(pattern_array, dtype=np.uint16)
        h, w = pattern_array.shape
        
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Pack 16-bit into Red (LSB) and Green (MSB)
        rgb_array[:, :, 0] = (pattern_array & 0xFF).astype(np.uint8)
        rgb_array[:, :, 1] = (pattern_array >> 8).astype(np.uint8)
        
        self._current_array = np.ascontiguousarray(rgb_array)
        qimg = QImage(self._current_array.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.label.setPixmap(pixmap)
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        """Handles key presses."""
        step_pos = 10
        step_angle = 5.0
        step_radius = 5
        step_period = 1.0
        step_depth = 1 # Step for 16-bit depth

        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
            return
        elif key == Qt.Key_C:
            self.checkerboard_mode = not self.checkerboard_mode
        elif key == Qt.Key_Comma:
            self.angle = (self.angle - step_angle) % 360
        elif key == Qt.Key_Period:
            self.angle = (self.angle + step_angle) % 360
        elif key == Qt.Key_Left:
            self.aperture_x -= step_pos
        elif key == Qt.Key_Right:
            self.aperture_x += step_pos
        elif key == Qt.Key_Up:
            self.aperture_y -= step_pos
        elif key == Qt.Key_Down:
            self.aperture_y += step_pos
        elif key == Qt.Key_PageUp:
            if self.checkerboard_mode: self.checkerboard_size += step_radius
            else: self.aperture_radius += step_radius
        elif key == Qt.Key_PageDown:
            if self.checkerboard_mode: self.checkerboard_size = max(1, self.checkerboard_size - step_radius)
            else: self.aperture_radius = max(0, self.aperture_radius - step_radius)
        elif key == Qt.Key_BracketLeft:
            self.grating_period = max(2.0, self.grating_period - step_period)
        elif key == Qt.Key_BracketRight:
            self.grating_period += step_period
        elif key == Qt.Key_Minus:
            self.grating_depth = max(0.0, self.grating_depth - step_depth)
        elif key == Qt.Key_Equal: 
            self.grating_depth = min(65535, self.grating_depth + step_depth)
        else:
            return

        self.save_config()
        self.update_pattern()

    def closeEvent(self, event):
        if hasattr(self, 'preview_window'):
            self.preview_window.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    TARGET_SCREEN = 1 
    SLM_WIDTH = 512  
    SLM_HEIGHT = 512  
    CONFIG_FILE = "align16.json"
    
    slm_window = SLMDriver(screen_index=TARGET_SCREEN, width=SLM_WIDTH, height=SLM_HEIGHT, config_file=CONFIG_FILE)
    slm_window.show()
    
    sys.exit(app.exec_())