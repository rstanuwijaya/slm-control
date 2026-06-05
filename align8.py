import sys
import os
import json
import signal
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt
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
        # so you can control the pattern while the preview window is focused
        self.slm_driver.keyPressEvent(event)


class SLMDriver(QWidget):
    def __init__(self, screen_index=1, width=400, height=400, config_file="align.json"):
        super().__init__()
        self.width = width
        self.height = height
        self.screen_index = screen_index
        self.config_file = config_file
        
        # Default parameters
        self.angle = 0.0              # Grating angle in degrees
        self.aperture_x = width // 2  # Aperture center X
        self.aperture_y = height // 2 # Aperture center Y
        self.aperture_radius = 150    # Aperture radius in pixels
        self.grating_period = 30.0    # Grating period in pixels
        self.grating_depth = 255.0    # Grating depth (amplitude/phase max)
        
        # Checkerboard state
        self.checkerboard_mode = False
        self.checkerboard_size = 32   # Size of checkerboard squares in pixels
        
        self.load_config()
        self.initUI()
        
        # Initialize and show the preview window
        self.preview_window = PreviewWindow(self, width=self.width, height=self.height)
        self.preview_window.show()
        
        self.update_pattern()

    def initUI(self):
        # Set window flags to remove borders/title bar and keep it on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(self.width, self.height)

        # Create a label that will hold our phase/amplitude images
        self.label = QLabel(self)
        self.label.resize(self.width, self.height)

        # Position the SLM window on the specific screen
        screens = QApplication.screens()
        if self.screen_index < len(screens):
            target_screen = screens[self.screen_index]
            rect = target_screen.geometry()
            self.move(rect.left(), rect.top())
            print(f"SLM Window moved to Screen {self.screen_index}: {rect.width()}x{rect.height()} at ({rect.left()}, {rect.top()})")
        else:
            print(f"Warning: Screen index {self.screen_index} not found. Using default screen.")
            
        print("\n--- SLM Controls ---")
        print("[,]       : Decrease/Increase Grating Period")
        print("-, =      : Decrease/Increase Grating Depth")
        print("<, >      : Rotate Grating Angle")
        print("Arrows    : Move Aperture X/Y")
        print("C         : Toggle Checkerboard Mode")
        print("PgUp/PgDn : Enlarge/Reduce Aperture Radius (or Checkerboard Size if C is active)")
        print("Esc       : Close Window\n")

    def load_config(self):
        """Loads the configuration from config.txt if it exists."""
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
                    
                    # Sanity check for critical parameters loaded from file
                    if self.grating_period <= 0:
                        self.grating_period = 2.0
                        
                print("Configuration loaded successfully.")
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self):
        """Saves the current configuration to config.txt."""
        config = {
            "angle": self.angle,
            "aperture_x": self.aperture_x,
            "aperture_y": self.aperture_y,
            "aperture_radius": self.aperture_radius,
            "grating_period": self.grating_period,
            "grating_depth": self.grating_depth,
            "checkerboard_size": self.checkerboard_size
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def update_pattern(self):
        """Calculates the new grating/checkerboard and aperture mask, then displays it."""
        # 1. Generate the rotated grating
        x = np.arange(self.width) - self.width // 2
        y = np.arange(self.height) - self.height // 2
        X, Y = np.meshgrid(x, y)
        
        # Robust mathematical generation
        if self.grating_period <= 0:
            self.grating_period = 2.0 # Fallback to prevent division by zero
            
        theta = np.radians(self.angle)
        cos_theta = np.round(np.cos(theta), 10)
        sin_theta = np.round(np.sin(theta), 10)
        
        phase = (X * cos_theta + Y * sin_theta) / self.grating_period
        fractional_phase = np.mod(phase, 1.0)
        
        grating_float = fractional_phase * self.grating_depth
        grating_wrapped = np.mod(grating_float, 256)
        
        # Safely cast to uint8
        grating = np.floor(grating_wrapped).astype(np.uint8)

        # 2. Create absolute coordinate grids for the aperture mask and checkerboard
        X_abs, Y_abs = np.meshgrid(np.arange(self.width), np.arange(self.height))
        aperture_mask = (X_abs - self.aperture_x)**2 + (Y_abs - self.aperture_y)**2 <= self.aperture_radius**2

        if self.checkerboard_mode:
            # Generate a checkerboard mask safely
            check_size = max(1, self.checkerboard_size) # Prevent division by zero
            check_x = (X_abs // check_size) % 2
            check_y = (Y_abs // check_size) % 2
            checkerboard_mask = np.logical_xor(check_x, check_y)
            
            # Alternate between grating and uniform pattern (0)
            base_pattern = np.where(checkerboard_mask, grating, 0).astype(np.uint8)
        else:
            base_pattern = grating

        # 3. Apply the aperture mask (0 outside aperture, pattern inside)
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        pattern[aperture_mask] = base_pattern[aperture_mask]

        self.display_pattern(pattern)

    def display_pattern(self, pattern_array):
        """Updates both the SLM window and the Preview window."""
        pattern_array = np.ascontiguousarray(pattern_array, dtype=np.uint8)
        h, w = pattern_array.shape
        bytes_per_line = w
        
        # Create QImage and QPixmap
        qImg = QImage(pattern_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        
        # Update SLM Window
        self.label.setPixmap(pixmap)
        
        # Update Preview Window
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        """Handles key presses for tuning the SLM pattern."""
        step_pos = 10       # Pixels to move per arrow key press
        step_angle = 5.0    # Degrees to rotate per comma/dot press
        step_radius = 5     # Pixels to resize aperture per PageUp/PageDown
        step_period = 1.0   # Pixels to change period
        step_depth = 5.0    # Grayscale values to change depth

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
            if self.checkerboard_mode:
                self.checkerboard_size += step_radius
            else:
                self.aperture_radius += step_radius
        elif key == Qt.Key_PageDown:
            if self.checkerboard_mode:
                self.checkerboard_size = max(1, self.checkerboard_size - step_radius)
            else:
                self.aperture_radius = max(0, self.aperture_radius - step_radius)
        elif key == Qt.Key_BracketLeft:
            self.grating_period = max(2.0, self.grating_period - step_period)
        elif key == Qt.Key_BracketRight:
            self.grating_period += step_period
        elif key == Qt.Key_Minus:
            self.grating_depth = max(0.0, self.grating_depth - step_depth)
        elif key == Qt.Key_Equal: 
            self.grating_depth += step_depth
        else:
            return # Ignore other keys

        # If a valid key was pressed, save the new state and update the screens
        self.save_config()
        self.update_pattern()

    def closeEvent(self, event):
        """Ensure the preview window closes when the main SLM window closes."""
        if hasattr(self, 'preview_window'):
            self.preview_window.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Enable Ctrl+C in the terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    TARGET_SCREEN = 1 
    SLM_WIDTH = 512  
    SLM_HEIGHT = 512  
    CONFIG_FILE = "align8.json"
    
    slm_window = SLMDriver(screen_index=TARGET_SCREEN, width=SLM_WIDTH, height=SLM_HEIGHT, config_file=CONFIG_FILE)
    slm_window.show()
    
    sys.exit(app.exec_())