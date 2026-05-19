import sys
import os
import json
import signal
import numpy as np
import jax
import jax.numpy as jnp
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap

# ---------------------------------------------------------
# JAX JIT-Compiled Pattern Generator
# ---------------------------------------------------------
@jax.jit(static_argnames=['width', 'height'])
def compute_pattern(width, height, angle, aperture_x, aperture_y, aperture_radius,
                    grating_period, grating_depth, array_pitch, checkerboard_mode, 
                    checkerboard_size, zoom_level, inner_square_size, outer_square_size,
                    multipliers, outer_multiplier):
    """
    Pure function to compute the SLM pattern using JAX.
    JIT compilation makes this extremely fast after the first execution.
    """
    # Apply zoom level to geometric parameters
    effective_pitch = array_pitch * zoom_level
    effective_radius = aperture_radius * zoom_level
    effective_inner = inner_square_size * zoom_level
    effective_outer = outer_square_size * zoom_level

    # 1. Generate the rotated grating base phase
    x = jnp.arange(width) - width // 2
    y = jnp.arange(height) - height // 2
    X, Y = jnp.meshgrid(x, y)
    
    theta = jnp.radians(angle)
    cos_theta = jnp.round(jnp.cos(theta), 10)
    sin_theta = jnp.round(jnp.sin(theta), 10)
    
    phase = (X * cos_theta + Y * sin_theta) / grating_period
    fractional_phase = jnp.mod(phase, 1.0)

    # 2. Create absolute coordinate grids
    X_abs, Y_abs = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    
    # Vectorized 8x8 disk generation
    offsets = (jnp.arange(8) - 3.5) * effective_pitch
    ox, oy = jnp.meshgrid(offsets, offsets)
    cx = aperture_x + ox.flatten()
    cy = aperture_y + oy.flatten()
    
    # Broadcast to calculate distances from all pixels to all 64 centers simultaneously
    dist_sq = (X_abs[..., None] - cx)**2 + (Y_abs[..., None] - cy)**2
    
    # Create masks for each of the 64 disks
    disk_masks = dist_sq <= effective_radius**2
    
    # Map the premultipliers to their respective disks
    # disk_masks is (H, W, 64), multipliers is (64,)
    disk_multiplier_map = jnp.sum(disk_masks * multipliers, axis=-1)
    
    # Generate outer and inner square rectangle masks
    dx = jnp.abs(X_abs - aperture_x)
    dy = jnp.abs(Y_abs - aperture_y)
    
    rect_mask_outer = (dx <= effective_outer) & (dy <= effective_outer)
    rect_mask_inner = (dx <= effective_inner) & (dy <= effective_inner)
    
    # Frame is the area inside the outer square but outside the inner square
    frame_mask = rect_mask_outer & ~rect_mask_inner
    frame_multiplier_map = frame_mask * outer_multiplier
    
    # Combine multipliers (assuming disks and frame don't overlap)
    total_multiplier_map = disk_multiplier_map + frame_multiplier_map
    
    # Combine masks to know where to draw the pattern at all
    disk_mask_any = jnp.any(disk_masks, axis=-1)
    aperture_mask = disk_mask_any | frame_mask

    # Modulate grating depth with the multiplier map and apply pi phase shift for negative values
    abs_multiplier = jnp.abs(total_multiplier_map)
    base_grating = abs_multiplier * fractional_phase * grating_depth
    
    grating_float = jnp.where(
        total_multiplier_map >= 0,
        base_grating,
        base_grating + (grating_depth / 2.0)  # pi phase shift
    )

    grating_wrapped = jnp.mod(grating_float, grating_depth)
    grating = jnp.floor(grating_wrapped).astype(jnp.uint8)

    # 3. Checkerboard logic
    check_size = jnp.maximum(1, checkerboard_size)
    check_x = (X_abs // check_size) % 2
    check_y = (Y_abs // check_size) % 2
    checkerboard_mask = check_x ^ check_y  # Logical XOR for checkerboard
    
    # Apply checkerboard if enabled
    base_pattern = jnp.where(
        checkerboard_mode,
        jnp.where(checkerboard_mask, grating, 0),
        grating
    ).astype(jnp.uint8)

    # 4. Apply the aperture mask (0 outside aperture, pattern inside)
    pattern = jnp.where(aperture_mask, base_pattern, 0).astype(jnp.uint8)
    
    return pattern


class PreviewWindow(QWidget):
    """A standard window to preview the SLM pattern on the main monitor."""
    def __init__(self, slm_driver, width=512, height=512):
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
    def __init__(self, screen_index=1, width=512, height=512, config_file="config.txt", image_file="image.csv"):
        super().__init__()
        self.width = width
        self.height = height
        self.screen_index = screen_index
        self.config_file = config_file
        self.image_file = image_file
        
        # Default parameters
        self.angle = 0.0              
        self.aperture_x = width // 2  
        self.aperture_y = height // 2 
        self.aperture_radius = 15     
        self.grating_period = 30.0    
        self.grating_depth = 255.0    
        self.array_pitch = 35.0       
        
        # Zoom and square frames
        self.zoom_level = 1.0
        self.inner_square_size = 140.0
        self.outer_square_size = 155.0
        
        # Checkerboard state
        self.checkerboard_mode = False
        self.checkerboard_size = 32   
        
        # New parameters for multipliers
        self.A = 1.0
        self.outer_state = 1 # 1 -> 0, 2 -> +A, 3 -> -A
        self.multipliers = np.ones(64, dtype=np.float32)
        
        self.load_csv()
        self.load_config()
        self.initUI()
        
        # Initialize and show the preview window
        self.preview_window = PreviewWindow(self, width=self.width, height=self.height)
        self.preview_window.show()
        
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.addPath(os.path.abspath(self.image_file))
        self.file_watcher.fileChanged.connect(self.on_image_csv_changed)
        
        self.update_pattern()

    def load_csv(self):
        """Loads the 8x8 grid of multipliers from image.csv."""
        if os.path.exists(self.image_file):
            try:
                data = np.loadtxt(self.image_file, delimiter=",")
                if data.size == 64:
                    self.multipliers = data.flatten().astype(np.float32)
                    print("Successfully loaded multipliers from image.csv")
                else:
                    print(f"Warning: image.csv contains {data.size} elements, expected 64. Using default array of 1s.")
            except Exception as e:
                print(f"Failed to load image.csv: {e}")
        else:
            print("image.csv not found. Using default array of 1s for the 8x8 grid.")

    def on_image_csv_changed(self, path):
        self.load_csv()
        self.update_pattern()
        self.file_watcher.addPath(path)

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
            print(f"SLM Window moved to Screen {self.screen_index}: {rect.width()}x{rect.height()} at ({rect.left()}, {rect.top()})")
        else:
            print(f"Warning: Screen index {self.screen_index} not found. Using default screen.")
            
        print("\n--- SLM Controls ---")
        print("[,]       : Decrease/Increase Grating Period")
        print("-, =      : Decrease/Increase Grating Depth")
        print("<, >      : Rotate Grating Angle")
        print("Arrows    : Move Aperture X/Y")
        print("PgUp/PgDn : Zoom In / Zoom Out (Scales overall geometry)")
        print("W, S      : Increase/Decrease Lens Size (Aperture Radius)")
        print("E, D      : Increase/Decrease Inner Square Size")
        print("R, F      : Increase/Decrease Outer Square Size")
        print("C         : Toggle Checkerboard Mode")
        print("1, 2, 3   : Set outer area multiplier to 0, +A, -A respectively")
        print("Q, A      : Increase/Decrease A (bounded between 0 and 1)")
        print("Esc       : Close Window\n")

    def load_config(self):
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
                    self.array_pitch = config.get("array_pitch", self.array_pitch)
                    self.checkerboard_size = config.get("checkerboard_size", self.checkerboard_size)
                    
                    self.zoom_level = config.get("zoom_level", self.zoom_level)
                    self.inner_square_size = config.get("inner_square_size", self.inner_square_size)
                    self.outer_square_size = config.get("outer_square_size", self.outer_square_size)
                    
                    self.A = config.get("A", self.A)
                    self.outer_state = config.get("outer_state", self.outer_state)
                    
                    if self.grating_period <= 0:
                        self.grating_period = 2.0
                        
                print("Configuration loaded successfully.")
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self):
        config = {
            "angle": self.angle,
            "aperture_x": self.aperture_x,
            "aperture_y": self.aperture_y,
            "aperture_radius": self.aperture_radius,
            "grating_period": self.grating_period,
            "grating_depth": self.grating_depth,
            "array_pitch": self.array_pitch,
            "checkerboard_size": self.checkerboard_size,
            "zoom_level": self.zoom_level,
            "inner_square_size": self.inner_square_size,
            "outer_square_size": self.outer_square_size,
            "A": self.A,
            "outer_state": self.outer_state
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def update_pattern(self):
        """Calls the JIT-compiled JAX function and updates the display."""
        safe_period = max(2.0, self.grating_period)
        
        # Determine the outer multiplier based on the current state
        if self.outer_state == 1:
            current_outer_multiplier = 0.0
        elif self.outer_state == 2:
            current_outer_multiplier = self.A
        else:
            current_outer_multiplier = -self.A
            
        # Call the JAX JIT-compiled function
        jax_pattern = compute_pattern(
            width=self.width,
            height=self.height,
            angle=self.angle,
            aperture_x=self.aperture_x,
            aperture_y=self.aperture_y,
            aperture_radius=self.aperture_radius,
            grating_period=safe_period,
            grating_depth=self.grating_depth,
            array_pitch=self.array_pitch,
            checkerboard_mode=self.checkerboard_mode,
            checkerboard_size=self.checkerboard_size,
            zoom_level=self.zoom_level,
            inner_square_size=self.inner_square_size,
            outer_square_size=self.outer_square_size,
            multipliers=self.multipliers,
            outer_multiplier=current_outer_multiplier
        )
        
        # Convert JAX DeviceArray back to a standard NumPy array for PyQt5
        pattern = np.array(jax_pattern)
        
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
        step_pos = 10       
        step_angle = 5.0    
        step_period = 1.0   
        step_depth = 5.0    
        step_zoom = 0.05
        step_radius = 1.0
        step_square = 5.0
        step_A = 0.05

        key = event.key()

        if key == Qt.Key_Escape:
            self.close()
            return
        elif key == Qt.Key_C:
            self.checkerboard_mode = not self.checkerboard_mode
            
        # Multiplier Controls
        elif key == Qt.Key_1:
            self.outer_state = 1
        elif key == Qt.Key_2:
            self.outer_state = 2
        elif key == Qt.Key_3:
            self.outer_state = 3
        elif key == Qt.Key_Q:
            self.A = min(1.0, self.A + step_A)
        elif key == Qt.Key_A:
            self.A = max(0.0, self.A - step_A)            
        # Geometry Controls
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
            
        # Zoom Controls
        elif key == Qt.Key_PageUp:
            self.zoom_level += step_zoom
        elif key == Qt.Key_PageDown:
            self.zoom_level = max(0.1, self.zoom_level - step_zoom)
            
        # Lens Size Controls
        elif key == Qt.Key_W:
            self.aperture_radius += step_radius
        elif key == Qt.Key_S:
            self.aperture_radius = max(0.0, self.aperture_radius - step_radius)
            
        # Inner Square Controls
        elif key == Qt.Key_E:
            self.inner_square_size += step_square
        elif key == Qt.Key_D:
            self.inner_square_size = max(0.0, self.inner_square_size - step_square)
            
        # Outer Square Controls
        elif key == Qt.Key_R:
            self.outer_square_size += step_square
        elif key == Qt.Key_F:
            self.outer_square_size = max(0.0, self.outer_square_size - step_square)

        # Grating Controls
        elif key == Qt.Key_BracketLeft:
            self.grating_period = max(2.0, self.grating_period - step_period)
        elif key == Qt.Key_BracketRight:
            self.grating_period += step_period
        elif key == Qt.Key_Minus:
            self.grating_depth = max(0.0, self.grating_depth - step_depth)
        elif key == Qt.Key_Equal: 
            self.grating_depth += step_depth
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
    CONFIG_FILE = "config.json"
    IMAGE_FILE = "image.csv"
    
    slm_window = SLMDriver(screen_index=TARGET_SCREEN, width=SLM_WIDTH, height=SLM_HEIGHT, config_file=CONFIG_FILE, image_file=IMAGE_FILE)
    slm_window.show()
    
    sys.exit(app.exec_())