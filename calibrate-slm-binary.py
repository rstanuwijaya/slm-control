import sys
import numpy as np
import pyvisa
import time
import signal
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import matplotlib.pyplot as plt

class SLMDriver(QWidget):
    def __init__(self, width=512, height=512, period=2, screen_index=1):
        super().__init__()
        self.w, self.h = width, height
        self.period = period
        self._current_array = None # Prevents garbage collection of the image buffer

        # Window Setup
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(self.w, self.h)
        self.label = QLabel(self)
        self.label.resize(self.w, self.h)
        
        screens = QApplication.screens()
        if screen_index < len(screens):
            rect = screens[screen_index].geometry()
            self.move(rect.left(), rect.top())

        self.preview = QLabel("SLM Preview")
        self.preview.setFixedSize(self.w // 2, self.h // 2)
        self.preview.show()

    def update_pattern(self, depth, mode='row'):
        # Create coordinate grids
        x = np.arange(self.w) // self.period
        y = np.arange(self.h) // self.period
        X, Y = np.meshgrid(x, y)

        if mode == 'column':
            mask = X % 2
        elif mode == 'row':
            mask = Y % 2
        elif mode == 'uniform':
            mask = np.ones_like(X, dtype=np.uint16)
        else: # checkerboard
            mask = (X + Y) % 2
        
        # Calculate the 16-bit pattern
        pattern = (mask * depth).astype(np.uint16)
        
        # --- BIT PACKING FOR DVI ---
        # Create an RGB image array (Height x Width x 3 channels)
        rgb_array = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Pack the 16-bit data into the Red and Green channels
        # MSB (Most Significant Byte) goes to Green
        rgb_array[:, :, 1] = (pattern >> 8).astype(np.uint8)
        
        # LSB (Least Significant Byte) goes to Red
        rgb_array[:, :, 0] = (pattern & 0xFF).astype(np.uint8)
        
        # Blue channel [:, :, 2] remains 0
        
        # Ensure contiguous memory for Qt
        self._current_array = np.ascontiguousarray(rgb_array)
        
        # 3 bytes per pixel for RGB888 format
        bytes_per_line = self.w * 3
    
        # Create QImage using standard RGB888 format
        qimg = QImage(self._current_array.data, self.w, self.h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        self.label.setPixmap(pix)
        self.preview.setPixmap(pix.scaled(self.w//2, self.h//2, Qt.KeepAspectRatio))
        QApplication.processEvents()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            sys.exit()

if __name__ == '__main__':
    PERIOD = 16  # Adjust as needed
    MODE = 'checkerboard'  # 'row', 'column', or 'checkerboard'    
    STEP_SIZE = 256 * 4 # Step size for the 16-bit sweep. 
    DELAY = 0.2  # Delay between measurements in seconds

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    slm = SLMDriver(512, 512, PERIOD, screen_index=1)
    slm.show()

    try:
        rm = pyvisa.ResourceManager('@py')
        pm = rm.open_resource('USB0::4883::32944::P3001202::0::INSTR')
    except:
        print("Visa Error: Power meter not found.")
        # sys.exit()

    results = []
    
    # Sweep through the 16-bit range
    for gray in range(0, 65536, STEP_SIZE):
        slm.update_pattern(gray, mode=MODE)
        time.sleep(DELAY)
        
        try:
            power = float(pm.query('MEAS:POW?'))
        except Exception as e:
            power = 0.0
            
        results.append([gray, power])
        print(f"Mode: {MODE} | Gray: {gray:5d} | Power: {power:.3e}")

    # Ensure we test the absolute maximum value
    if (65536 - 1) % STEP_SIZE != 0:
        gray = 65535
        slm.update_pattern(gray, mode=MODE)
        time.sleep(0.1)
        try:
            power = float(pm.query('MEAS:POW?'))
        except Exception as e:
            power = 0.0
        results.append([gray, power])
        print(f"Mode: {MODE} | Gray: {gray:5d} | Power: {power:.3e}")

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"slmcali/{TIMESTAMP}_binary_calib_{MODE}_period{PERIOD}.csv"
    print("Calibration complete. Saving results...")
    np.savetxt(filename, results, delimiter=",")
    print(f"Results saved to {filename}")

    g, p = np.loadtxt(filename, delimiter=',').T
    plt.plot(g, p)
    plt.show()
    sys.exit(0)