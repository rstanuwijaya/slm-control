import sys
import numpy as np
import pyvisa
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import signal
from datetime import datetime

class SLMDriver(QWidget):
    def __init__(self, width=512, height=512, period=16, screen_index=1):
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
        else:
            print("Warning: Target screen index not found. Displaying on primary.")

        # Preview Window for the control monitor
        self.preview = QLabel("SLM Preview - Phase Ramp")
        self.preview.setFixedSize(self.w // 2, self.h // 2)
        self.preview.show()

    def update_ramp(self, depth, direction='h'):
        """
        Generates a sawtooth phase ramp.
        depth: The maximum 16-bit value at the peak of the ramp (0-65535).
        direction: 'h' for horizontal (grating lines are vertical), 
                   'v' for vertical (grating lines are horizontal).
        """
        # Create coordinate grids
        x = np.arange(self.w)
        y = np.arange(self.h)
        X, Y = np.meshgrid(x, y)

        if direction == 'h':
            # Horizontal ramp: value increases along X
            ramp = (X % self.period) / self.period
        else:
            # Vertical ramp: value increases along Y
            ramp = (Y % self.period) / self.period
        
        # Calculate the 16-bit pattern (Sawtooth)
        # Values go from 0 to depth within one period
        pattern = (ramp * depth).astype(np.uint16)
        
        # --- BIT PACKING FOR 16-BIT DVI SLM ---
        # Create an RGB image array (Height x Width x 3 channels)
        rgb_array = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Pack the 16-bit data into the Red and Green channels
        # MSB (Most Significant Byte) goes to Green
        # rgb_array[:, :, 1] = (pattern >> 8).astype(np.uint8)
        # # LSB (Least Significant Byte) goes to Red
        # rgb_array[:, :, 0] = (pattern & 0x00).astype(np.uint8)

        rgb_array[:, :, 0] = (pattern >> 8).astype(np.uint8)
        rgb_array[:, :, 1] = (pattern >> 8).astype(np.uint8)
        rgb_array[:, :, 2] = (pattern >> 8).astype(np.uint8)


        # Ensure contiguous memory for Qt
        self._current_array = np.ascontiguousarray(rgb_array)
        bytes_per_line = self.w * 3
    
        # Create QImage using RGB888 format
        qimg = QImage(self._current_array.data, self.w, self.h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        self.label.setPixmap(pix)
        self.preview.setPixmap(pix.scaled(self.w//2, self.h//2, Qt.KeepAspectRatio))
        
        # Force UI update
        QApplication.processEvents()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            sys.exit()

if __name__ == '__main__':
    # --- CONFIGURATION ---
    PERIOD = 4      # Pixels per ramp cycle
    DIRECTION = 'v'  # 'h' for horizontal, 'v' for vertical
    STEP_SIZE = 1024 # Step size for the 16-bit sweep (0 to 65535)
    DELAY = 0.2      # Delay for power meter stabilization
    # ---------------------

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    slm = SLMDriver(512, 512, PERIOD, screen_index=1)
    slm.show()

    # Initialize Power Meter
    try:
        rm = pyvisa.ResourceManager('@py')
        # Update this string with your actual VISA address
        pm = rm.open_resource('USB0::4883::32944::P3001202::0::INSTR')
        print("Power meter connected.")
    except Exception as e:
        print(f"Visa Error: Power meter not found. {e}")
        # sys.exit() # Uncomment to require power meter

    results = []
    
    print(f"Starting Ramp Sweep (Period: {PERIOD}, Direction: {DIRECTION})...")

    try:
        # Sweep the peak intensity (depth) of the ramp from 0 to 65535
        for gray in range(0, 65536, STEP_SIZE):
            slm.update_ramp(gray, direction=DIRECTION)
            time.sleep(DELAY)
            
            # Measure power (if meter is connected)
            try:
                power = float(pm.query('MEAS:POW?'))
            except:
                power = 0.0
                
            results.append([gray, power])
            print(f"Depth: {gray:5d} | Power: {power:.3e}")

        # Ensure we test the absolute maximum 16-bit value
        slm.update_ramp(65535, direction=DIRECTION)
        time.sleep(DELAY)
        try:
            power = float(pm.query('MEAS:POW?'))
            results.append([65535, power])
        except:
            pass

    except KeyboardInterrupt:
        print("Sweep interrupted by user.")

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save Data
    filename = f"slmcali/{TIMESTAMP}_ramp_calib_{DIRECTION}_p{PERIOD}.csv"
    np.savetxt(filename, results, delimiter=",", header="depth,power")
    print(f"Calibration complete. Results saved to {filename}")
    
    sys.exit(app.exec_())