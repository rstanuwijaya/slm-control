import cv2
import numpy as np
from pylablib.devices import Thorlabs
import slm
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import sys
import os
from datetime import datetime
import shutil

def capture_image(cam, filename, path):
    """
    Captures an image using the already opened camera instance,
    shifts the 12-bit data by 4 bits to fill a 16-bit range, 
    and saves it as a 16-bit PNG.
    """
    try:
        image_data = cam.snap()
        # print data type of image_data
        print(image_data.dtype)

        print(f"Image captured! Shape: {image_data.shape}, dtype: {image_data.dtype}")
        
        # Convert to uint16 (if not already) and shift left by 4 bits
        # 12-bit max is 4095. Shifted by 4 (multiplied by 16) makes it 65520.
        image_data_16bit = image_data.astype(np.uint16)
        image_data_shifted = np.left_shift(image_data_16bit, 6)
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        # Save as 16-bit PNG
        cv2.imwrite(full_path, image_data_shifted)
        print(f"Image saved as '{full_path}'")
    except Exception as e:
        print(f"An error occurred during capture: {e}")

def run_measurements():
    TARGET_SCREEN = 1
    SLM_WIDTH = 512  # Updated to match slm.json
    SLM_HEIGHT = 512
    CONFIG_FILE = "slm.json"
    
    # Generate timestamp for the image directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    IMAGE_DIR = f"./images/{timestamp}/"

    # copy slm.json to image_dir
    os.makedirs(IMAGE_DIR, exist_ok=True)
    shutil.copy2(CONFIG_FILE, IMAGE_DIR)
    shutil.copy2('measure.py', IMAGE_DIR)

    # 1. Initialize Camera Once
    try:
        cam = Thorlabs.ThorlabsTLCamera()
        print(f"Connected to camera: {cam.get_device_info().model}")
        print(f"Camera sensor: {cam.get_sensor_info()}")
        cam.set_exposure(0.02)
    except Exception as e:
        print(f"Failed to connect to the camera: {e}")
        return

    app = QApplication(sys.argv)

    # 2. Initialize SLM Window
    slm_window = slm.SLMDriver(
        screen_index=TARGET_SCREEN,
        width=SLM_WIDTH,
        height=SLM_HEIGHT,
        config_file=CONFIG_FILE,
        image_file="patterns/DCT1.csv"
    )
    slm_window.show()

    # 3. Prepare Task Queue (8 files x 3 states = 24 tasks)
    dct_files = [f"patterns/mDCT{i}.csv" for i in range(1, 9)]
    
    # Map slm.py outer_state to filename suffixes
    # 1 -> 0 (none), 2 -> +A (plus), 3 -> -A (minus)
    states = [
        (1, "none"),
        (2, "plus"),
        (3, "minus")
    ]
    
    tasks = []
    for dct in dct_files:
        for state_val, suffix in states:
            tasks.append((dct, state_val, suffix))

    current_index = [0]

    def finish_measurements():
        print("\nAll measurements complete. Cleaning up...")
        cam.close()
        print("Camera connection closed.")
        slm_window.close()
        app.quit()

    def process_next_task():
        idx = current_index[0]
        if idx >= len(tasks):
            finish_measurements()
            return

        csv_file, state_val, suffix = tasks[idx]
        print(f"\n--- Loading {csv_file} | State: {suffix} ---")
        
        # Update SLM properties
        slm_window.image_file = csv_file
        slm_window.load_csv()
        slm_window.outer_state = state_val
        slm_window.update_pattern()
        
        # Ensure file watcher is tracking the current file
        slm_window.file_watcher.addPath(os.path.abspath(csv_file))

        # Format filename: e.g., mDCT1_plus.png
        base_name = os.path.basename(csv_file).replace(".csv", "")
        png_name = f"{base_name}_{suffix}.png"

        # Wait 3 seconds for SLM liquid crystals to settle, then capture
        QTimer.singleShot(3000, lambda: perform_capture(png_name))

    def perform_capture(png_name):
        # Capture the image
        capture_image(cam, png_name, IMAGE_DIR)
        
        current_index[0] += 1
        # Wait 1 second before loading the next pattern
        QTimer.singleShot(1000, process_next_task)

    # Initial 3s delay for SLM window to render and camera to warm up, then start
    QTimer.singleShot(3000, process_next_task)

    sys.exit(app.exec_())


if __name__ == "__main__":
    run_measurements()