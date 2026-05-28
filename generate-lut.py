import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

def generate_lut(csv_filename, output_filename="phase_lut.csv"):
    # 1. Load the calibration data
    # Assuming CSV format: [grayscale_value, power_measurement]
    data = np.loadtxt(csv_filename, delimiter=',')
    gray_vals = data[:, 0]
    power = data[:, 1]

    # Optional: Smooth the power data to reduce noise from the power meter
    # A simple moving average or polynomial fit can be used. 
    # Here we use a polynomial fit (degree 5 is usually good for this curve)
    poly_coeffs = np.polyfit(gray_vals, power, 5)
    power_smooth = np.polyval(poly_coeffs, gray_vals)

    # 2. Normalize the smoothed power to [0, 1]
    p_min = np.min(power_smooth)
    p_max = np.max(power_smooth)
    p_norm = (power_smooth - p_min) / (p_max - p_min)

    # Ensure values are strictly within [0, 1] to avoid math domain errors in arccos
    p_norm = np.clip(p_norm, 0.0, 1.0)

    # 3. Calculate Phase from 0 to pi
    # Formula: Phase = 2 * arccos(sqrt(P_norm))
    phase = 2 * np.arccos(np.sqrt(p_norm))

    # 4. Unwrap the phase to go from 0 to 2*pi
    # The power curve goes from Max (0 phase) -> Min (pi phase) -> Max (2pi phase)
    # We need to find the index where the power is minimum (phase = pi)
    min_idx = np.argmin(power_smooth)
    
    # For grayscale values past the minimum power, the phase is actually increasing towards 2*pi
    phase[min_idx:] = 2 * np.pi - phase[min_idx:]

    # 5. Create the Lookup Table (LUT)
    # We want a function that takes a desired Phase and returns a Grayscale value
    # We use interpolation to map evenly spaced phase values (0 to 2pi) to grayscale
    
    # Ensure phase is strictly increasing for the interpolation to work
    # We slice up to the point where phase reaches its maximum valid 2*pi range
    valid_range = np.where(np.diff(phase) > 0)[0] 
    if len(valid_range) > 0:
        end_idx = valid_range[-1] + 1
    else:
        end_idx = len(phase)
        
    phase_clean = phase[:end_idx]
    gray_clean = gray_vals[:end_idx]

    # Create an interpolation function: Phase -> Grayscale
    phase_to_gray_interp = interp1d(phase_clean, gray_clean, bounds_error=False, fill_value="extrapolate")

    # Generate 256 evenly spaced phase values from 0 to 2*pi
    desired_phases = np.linspace(0, 2 * np.pi, 256)
    lut_gray_vals = phase_to_gray_interp(desired_phases)
    
    # Clip to valid 8-bit grayscale range and convert to integers
    lut_gray_vals = np.clip(np.round(lut_gray_vals), 0, 255).astype(np.uint8)

    # 6. Save the LUT
    # Save as a simple text file or CSV. 
    # Index is the desired phase (scaled 0-255), Value is the Grayscale to send to the SLM
    np.savetxt(output_filename, lut_gray_vals, fmt='%d', header='Grayscale_Value', comments='')
    print(f"LUT successfully saved to {output_filename}")

    # 7. Plotting for verification
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(gray_vals, power, '.', label='Raw Data')
    plt.plot(gray_vals, power_smooth, '-', label='Smoothed')
    plt.xlabel('Grayscale (0-255)')
    plt.ylabel('Measured Power')
    plt.title('Power vs Grayscale')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(gray_vals, phase)
    plt.xlabel('Grayscale (0-255)')
    plt.ylabel('Calculated Phase (radians)')
    plt.title('Phase vs Grayscale')

    plt.subplot(1, 3, 3)
    plt.plot(desired_phases, lut_gray_vals)
    plt.xlabel('Desired Phase (0 to 2pi)')
    plt.ylabel('Required Grayscale')
    plt.title('Final LUT')

    plt.tight_layout()
    plt.show()

    return lut_gray_vals

if __name__ == '__main__':
    # Replace 'calib_checkerboard.csv' with your actual filename
    lut = generate_lut('calib_checkerboard_period4.csv', 'slm_532_lut.csv')
    sys.exit(0)