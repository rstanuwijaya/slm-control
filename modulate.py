import jax
import jax.numpy as jnp
import numpy as np
import os

@jax.jit
def amplitude_to_slm_depth(target_ampl):
    """
    Converts a target amplitude matrix (-1 to 1) to SLM modulation depth.
    The magnitude determines the depth (0 to 1), and the sign is preserved 
    to represent a 0 or pi phase shift.
    """
    # 1. Separate magnitude and sign
    A_mag = jnp.abs(target_ampl)
    A_sign = jnp.sign(target_ampl) # Returns -1, 0, or 1
    
    # Clip magnitude to strictly [0, 1] to avoid math errors
    A = jnp.clip(A_mag, 0.0, 1.0)
    
    # 2. Initial guess for v using the Taylor expansion of sinc(v) ≈ 1 - v^2/6
    v = jnp.sqrt(6.0 * (1.0 - A))
    
    # 3. Define a single Newton-Raphson update step
    def newton_step(i, v_current):
        # Prevent division by zero when A is exactly 1 (which means v should be 0)
        v_safe = jnp.where(v_current < 1e-5, 1e-5, v_current)
        
        # f(v) = sin(v)/v - A = 0
        f = jnp.sin(v_safe) / v_safe - A
        
        # Derivative f'(v) = (v*cos(v) - sin(v)) / v^2
        df = (v_safe * jnp.cos(v_safe) - jnp.sin(v_safe)) / (v_safe**2)
        
        # Update rule: v_new = v - f(v)/f'(v)
        v_new = v_safe - f / df
        
        # Force exactly 0 if target amplitude is 1.0 to avoid floating point noise
        return jnp.where(A >= 0.99999, 0.0, v_new)
        
    # 4. Run the Newton-Raphson solver for 10 iterations
    v_sol = jax.lax.fori_loop(0, 10, newton_step, v)
    
    # 5. Convert v back to phase depth phi0 (in radians)
    # v = (2*pi - phi0) / 2  =>  phi0 = 2*pi - 2*v
    phi0 = 2 * jnp.pi - 2 * v_sol
    
    # 6. Normalize phi0 to [0, 1] for the SLM depth magnitude
    slm_image_mag = phi0 / (2 * jnp.pi)
    
    # 7. Reapply the sign (negative values will map to negative depths)
    # Note: If target_ampl was 0, A_sign is 0, making the final output exactly 0.
    slm_image = slm_image_mag * jnp.where(A_sign == 0, 1.0, A_sign)
    
    return slm_image

def main():
    INPUT_FILES = ['DCT1.csv', 'DCT2.csv', 'DCT3.csv', 'DCT4.csv', 'DCT5.csv', 'DCT6.csv', 'DCT7.csv', 'DCT8.csv', 'image.csv']
    OUTPUT_FILES = ['mDCT1.csv', 'mDCT2.csv', 'mDCT3.csv', 'mDCT4.csv', 'mDCT5.csv', 'mDCT6.csv', 'mDCT7.csv', 'mDCT8.csv', 'mimage.csv']

    for input_file, output_file in zip(INPUT_FILES, OUTPUT_FILES):
        if not os.path.exists(input_file):
            print(f"'{input_file}' not found. Generating a random 8x8 matrix for testing...")
            dummy_data = np.random.uniform(-1.0, 1.0, (8, 8))
            np.savetxt(input_file, dummy_data, delimiter=',')

        print(f"Processing '{input_file}' -> '{output_file}'...")
        target_ampl = np.loadtxt(input_file, delimiter=',')

        if target_ampl.shape != (8, 8):
            print(f"Warning: Expected an 8x8 matrix, but got {target_ampl.shape}")

        target_ampl_jax = jnp.array(target_ampl)
        slm_depths_jax = amplitude_to_slm_depth(target_ampl_jax)

        slm_depths = np.array(slm_depths_jax)
        np.savetxt(output_file, slm_depths, delimiter=',', fmt='%.6f')
        print(f"Saved SLM modulation depths to '{output_file}'.")

if __name__ == "__main__":
    main()