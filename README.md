Guide for aligning experimental setup from scratch.

There are few steps we need for characterization experiment:
1. SLM phase calibration, to know the phase response of the SLM. We only need to do it once, except when we change the SLM or the laptop 
2. Metasurface alignment, which we need to redo every time we realign the setup, or change camera setting
3. Image compression experiment, in which we need to run optimized scan of each 8x8 pixel block from a large image

For the current stage, I have the code to work on (1) and partially (2). So, we need to aim to get consistent result between the simulation and experimental result for the alignment before we can proceed to (3)

# Step 1: SLM Calibration

The goal for the SLM Calibration is to obtain an "LUT" (Lookup Table) file that maps the greyscale level to the desired SLM phase. In general, this relation is nonlinear and we need to figure out the relation. The main files we use will be `calibrate-slm-binary.py` and `generate-lut.ipynb`

## Experimental Setup (SLM Calibration)
1. Align SLM, 75cm lens, and iris; such that the 0 order (undiffracted) beam from the SLM passes through the iris.
2. Put power meter probe to measure the power of the undiffracted beam, and connect to laptop to measure the intensity.

## Running the experiment
1. Run `python align8.py` to roughly check if the SLM is working. It should show multiple focal spots when we increase the grating depth.
1. Check the config within the `calibrate-slm-binary.py`. Look for the following setting in the file:
```
    PERIOD = 64 # Change to different period as needed, from 16, 32, 64, and 128 
    MODE = 'checkerboard'     
    STEP_SIZE = 256 * 4  
    DELAY = 0.2
```
1. Run `python calibrate-slm-binary.py`. It will scan through the grating with different phase level. One of the phase is fixed at 0 and the other one will be varied. 
1. Once the scan is finished, it will generate a plot. Verify if the plot starts from maximum, reach minimum, and goes back up to maximum again. 
1. Interpreting the result:
    - The 2nd maximum will be lower from the initial maximum due to Fringing Fields (Phase Adjacent Crosstalk). 
    - If the period is too small, the Fringing Fields becomes too strong and the 2nd maximum will not show up at all.
    - If the period is too large, the diffraction angle diminishes and more intensity gets passed through the iris, thus causing the dip to become more shallow. 

## Genererating LUT
1. The previous code will also generate a gile in the `/slmcali` folder. Check for the new files, and import it to `generate-lut.ipynb` to visualize.
1. Context: The undiffracted intensity is related to the phase by the following relation $$I(\phi) = I_{min} + (I_{max} - I_{min})\cos^2(\phi/2) $$
1. To find the phase from intensity, we invert this: $$\phi = 2 \arccos\left(\sqrt{\frac{I - I_{min}}{I_{max} - I_{min}}}\right)$$
1. Piecewise Curve Fitting: To create a smooth, noise-free LUT, it fits the extrapolated data to two mathematical models:
    1.   **Sigmoid Fit ($0$ to $\pi$):** Liquid crystals often have a "dead zone" or a slow start, which a Sigmoid function models very well.
    2.  **Constrained Polynomial ($\pi$ to $2\pi$):** A quadratic/linear fit handles the higher gray levels.
1. We then fit: $$I(g) \propto \cos^2\left(\frac{\text{PiecewisePhase}(g)}{2}\right)$$ with continuous phase condition at the piecewise boundary. 
1. Notice the phase response varies across the different pixel sizes. We generally want to use the larger pixel size as the Fringing Fields effect is minimized. We extrapolate the phase response to the infinite period to obtain the LUT. 
1. The notebook should ouput a LUT table and plots it. The higher voltage is largely linear, which is the regime we want to use. Set the voltage where the response starts to get linear as the `grating_depth_min`, and set the voltage where it modulates by $2\pi$ after extrapolation to `gratinng_depth_max` in the `slm.json` file. 

# Step 2: Metasurface Alignment
In this step, we focus on the alignment to make sure the SLM and Metasurace plane overlaps, and we can observe the focal points on the camera. We need consistency between the simulation result and the alignment result before we can proceed into Step3.

## Experimental Setup (Dynamic MS Alignment)
1. Power on SLM and Camera. Run Thorcam to open the camera interface, and VSCode at the metaJPEG folder to access the scripts.
2. Run `python slm.py` and open the `slm.json` to see the current config.
3. Set in the slm.json to be angle 30 deg, min_phase_depth to 100 and max_phase_depth to 180, grating period 6 pixels.
4. Press U to show uniform grating. Pass the 1st order through iris.
5. Make sure light hits camera. Press C to enable checkerboard, align the objective to focus the checkerboard pattern.
6. Insert analyzer, set s.t. no light pass through. Insert QWP before analyzer, set s.t. no light pass, rotate QWP by 45 degree.
7. Insert sample, refocus objective to slm checkerboard, and focus sample.
8. Disable both uniform and checkerboard mode. Align the square to match the sample.

## Running Experiment & Visualization
1. The code `measure.py` will also read from `slm.json` config to use the aligned config and show different SLM patterns for the 8 DCT basis. 
2. The code will also capture the image from the camera and store it in the `images/{current_time}` folder.
3. After the experiment is done, use `visualize.ipynb` to show the captured images. (the file was corrupted when I copied it over, so I need to redo the visualization part. I need to wait until I can relog to my UStA account to access the data)

# Step 3: Image Compression
To Be Done