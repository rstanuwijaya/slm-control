import numpy as np
import matplotlib.pyplot as plt

def check_gamma():
    img = np.zeros((512, 512), dtype=np.uint8)
    img[:256, ::2] = 255
    img[256:, :] = 128
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("If halves match brightness, Gamma = 1.0\nIf bottom is lighter, Gamma = 2.2")
    plt.axis('off')
    plt.show()

def gray_steps():
    H = 256
    img = np.zeros((H, H), dtype=np.uint8)
    step = H // 5
    img[0*step:1*step, :] = 0
    img[1*step:2*step, :] = 64
    img[2*step:3*step, :] = 128
    img[3*step:4*step, :] = 192
    img[4*step:H, :] = 255
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Gray steps: 0, 64, 128, 192, 255")
    plt.axis('off')
    plt.show()

gray_steps()
