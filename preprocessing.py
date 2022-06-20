import numpy as np
import constants
from PIL import Image

def process_image(image: np.ndarray):
    img = Image.fromarray(image, "RGB").convert("L")
    img = img.crop((0, 0, constants.IMG_DIM_X, constants.IMG_DIM_Y)) # start_x, start_y, end_x, end_y
    img = np.array(img)
    img = img / 255. 
    img = np.array([img])
    img = img[..., np.newaxis]
    return img


