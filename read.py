import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageEnhance

def draw(image):
    # Visualisation of RGB images
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    plt.show()

# Adjusting the brightness
def adjust_brightness(image, target_brightness):
    # Calculate the average brightness of the current image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    print(current_brightness)
    # Calculating the Brightness Adjustment Factor
    brightness_factor = target_brightness / current_brightness
    # Adjusting the brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image

# Enhancement of image contrast
def enhance(img, clip_limit=1.5, tile_grid_size=(10, 10)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Creating CLAHE Objects
    clahe = cv2.createCLAHE(clip_limit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)

    # Merge channel
    lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Adjust saturation
def adjust_saturation(img, factor):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    # Multiply the value of the S-channel by a factor and ensure that the value is in the range 0-255
    s_adjusted = np.clip(s * factor, 0, 255).astype(np.uint8)
    # Merge the adjusted S-channel with other channels
    img_hsv_adjusted = cv2.merge((h, s_adjusted, v))
    # Converting HSV images back to BGR colour space
    img_bgr_output = cv2.cvtColor(img_hsv_adjusted, cv2.COLOR_HSV2BGR)
    return img_bgr_output

def shuchu(tif_file):
    # Opening a TIFF file
    with rasterio.open(tif_file) as src:
        # Read all bands (assuming band order is B02, B03, B04, B08, B12)
        bands = src.read()  # Shape is (number of bands, height, width), here (5, height, width)
        # profile = src.profile  # Getting Metadata

    # Assign bands (assuming the order of bands in TIFF is B02, B03, B04, B08, B12)
    blue = bands[0].astype(float)  # B02 - Blue
    green = bands[1].astype(float)  # B03 - Green
    red = bands[2].astype(float)  # B04 - Red
    nir = bands[3].astype(float)  # B08 - near infrared
    swir = bands[4].astype(float)  # B12 - short-wave infrared

    # True colour regularisation
    rgb_origin = np.dstack((red, green, blue))
    array_min, array_max = rgb_origin.min(), rgb_origin.max()
    rgb_normalized = ((rgb_origin - array_min) / (array_max - array_min)) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)

    # Adjusting the brightness
    img_adjusted = adjust_brightness(rgb_normalized, target_brightness=100)
    # Output
    draw(img_adjusted)

    # False colour
    flass_color = np.dstack((nir, red, green))
    array_flass_min, array_flass_max = flass_color.min(), flass_color.max()
    flass_color_normalized = ((flass_color - array_flass_min) / (array_flass_max - array_flass_min)) * 255
    flass_color_normalized = flass_color_normalized.astype(np.uint8)

    # Adjusting the brightness
    false_color_adjusted = adjust_brightness(flass_color_normalized, target_brightness=100)
    draw(false_color_adjusted)

    return img_adjusted, false_color_adjusted

if __name__ == '__main__':

    nofire_path = "D:\\Desktop\\2019_1101_nofire_B2348_B12_10m_roi.tif"
    fire_path = "D:\\Desktop\\2020_0427_fire_B2348_B12_10m_roi.tif"

    nofire_rgb = shuchu(nofire_path)
    # fire_rgb = shuchu(fire_path)

    # cv2.imwrite("output.jpg", nofire_img_adjusted)
