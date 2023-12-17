from PIL import TiffImagePlugin
from PIL import Image
import numpy as np

def preview(arr):
    min_h = min(arr.flatten())
    max_h = max(arr.flatten())
    height_diff = max_h - min_h

    arr = arr - min_h
    arr = arr / height_diff
    arr = arr * 255
    arr = arr.astype(np.uint8)
    viewable_im = Image.fromarray(arr)
    viewable_im.show()
    
def apply_shade(arr, azimuth=315, altitude=45):
    new_arr = np.zeros((arr.shape[0] - 2, arr.shape[1] - 2))
    zenith_deg = 90 - altitude
    zenith_rad = np.radians(zenith_deg)
    azimuth_math = (360.0 - azimuth + 90) % 360.0
    azimuth_rad = np.radians(azimuth_math)

    """ moving window for pixel e
        a b c
        d e f
        g h i
    """
    cellsize = 5
    z_factor = 1
    for i in range(1, len(arr) - 1):
        print(f'{i} / {len(arr)}')
        for j in range(1, len(arr[i]) - 1):
            change_x = (
                (
                    arr[i-1][j+1] + (2 * arr[i][j+1]) + arr[i+1][j+1]
                ) - (
                    arr[i-1][j-1] + (2 * arr[i][j-1]) + arr[i+1][j-1]
                )
            ) / (8 * cellsize)
            
            change_y = (
                (
                    arr[i+1][j-1] + (2 * arr[i+1][j]) + arr[i+1][j+1]
                ) - (
                    arr[i-1][j-1] + (2 * arr[i-1][j]) + arr[i-1][j+1]
                )
            ) / (8 * cellsize)
            
            slope_rad = np.arctan(z_factor * np.sqrt(change_x**2 + change_y**2)) 
            
            if change_x != 0:
                aspect_rad = np.arctan2(change_y, -change_x)
                if aspect_rad < 0:
                    aspect_rad = 2 * np.pi + aspect_rad
            else:
                if change_y > 0:
                    aspect_rad = np.pi / 2
                elif change_y < 0:
                    aspect_rad = 3 * np.pi / 2
                else:
                    aspect_rad = 0
            
            hillshade = 255.0 * (
                (
                    np.cos(zenith_rad) * np.cos(slope_rad)
                ) + (
                    np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
                )
            )
            if hillshade < 0:
                hillshade = 0
            new_arr[i-1][j-1] = hillshade
    return np.array(new_arr)

def combine_arrays(arr, shaded_arr):
    new_arr = []
    row = []
    for i in range(len(arr)):
        row.append(arr[0][i])
    new_arr.append(row)
    for i in range(1, len(arr) - 1):
        new_row = []
        new_row.append(arr[i][0])
        for j in range(1, len(arr[i]) - 1):
            new_row.append(arr[i][j] + shaded_arr[i-1][j-1])
        new_row.append(arr[i][-1])
        new_arr.append(new_row)
    row = []
    for i in range(len(arr)):
        row.append(arr[-1][i])
    new_arr.append(row)
    return np.array(new_arr)

im = Image.open("N48E096_FABDEM_V1-0.tif")
arr = np.array(im)

preview(arr)

shaded_arr = apply_shade(arr)

combined_arr = combine_arrays(arr, shaded_arr)

preview(shaded_arr)

preview(combined_arr)