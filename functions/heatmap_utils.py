import cv2
import numpy as np
from tqdm import tqdm

def GaussianMask(sizex, sizey, sigma=33, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

def Fixpos2Densemap(fix_arr, width, height, sigma=33, alpha=0.5, threshold=10):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap
    """

    heatmap = np.zeros((height, width), np.float32)
    for n_subject in tqdm(range(fix_arr.shape[0])):
        heatmap += GaussianMask(width, height, sigma,
                                (fix_arr[n_subject, 0],
                                 fix_arr[n_subject, 1]))

    # Normalization
    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255
    heatmap = heatmap.astype("uint8")

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def make_heatmap(p, resolution=100, sigma=10, threshold=10):
    if resolution is None:
        resolution = np.nanmax(p.flatten())
    width = resolution  # resolution of the output figure
    height = resolution
    fix_arr = p * resolution
    alpha = 0.5
    heatmap = Fixpos2Densemap(fix_arr, width, height, sigma, alpha, threshold)
    return heatmap

def gridmaps(p, grid_size=(5, 5), normalize=True):
    '''
    p: np.array, (n x 2), float, fixation points, value range 0-1,
    '''
    #convert fixation points to pixels
    p = (p.reshape((-1, 2)) * (np.array(grid_size).reshape((1, 2)) - 1.)).astype(int)
    #make a pixel grid
    M = np.zeros(grid_size)
    #count pixel fixation
    for i in range(p.shape[0]):
        M[p[i, 0], p[i, 1]] += 1
    #normalize map
    if normalize:
        M = M / np.sum(M.flatten())
    return M
