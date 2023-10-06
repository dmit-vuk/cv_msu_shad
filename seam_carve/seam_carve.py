from scipy.ndimage import convolve
import numpy as np

def yuv(image):
    r,g,b = image.transpose(2, 0, 1)
    res = 0.299 * r + 0.587 * g + 0.114 * b
    return res
   
def compute_energy(image):
    img = yuv(image)
    w_x = np.zeros((3,3),dtype = 'float64')
    w_y = np.zeros((3,3),dtype = 'float64')
    w_x[1,0] = -1/2
    w_x[1,2] = 1/2
    w_y[0,1] = -1/2
    w_y[2,1] = 1/2
    map_x = convolve(img, w_x, mode="nearest")
    map_y = convolve(img, w_y, mode="nearest")
    map_x[..., 0] *= 2
    map_x[..., -1] *= 2
    map_y[0] *= 2
    map_y[-1] *= 2
    return np.sqrt(map_x**2 + map_y**2)

def seam_matrix_horiz(energy, mask=None):
    h, w = energy.shape
    matrix = np.zeros((h, w), dtype='float64')
    if mask is not None:
        energy += mask.astype('float64')*h*w*256
    matrix[0] = energy[0]
    for i in range(1, h):
        for j in range(w):
            left = matrix[i-1, j-1] if j > 0 else 1e10
            right = matrix[i-1, j+1] if j < w-1 else 1e10
            matrix[i, j] = min(left, matrix[i-1, j], right) + energy[i, j]
    return matrix

def compute_seam_matrix(energy, mode, mask=None):
    if mode == 'horizontal' or mode == 'horizontal shrink':
        res = seam_matrix_horiz(energy, mask)
    elif mode == 'vertical' or mode == 'vertical shrink':
        res = seam_matrix_horiz(energy.T, mask.T).T if mask is not None else seam_matrix_horiz(energy.T).T
    else:
        raise Exception(f'Not right mode. Expected "vertical" or "horizontal", but get {mode}')
    return res

def mask_seam_horiz(seam_matrix):
    h, w = seam_matrix.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    last = np.argmin(seam_matrix[-1])
    mask[h-1, last] = 1
    for i in range(h-2, -1, -1):
        left = 0 if last == 0 else 1
        right = 1 if last == w-1 else 2
        last = last + np.argmin(seam_matrix[i, last-left:last+right]) - left
        mask[i, last] = 1
    return mask

def remove_vert(image, mask_seam, mask=None):
    h, w, c = image.shape
    new_image = np.ones((h-1, w, c))
    mask_new = np.ones((h-1, w)) if mask is not None else None
    for i in range(w):
        clm = mask_seam[..., i]
        idx = np.where(clm == 0)[0]
        new_image[:,i,:] *= image[idx, i, ...]
        if mask is not None:
            mask_new[:,i] *= mask[idx,i]
    return new_image.astype(np.uint8), mask_new

def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    if mode == 'horizontal shrink':
        mask_seam = mask_seam_horiz(seam_matrix)
        if mask is None:
            new_image, mask_new = remove_vert(image.transpose(1, 0, 2), mask_seam.T, None)
        else:
            new_image, mask_new = remove_vert(image.transpose(1, 0, 2), mask_seam.T, mask.T)
            mask_new = mask_new.T
        new_image = new_image.transpose(1, 0, 2)
    elif mode == 'vertical shrink':
        mask_seam = mask_seam_horiz(seam_matrix.T).T
        new_image, mask_new = remove_vert(image, mask_seam, mask)
    else:
        raise Exception(f'Not right mode. Expected "vertical shrink" or "horizontal shrink", but get {mode}')
    return new_image, mask_new, mask_seam

def seam_carve(image, mode, mask=None):
    seam_matrix = compute_seam_matrix(compute_energy(image), mode, mask)
    image_new, mask_new, mask_seam = remove_minimal_seam(image, seam_matrix, mode, mask)
    return image_new, mask_new, mask_seam