import numpy as np
from numpy.fft import fft2, ifftshift, ifft2, fftshift

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    kernel = np.zeros((size, size))
    center = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            r = (i - center)**2 + (j - center)**2
            kernel[i, j] = np.exp(-r / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return kernel / kernel.sum()

def pad_kernel(kernel, target):
    th, tw = target
    kh, kw = kernel.shape[:2]
    ph, pw = th - kh, tw - kw

    padding = [((ph+1) // 2, ph // 2), ((pw+1) // 2, pw // 2)]
    kernel = np.pad(kernel, padding)
    return kernel

def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    kernel_padded = pad_kernel(h, shape)
    return fft2(ifftshift(kernel_padded))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros(H.shape, dtype="complex")
    H_inv[abs(H) <= threshold] = np.conj(0)
    H_inv[abs(H) > threshold] = 1 / H[abs(H) > threshold]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)
    F = G*H_inv
    return abs(fftshift(ifft2(F)))



def wiener_filtering(blurred_img, h, K=0):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conj(H)
    F = H_conj / (abs(H_conj)**2 + K) * G
    image = abs(fftshift(ifft2(F)))
    return image


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    max_i = 255
    rmse = np.sqrt(((img1 - img2)**2).mean())
    return 20 * np.log10(max_i / rmse)
