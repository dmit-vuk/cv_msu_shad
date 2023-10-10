import os
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    
    # Your code here
    
    # Отцентруем каждую строчку матрицы
    matrix = matrix.astype(np.float64)
    mean_vals = np.mean(matrix, axis = 1)
    matrix -= mean_vals[:,None]
    # Найдем матрицу ковариации
    cov = np.cov(matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenvalues, eigenvectors = LA.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    num_eig_vec = eigenvectors.shape[1]
    # Сортируем собственные значения в порядке убывания
    val_sorted = np.argsort(eigenvalues)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eigenvectors = eigenvectors[:, val_sorted]
    # Оставляем только p собственных векторов
    eigenvectors = eigenvectors[:, :p]
    # Проекция данных на новое пространство
    matrix_proj = eigenvectors.T @ matrix
    return eigenvectors, matrix_proj, mean_vals


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        eigenvectors, matrix_proj, mean_vals = comp        
        # Your code here
        channel = eigenvectors @ matrix_proj + mean_vals[:, None]
        result_img.append(channel)
    image = np.stack(np.clip(np.array(result_img),0,255).astype('uint8'),axis=2)
    return image


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[:,:,j],p))
        compressed = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(compressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    
    # Your code here
    r, g, b = img.transpose(2, 0, 1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    c_b = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
    c_r = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
    return np.clip(np.dstack((y, c_b, c_r)), 0, 255)

def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    
    # Your code here
    y, c_b, c_r = img.transpose(2, 0, 1)
    r = y + 0.1402 * (c_r-128)
    g = y - 0.34414 * (c_b-128) - 0.71414 * (c_r-128)
    b = y + 1.77 * (c_b-128)
    return np.clip(np.dstack((r, g, b)), 0, 255)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    # Your code here
    y, c_b, c_r  = rgb2ycbcr(rgb_img).transpose(2, 0, 1)
    c_b = gaussian_filter(c_b, sigma=10)
    c_r = gaussian_filter(c_r, sigma=10)
    ycbcr_img = np.dstack((y, c_b, c_r))
    res = np.clip(ycbcr2rgb(ycbcr_img).astype('int'), 0, 255)
    plt.imshow(res)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    y, c_b, c_r  = rgb2ycbcr(rgb_img).transpose(2, 0, 1)
    y = gaussian_filter(y, sigma=10)
    ycbcr_img = np.dstack((y, c_b, c_r))
    res = np.clip(ycbcr2rgb(ycbcr_img).astype('int'), 0, 255)
    plt.imshow(res)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    
    # Your code here
    component = gaussian_filter(component, sigma=10)
    return component[::2, ::2]

def alpha(u):
    return 1 / np.sqrt(2) if u == 0 else 1

def cos(x, u):
    return np.cos((2*x + 1)*np.pi*u / 16)

def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    G = np.zeros(block.shape)
    for u in range(block.shape[0]):
        for v in range(block.shape[1]):
            summ = 0
            for x in range(block.shape[0]):
                for y in range(block.shape[1]):
                    summ += block[x, y] * cos(x, u) * cos(y, v)
            G[u, v] = summ * alpha(u) * alpha(v) / 4
    return G


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    
    # Your code here
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here
    if q < 50:
        s = 5000 / q
    elif q <= 99:
        s = 200 - 2*q
    else:
        s = 1
    quant_mat = np.trunc((50 + s*default_quantization_matrix) / 100)
    quant_mat[quant_mat==0] = 1
    return quant_mat


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    # Your code here
    zigzag_list = []
    block = np.rot90(block)
    for k in range(-block.shape[0], block.shape[1]):
        diag = np.diag(block, k=k)
        if k % 2 == 1:
            diag = diag[::-1]
        zigzag_list += list(diag)
    return zigzag_list


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here
    compressed_list = []
    cnt, zero = 0, False
    for elem in zigzag_list:
        if elem == 0:
            zero = True
            cnt += 1
        else:
            if zero:
                compressed_list += [0, cnt]
                cnt, zero = 0, False
            compressed_list.append(elem)
    if zero:
        compressed_list += [0, cnt]
    return compressed_list

def get_blocks(comp, h, w):
    blocks = []
    for i in range(h):
        for j in range(w):
            blocks.append(comp[i*8 : (i+1)*8, j*8 : (j+1)*8] - 128)
    return blocks

def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    
    # Переходим из RGB в YCbCr
    y, cb, cr = rgb2ycbcr(img).transpose(2, 0, 1)
    # Уменьшаем цветовые компоненты
    cb = downsampling(cb)
    cr = downsampling(cr)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    blocks_y = get_blocks(y, y.shape[0]//8, y.shape[1]//8)
    blocks_cb = get_blocks(cb, cb.shape[0]//8, cb.shape[1]//8)
    blocks_cr = get_blocks(cr, cr.shape[0]//8, cr.shape[1]//8)
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    blocks_y_comp, blocks_cb_comp, blocks_cr_comp = [], [], []
    for block in blocks_y:
        blocks_y_comp.append(compression(zigzag(quantization(dct(block), quantization_matrixes[0]))))
    for block in blocks_cb:
        blocks_cb_comp.append(compression(zigzag(quantization(dct(block), quantization_matrixes[1]))))
    for block in blocks_cr:
        blocks_cr_comp.append(compression(zigzag(quantization(dct(block), quantization_matrixes[1]))))
    return [blocks_y_comp, blocks_cb_comp, blocks_cr_comp]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    
    # Your code here
    uncompressed_list = []
    zero = False
    for i in range(len(compressed_list)):
        if compressed_list[i] == 0:
            zero = True
            for _ in range(compressed_list[i+1]):
                uncompressed_list.append(0)
        elif not zero:
            uncompressed_list.append(compressed_list[i])
        else:
            zero = False
    return uncompressed_list

def kth_diag_indices(matrix, k):
    rows, cols = np.diag_indices_from(matrix)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def inverse_zigzag(inputs):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    # Your code here
    block = np.zeros((8, 8))
    i = 0
    for k in range(-8, 8):
        rows, cols = kth_diag_indices(block, k=k)
        diag = inputs[i:i+len(rows)]
        i += len(rows)
        if k % 2 == 1:
            diag = diag[::-1]
        block[rows, cols] = diag
    return np.rot90(block, 3)


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    # Your code here
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    F = np.zeros((block.shape))
    for x in range(block.shape[0]):
        for y in range(block.shape[1]):
            summ = 0
            for u in range(block.shape[0]):
                for v in range(block.shape[1]):
                    summ += alpha(u)*alpha(v)*block[u, v]*cos(x, u)*cos(y, v)
            F[x, y] = np.round(summ / 4)
    return F


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    # Your code here
    res = np.zeros((component.shape[0]*2,component.shape[1]*2))
    for i in range(component.shape[0]):
        for j in range(component.shape[1]):
            res[i*2,j*2] = component[i,j]
            res[i*2+1,j*2] = component[i,j]
            res[i*2,j*2+1] = component[i,j]
            res[i*2+1,j*2+1] = component[i,j]
    return res

def get_comp(blocks, h, w):
    k = 0
    comp = np.zeros((h, w))
    for i in range(h//8):
        for j in range(w//8):  
            comp[i*8 : (i+1)*8, j*8 : (j+1)*8] = blocks[k] + 128
            k += 1
    return comp

def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    # Your code here
    block_y, block_cb, block_cr = [], [], []
    for res in result[0]:
        block_y.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(res)), quantization_matrixes[0])))
    for res in result[1]:
        block_cb.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(res)), quantization_matrixes[1])))
    for res in result[2]:
        block_cr.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(res)), quantization_matrixes[1])))
    
    y = get_comp(block_y, result_shape[0], result_shape[1])
    cb = get_comp(block_cb, result_shape[0]//2, result_shape[1]//2)
    cr = get_comp(block_cr, result_shape[0]//2, result_shape[1]//2)
    
    cb, cr = upsampling(cb), upsampling(cr)
    ybr = np.dstack((y, cb, cr))
    rgb_img = ycbcr2rgb(ybr)
    rgb_img = np.clip(np.array(rgb_img).astype('int32'),0,255).astype('uint8')
    return rgb_img


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        quantization_matrixes = [own_quantization_matrix(y_quantization_matrix,p),own_quantization_matrix(color_quantization_matrix,p)]
        compressed = jpeg_decompression(jpeg_compression(img,quantization_matrixes),img.shape,quantization_matrixes)
            
        axes[i // 3, i % 3].imshow(compressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), np.array(compressed, dtype=np.object_))
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

if __name__ == "__main__":
    # pca_visualize()
    # get_gauss_1()
    # get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()