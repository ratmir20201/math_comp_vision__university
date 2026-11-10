import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
(BASE / 'output').mkdir(exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


sys.stdout = Tee(sys.stdout, open(BASE / 'output' / 'console.txt', 'w', encoding='utf-8'))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=== Шаг 1. Загрузка и вывод изображения ===')
bgr = cv2.imread(str(BASE / 'input' / 'fruits.jpg'))
bgr = cv2.resize(bgr, (500, 350))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('Размер изображения (высота, ширина, каналы):', bgr.shape)
print('Тип данных:', bgr.dtype)
plt.figure(figsize=(7, 5))
plt.imshow(rgb)
plt.title('Исходное изображение')
plt.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step1_original.png', dpi=130)
plt.close()

print('\n=== Шаг 2. Свёртка с ядрами через cv2.filter2D ===')
kernels = {
    'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    'Edge detection 1': np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32),
    'Edge detection 2': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
    'Edge detection 3': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    'Box blur 3x3': np.ones((3, 3), dtype=np.float32) / 9,
    'Gaussian blur 3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16,
}
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()
axes[0].imshow(rgb)
axes[0].set_title('Оригинал')
axes[0].axis('off')
for ax, (name, k) in zip(axes[1:], kernels.items()):
    res = cv2.filter2D(bgr, -1, k)
    diff = np.abs(res.astype(int) - bgr.astype(int)).mean()
    print(f'{name}: сумма ядра = {k.sum():.2f}, среднее изменение яркости = {diff:.2f}')
    ax.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step2_filter2d.png', dpi=120)
plt.close()

print('\n=== Шаг 3. Корреляция и свёртка ===')
k_asym = np.array([[1, 2, 0], [0, 0, 0], [0, -2, -1]], dtype=np.float32)
corr = cv2.filter2D(gray, cv2.CV_32F, k_asym)
conv = cv2.filter2D(gray, cv2.CV_32F, cv2.flip(k_asym, -1))
print('Для несимметричного ядра корреляция и свёртка различаются: средняя разница = %.2f' % np.abs(corr - conv).mean())
sym = kernels['Gaussian blur 3x3']
d_sym = np.abs(cv2.filter2D(gray, cv2.CV_32F, sym) - cv2.filter2D(gray, cv2.CV_32F, cv2.flip(sym, -1))).max()
print('Для симметричного ядра Гаусса разница равна: %.2f' % d_sym)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].imshow(np.abs(corr), cmap='gray')
ax[0].set_title('Корреляция (filter2D)')
ax[1].imshow(np.abs(conv), cmap='gray')
ax[1].set_title('Свёртка (ядро повёрнуто на 180°)')
ax[2].imshow(np.abs(corr - conv), cmap='gray')
ax[2].set_title('Разность')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step3_corr_conv.png', dpi=120)
plt.close()

print('\n=== Шаг 4. Фильтры OpenCV: blur, GaussianBlur, medianBlur, bilateralFilter ===')
rng = np.random.default_rng(5)
noisy = bgr.copy()
mask = rng.random(bgr.shape[:2])
noisy[mask < 0.02] = 0
noisy[mask > 0.98] = 255
filters = {
    'blur (среднее) 5x5': cv2.blur(noisy, (5, 5)),
    'GaussianBlur 5x5': cv2.GaussianBlur(noisy, (5, 5), 1.5),
    'medianBlur 5': cv2.medianBlur(noisy, 5),
    'bilateralFilter': cv2.bilateralFilter(noisy, 9, 75, 75),
}
def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)
print('PSNR зашумлённого изображения относительно оригинала: %.2f дБ' % psnr(noisy, bgr))
for name, img in filters.items():
    print('PSNR после %s: %.2f дБ' % (name, psnr(img, bgr)))
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()
axes[0].imshow(rgb)
axes[0].set_title('Оригинал')
axes[1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
axes[1].set_title('С шумом «соль и перец»')
for ax, (name, img) in zip(axes[2:], filters.items()):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(name)
for a in axes:
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step4_smoothing.png', dpi=120)
plt.close()

print('\n=== Шаг 5. Оператор Собеля ===')
blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(sx ** 2 + sy ** 2)
mag8 = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
print('Максимум градиента по X: %.1f, по Y: %.1f, по модулю: %.1f' % (np.abs(sx).max(), np.abs(sy).max(), mag.max()))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
ax[0].imshow(blur, cmap='gray')
ax[0].set_title('Сглаженное серое изображение')
ax[1].imshow(np.abs(sx), cmap='gray')
ax[1].set_title('Sobel по X (вертикальные границы)')
ax[2].imshow(np.abs(sy), cmap='gray')
ax[2].set_title('Sobel по Y (горизонтальные границы)')
ax[3].imshow(mag8, cmap='gray')
ax[3].set_title('Модуль градиента')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step5_sobel.png', dpi=120)
plt.close()

print('\n=== Шаг 6. Детектор границ Кэнни ===')
thresholds = [(20, 60), (60, 120), (150, 250)]
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Серое изображение')
for a, (lo, hi) in zip(ax[1:], thresholds):
    edges = cv2.Canny(blur, lo, hi)
    print(f'Canny пороги {lo}/{hi}: пикселей границ = {int((edges > 0).sum())} ({(edges > 0).mean() * 100:.1f}%)')
    a.imshow(edges, cmap='gray')
    a.set_title(f'Canny {lo}/{hi}')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step6_canny.png', dpi=120)
plt.close()

print('\nГотово')
