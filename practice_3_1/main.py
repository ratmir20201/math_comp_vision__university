from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

path = Path('lenna.png')
print('Путь к изображению:', path.resolve())
print('Файл существует:', path.exists())

image_cv = cv2.imread(str(path))
image_pil = Image.open(path)

print('Тип OpenCV:', type(image_cv), '| dtype:', image_cv.dtype)
print('Тип PIL:', type(image_pil))
print('Форма OpenCV:', image_cv.shape)
print('Размер PIL (ширина, высота):', image_pil.size, '| режим:', image_pil.mode, '| формат:', image_pil.format)
print('Минимум и максимум OpenCV:', image_cv.min(), image_cv.max())
arr_pil = np.array(image_pil)
print('Минимум и максимум PIL:', arr_pil.min(), arr_pil.max())
for i, name in enumerate(['B', 'G', 'R']):
    print('Канал', name, ': min', image_cv[:, :, i].min(), 'max', image_cv[:, :, i].max(), 'mean', round(float(image_cv[:, :, i].mean()), 2))

cv2.imshow('OpenCV: lenna', image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_pil.show()

plt.figure(figsize=(5, 5))
plt.imshow(image_cv)
plt.title('Matplotlib без преобразования (BGR как RGB)')
plt.axis('off')
plt.show()

image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.imshow(image_rgb)
plt.title('Matplotlib после BGR -> RGB')
plt.axis('off')
plt.show()

cv2.imwrite('lenna_saved_cv.jpg', image_cv)
image_pil.save('lenna_saved_pil.jpg')
print('Размер lenna.png, КБ:', round(path.stat().st_size / 1024, 1))
print('Размер lenna_saved_cv.jpg, КБ:', round(Path('lenna_saved_cv.jpg').stat().st_size / 1024, 1))
print('Размер lenna_saved_pil.jpg, КБ:', round(Path('lenna_saved_pil.jpg').stat().st_size / 1024, 1))

gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
print('Форма серого изображения:', gray.shape, '| dtype:', gray.dtype)
print('Минимум и максимум серого:', gray.min(), gray.max())
gray_pil = image_pil.convert('L')
print('PIL L: размер', gray_pil.size, 'режим', gray_pil.mode)

plt.figure(figsize=(5, 5))
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.title('Lenna в оттенках серого')
plt.axis('off')
plt.show()

barbara = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)
print('Barbara: форма', barbara.shape, '| min', barbara.min(), '| max', barbara.max())

plt.figure(figsize=(5, 5))
plt.imshow(barbara, cmap='gray', vmin=0, vmax=255)
plt.title('Barbara (IMREAD_GRAYSCALE)')
plt.axis('off')
plt.show()

baboon_bgr = cv2.imread('baboon.png')
print('Baboon: форма', baboon_bgr.shape, '| dtype', baboon_bgr.dtype)
baboon = cv2.cvtColor(baboon_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5, 5))
plt.imshow(baboon)
plt.title('Baboon (RGB)')
plt.axis('off')
plt.show()

R, G, B = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
print('Средние значения каналов R, G, B:', round(float(R.mean()), 2), round(float(G.mean()), 2), round(float(B.mean()), 2))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, channel, title in zip(ax, [R, G, B], ['Red', 'Green', 'Blue']):
    a.imshow(channel, cmap='gray', vmin=0, vmax=255)
    a.set_title(title)
    a.axis('off')
plt.tight_layout()
plt.show()

y1, y2, x1, x2 = 30, 130, 120, 400
roi = baboon[y1:y2, x1:x2]
print('ROI: форма', roi.shape, '| среднее', round(float(roi.mean()), 2))

plt.figure(figsize=(6, 3))
plt.imshow(roi)
plt.title('ROI: область глаз')
plt.axis('off')
plt.show()

modified = baboon.copy()
modified[y1:y2, x1:x2] = 255 - modified[y1:y2, x1:x2]
cv2.rectangle(modified, (x1, y1), (x2, y2), (0, 255, 0), 3)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(baboon)
ax[0].set_title('Оригинал')
ax[1].imshow(modified)
ax[1].set_title('ROI инвертирована')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite('baboon_modified.png', cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
print('Сохранено: baboon_modified.png')
print('Оригинал не изменён:', bool(np.array_equal(baboon, cv2.cvtColor(baboon_bgr, cv2.COLOR_BGR2RGB))))