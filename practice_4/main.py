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

print('=== Шаг 1. Загрузка изображения ===')
bgr = cv2.imread(str(BASE / 'input' / 'fruits.jpg'))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print('Размер изображения (высота, ширина, каналы):', bgr.shape)
print('Тип данных:', bgr.dtype)

print('\n=== Шаг 2. Преобразование RGB -> HSV ===')
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
for i, name in enumerate(('H (оттенок)', 'S (насыщенность)', 'V (яркость)')):
    ch = hsv[:, :, i]
    print(f'{name}: min={ch.min()}, max={ch.max()}, среднее={ch.mean():.1f}')
print('Фон (угол изображения) в HSV:', hsv[5, 5])

print('\n=== Шаг 3. Диапазоны HSV для каждого вида фруктов ===')
ranges = {
    'Яблоко (красный)': [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    'Апельсин (оранжевый)': [((9, 120, 70), (20, 255, 255))],
    'Банан (жёлтый)': [((22, 120, 70), (35, 255, 255))],
}
colors = {
    'Яблоко (красный)': (0, 255, 0),
    'Апельсин (оранжевый)': (0, 0, 255),
    'Банан (жёлтый)': (255, 0, 255),
}
for name, parts in ranges.items():
    print(name + ':', ' и '.join(f'H {lo[0]}-{hi[0]}, S {lo[1]}-{hi[1]}, V {lo[2]}-{hi[2]}' for lo, hi in parts))

print('\n=== Шаг 4. Бинарные маски ===')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
masks = {}
for name, parts in ranges.items():
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in parts:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    masks[name] = mask
    print(f'{name}: белых пикселей = {int(cv2.countNonZero(mask))}, доля кадра = {cv2.countNonZero(mask) / mask.size * 100:.2f}%')
overlap = sum(int(cv2.countNonZero(cv2.bitwise_and(masks[a], masks[b]))) for a, b in (
    ('Яблоко (красный)', 'Апельсин (оранжевый)'), ('Апельсин (оранжевый)', 'Банан (жёлтый)'), ('Яблоко (красный)', 'Банан (жёлтый)')))
print('Пересечение масок разных видов, пикселей:', overlap)

print('\n=== Шаг 5. Контуры объектов ===')
result = rgb.copy()
min_area = 1500
for name, mask in masks.items():
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    cv2.drawContours(result, contours, -1, colors[name], 4)
    print(f'{name}: найдено объектов = {len(contours)}')
    for i, c in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(c)
        print(f'  объект {i}: площадь = {cv2.contourArea(c):.0f}, периметр = {cv2.arcLength(c, True):.1f}, рамка (x, y, w, h) = ({x}, {y}, {w}, {h})')

print('\n=== Шаг 6. Сохранение результатов ===')
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(rgb)
ax[0].set_title('Исходное изображение')
ax[1].imshow(hsv[:, :, 0], cmap='hsv', vmin=0, vmax=179)
ax[1].set_title('Канал H (оттенок)')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step1_original_hsv.png', dpi=110)
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (name, mask) in zip(ax, masks.items()):
    a.imshow(mask, cmap='gray')
    a.set_title('Маска: ' + name)
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step2_masks.png', dpi=110)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(rgb)
ax[0].set_title('Исходное изображение')
ax[1].imshow(result)
ax[1].set_title('Контуры: яблоки - зелёный, апельсины - синий, бананы - пурпурный')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig(BASE / 'output' / 'step3_result.png', dpi=110)
plt.close()
cv2.imwrite(str(BASE / 'output' / 'result.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print('Файлы сохранены в папку output')
