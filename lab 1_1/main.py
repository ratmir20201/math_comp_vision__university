import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.uint8)

print(A)
print("Форма:", A.shape)
print("Количество измерений:", A.ndim)
print("Тип данных:", A.dtype)
print("Минимум, максимум, среднее:", A.min(), A.max(), A.mean())
print("Количество элементов:", A.size)
print("A[1, 2] =", A[1, 2])

gray = np.array(
    [
        [0, 40, 80, 120, 160],
        [40, 80, 120, 160, 200],
        [80, 120, 160, 200, 240],
        [120, 160, 200, 240, 255],
        [160, 200, 240, 255, 255],
    ],
    dtype=np.uint8,
)

plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.colorbar(label="Интенсивность пикселей")
plt.title("Матрица оттенков серого")
plt.show()

gray_changed = gray.copy()
gray_changed[2, 2] = 0
plt.imshow(gray_changed, cmap="gray", vmin=0, vmax=255)
plt.colorbar(label="Интенсивность пикселей")
plt.title("Матрица после изменения элемента [2, 2]")
plt.show()

rgb = np.zeros((200, 300, 3), dtype=np.uint8)
rgb[:, :100] = [255, 0, 0]
rgb[:, 100:200] = [0, 255, 0]
rgb[:, 200:] = [0, 0, 255]

plt.imshow(rgb)
plt.title(f"RGB Image: {rgb.shape}")
plt.axis("off")
plt.show()

print("rgb[50, 150] =", rgb[50, 150])

img = Image.open("image.jpg").convert("RGB")
img_array = np.array(img)

print("Форма:", img_array.shape)
print("Тип:", img_array.dtype)
print("Минимальное/максимальное значение:", img_array.min(), img_array.max())
print("Среднее значение:", img_array.mean())

plt.imshow(img_array)
plt.axis("off")
plt.show()

h, w, _ = img_array.shape
print("Высота:", h, "Ширина:", w, "Каналов:", img_array.shape[2])
fragment = img_array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
print("Форма фрагмента:", fragment.shape)

plt.imshow(fragment)
plt.title("Центр фрагмента")
plt.axis("off")
plt.show()

R = img_array[:, :, 0]
G = img_array[:, :, 1]
B = img_array[:, :, 2]

print(
    "Среднее по каналам R, G, B:",
    R.mean().round(2),
    G.mean().round(2),
    B.mean().round(2),
)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, channel, title in zip(ax, [R, G, B], ["Red", "Green", "Blue"]):
    a.imshow(channel, cmap="gray", vmin=0, vmax=255)
    a.set_title(title)
    a.axis("off")
plt.tight_layout()
plt.show()

bright = np.clip(img_array.astype(np.int16) + 50, 0, 255).astype(np.uint8)
contrast = np.clip((img_array.astype(np.float32) - 128) * 1.5 + 128, 0, 255).astype(
    np.uint8
)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, im, title in zip(
    ax, [img_array, bright, contrast], ["Original", "Brightness +50", "Contrast ×1.5"]
):
    a.imshow(im)
    a.set_title(title)
    a.axis("off")
plt.tight_layout()
plt.show()

print(
    "Среднее: оригинал",
    img_array.mean().round(2),
    "яркость",
    bright.mean().round(2),
    "контраст",
    contrast.mean().round(2),
)

print("Форма:", img_array.shape)
print("Тип данных:", img_array.dtype)
print(
    "Минимум:",
    img_array.min(),
    "Максимум:",
    img_array.max(),
    "Среднее:",
    img_array.mean(),
)

center = img_array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

no_green = img_array.copy()
no_green[:, :, 1] = 0

flipped = np.flip(img_array, axis=1)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for a, im, title in zip(
    ax.ravel(),
    [img_array, center, no_green, flipped],
    [
        "Оригинал",
        "Центральный фрагмент",
        "Зелёный канал = 0",
        "Отражение по горизонтали",
    ],
):
    a.imshow(im)
    a.set_title(title)
    a.axis("off")
plt.tight_layout()
plt.show()

print(
    "Среднее по каналам после обнуления G:",
    no_green[:, :, 0].mean().round(2),
    no_green[:, :, 1].mean().round(2),
    no_green[:, :, 2].mean().round(2),
)
print(
    "Совпадает ли flipped[:, 0] с оригиналом[:, -1]:",
    np.array_equal(flipped[:, 0], img_array[:, -1]),
)
