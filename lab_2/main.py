from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageFilter

Path("output").mkdir(exist_ok=True)

image_cv = cv2.imread("input/image.jpg")
if image_cv is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

height, width, channels = image_cv.shape
print("Размер:", width, "x", height)
print("Количество каналов:", channels)
print("Тип данных:", image_cv.dtype)

image_pil = Image.open("input/image.jpg")
print("Размер:", image_pil.size)
print("Цветовой режим:", image_pil.mode)
print("Формат:", image_pil.format)

image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 5))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Изображение, считанное OpenCV")
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(image_cv)
plt.axis("off")
plt.title("Без cvtColor: каналы перепутаны")
plt.show()

gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/gray_opencv.png", gray_cv)
gray_pil = image_pil.convert("L")
gray_pil.save("output/gray_pil.png")
print(
    "Серое OpenCV:",
    gray_cv.shape,
    "PIL:",
    gray_pil.size,
    "разница пикселей:",
    int(np.abs(gray_cv.astype(int) - np.array(gray_pil).astype(int)).max()),
)

new_size = (640, 480)
resized_cv = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_AREA)
resized_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
cv2.imwrite("output/resized_opencv.jpg", resized_cv)
resized_pil.save("output/resized_pil.jpg", quality=90)

target_width = 600
h, w = image_cv.shape[:2]
ratio = target_width / w
target_height = int(h * ratio)
resized_ratio = cv2.resize(
    image_cv, (target_width, target_height), interpolation=cv2.INTER_AREA
)
cv2.imwrite("output/resized_ratio.png", resized_ratio)
print("С сохранением пропорций:", target_width, "x", target_height)

crop_cv = image_cv[50:350, 100:500]
cv2.imwrite("output/crop_opencv.png", crop_cv)
crop_pil = image_pil.crop((100, 50, 500, 350))
crop_pil.save("output/crop_pil.png")
print("Кадр OpenCV:", crop_cv.shape, "PIL:", crop_pil.size)

rotated_cv = cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
flipped_cv = cv2.flip(image_cv, 1)
cv2.imwrite("output/rotated_cv.png", rotated_cv)
cv2.imwrite("output/flipped_cv.png", flipped_cv)

rotated_pil = image_pil.rotate(30, expand=True, fillcolor="white")
flipped_pil = ImageOps.mirror(image_pil)
rotated_pil.save("output/rotated_pil.png")
flipped_pil.save("output/flipped_pil.png")
print(
    "Поворот 90 градусов:",
    rotated_cv.shape,
    "Поворот 30 градусов с expand:",
    rotated_pil.size,
)

blur_cv = cv2.GaussianBlur(image_cv, (9, 9), 0)
edges_cv = cv2.Canny(gray_cv, 80, 160)
cv2.imwrite("output/blur_opencv.png", blur_cv)
cv2.imwrite("output/edges_opencv.png", edges_cv)

blur_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=3))
sharp_pil = image_pil.filter(ImageFilter.SHARPEN)
blur_pil.save("output/blur_pil.png")
sharp_pil.save("output/sharp_pil.png")
print("Доля граничных пикселей Canny:", round(float((edges_cv > 0).mean()), 4))

annotated = image_cv.copy()
cv2.rectangle(annotated, (40, 40), (300, 220), (0, 255, 0), 3)
cv2.circle(annotated, (420, 180), 60, (255, 0, 0), 3)
cv2.putText(annotated, "OpenCV", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("output/annotated.png", annotated)

pil_from_cv = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
cv_from_pil = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
pil_from_cv.save("output/converted_to_pil.png")
cv2.imwrite("output/converted_to_cv.png", cv_from_pil)
print("Массив из PIL:", np.array(image_pil).shape, np.array(image_pil).dtype)

cv2.imwrite("output/quality_95.jpg", image_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite("output/quality_40.jpg", image_cv, [cv2.IMWRITE_JPEG_QUALITY, 40])
image_pil.save("output/pil_quality_95.jpg", quality=95)
image_pil.save("output/pil_quality_40.jpg", quality=40)

print("\nРазмеры файлов, КБ:")
for name in [
    "gray_opencv.png",
    "gray_pil.png",
    "resized_opencv.jpg",
    "resized_pil.jpg",
    "quality_95.jpg",
    "quality_40.jpg",
    "pil_quality_95.jpg",
    "pil_quality_40.jpg",
]:
    print(name, round(Path("output", name).stat().st_size / 1024, 1))

print("\n=== Основная практическая задача ===")
img_cv = cv2.imread("input/image.jpg")
img_pil = Image.open("input/image.jpg")
print(
    "OpenCV:",
    img_cv.shape,
    img_cv.dtype,
    "| PIL:",
    img_pil.size,
    img_pil.mode,
    img_pil.format,
)

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Исходное изображение (RGB)")
plt.show()

h, w = img_cv.shape[:2]
target_w = 800
target_h = int(h * target_w / w)
interp = cv2.INTER_AREA if target_w < w else cv2.INTER_CUBIC
main_resized = cv2.resize(img_cv, (target_w, target_h), interpolation=interp)
cv2.imwrite("output/main_01_resized_800.png", main_resized)
print("После изменения размера:", main_resized.shape[1], "x", main_resized.shape[0])

rh, rw = main_resized.shape[:2]
main_crop = main_resized[rh // 4 : 3 * rh // 4, rw // 4 : 3 * rw // 4]
cv2.imwrite("output/main_02_center_crop.png", main_crop)
print("Центральная часть:", main_crop.shape[1], "x", main_crop.shape[0])

main_blur = cv2.GaussianBlur(main_crop, (9, 9), 0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
main_sharp = cv2.filter2D(main_crop, -1, kernel)
cv2.imwrite("output/main_03_crop_blur.png", main_blur)
cv2.imwrite("output/main_04_crop_sharp.png", main_sharp)
print(
    "Средний градиент: исходный фрагмент",
    round(float(np.mean(np.abs(np.diff(main_crop.astype(float), axis=1)))), 2),
    "размытие",
    round(float(np.mean(np.abs(np.diff(main_blur.astype(float), axis=1)))), 2),
    "резкость",
    round(float(np.mean(np.abs(np.diff(main_sharp.astype(float), axis=1)))), 2),
)

final = main_resized.copy()
cv2.rectangle(final, (rw // 4, rh // 4), (3 * rw // 4, 3 * rh // 4), (0, 255, 0), 3)
cv2.putText(
    final,
    "Pagoda",
    (rw // 4, rh // 4 - 12),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2,
)
cv2.imwrite("output/main_05_final.png", final)
cv2.imwrite("output/main_05_final_q60.jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 60])

png_size = Path("output/main_05_final.png").stat().st_size / 1024
jpg_size = Path("output/main_05_final_q60.jpg").stat().st_size / 1024
print(
    "PNG, КБ:",
    round(png_size, 1),
    "JPEG q60, КБ:",
    round(jpg_size, 1),
    "отношение:",
    round(png_size / jpg_size, 1),
)

from_png = cv2.imread("output/main_05_final.png")
from_jpg = cv2.imread("output/main_05_final_q60.jpg")
diff = np.abs(final.astype(int) - from_jpg.astype(int))
print("PNG совпадает с оригиналом:", bool(np.array_equal(final, from_png)))
print(
    "JPEG: средняя ошибка пикселей",
    round(float(diff.mean()), 2),
    "максимальная",
    int(diff.max()),
)
