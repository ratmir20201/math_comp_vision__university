from pathlib import Path

import cv2

Path("output").mkdir(exist_ok=True)
img = cv2.imread("input/image.jpg")
if img is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")
h, w = img.shape[:2]


def v1():
    small = cv2.resize(
        img, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA
    )
    sh, sw = small.shape[:2]
    m = cv2.getRotationMatrix2D((sw / 2, sh / 2), 15, 1.0)
    rotated = cv2.warpAffine(small, m, (sw, sh))
    contrast = cv2.convertScaleAbs(rotated, alpha=1.5, beta=-60)
    cv2.imwrite("output/v1_result.png", contrast)
    print("Исходный размер:", w, "x", h, "-> после уменьшения на 25 %:", sw, "x", sh)
    print(
        "Стандартное отклонение яркости: до",
        round(float(rotated.std()), 1),
        "после",
        round(float(contrast.std()), 1),
    )


if __name__ == "__main__":
    print("Вариант 1")
    v1()
