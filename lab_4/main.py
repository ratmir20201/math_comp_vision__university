import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

INPUT = Path("input")
OUT_CV = Path("output/opencv")
OUT_PIL = Path("output/pil")
for folder in (INPUT, OUT_CV, OUT_PIL):
    folder.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def psnr(a, b):
    return float(cv2.PSNR(a, b))


def sharpness(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def edge_stats(edges):
    count = int((edges > 0).sum())
    return count, 100.0 * count / edges.size


def normalize8(arr):
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_cv(name, img):
    cv2.imwrite(str(OUT_CV / name), img)


def save_pil(name, img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(OUT_PIL / name)


def show_grid(images, titles, filename, cols=3):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_CV / filename, dpi=120)
    plt.show()


def add_gaussian_noise(img, sigma=15):
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.03):
    noisy = img.copy()
    mask = rng.random(img.shape)
    noisy[mask < amount / 2] = 0
    noisy[(mask >= amount / 2) & (mask < amount)] = 255
    return noisy


def median_cv(img, k):
    return cv2.medianBlur(img, k)


def median_pil(img, size):
    return np.array(Image.fromarray(img).filter(ImageFilter.MedianFilter(size)))


def box_blur_pil(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.BoxBlur(radius)))


def gaussian_cv(img, k, sigma):
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def gaussian_pil(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))


def bilateral_cv(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def sharpen_cv(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def sharpen_pil(img):
    return np.array(Image.fromarray(img).filter(ImageFilter.SHARPEN))


def unsharp_pil(img):
    return np.array(
        Image.fromarray(img).filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
    )


def sobel_cv(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return (
        normalize8(np.abs(gx)),
        normalize8(np.abs(gy)),
        normalize8(cv2.magnitude(gx, gy)),
    )


def scharr_cv(img):
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    return (
        normalize8(np.abs(gx)),
        normalize8(np.abs(gy)),
        normalize8(cv2.magnitude(gx, gy)),
    )


def prewitt_cv(img):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = cv2.filter2D(img.astype(np.float64), -1, kx)
    gy = cv2.filter2D(img.astype(np.float64), -1, ky)
    return (
        normalize8(np.abs(gx)),
        normalize8(np.abs(gy)),
        normalize8(cv2.magnitude(gx, gy)),
    )


def laplacian_cv(img, ksize=3):
    return normalize8(np.abs(cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)))


def find_edges_pil(img):
    return np.array(Image.fromarray(img).filter(ImageFilter.FIND_EDGES))


def contour_pil(img):
    return np.array(Image.fromarray(img).filter(ImageFilter.CONTOUR))


def canny_cv(img, low, high):
    return cv2.Canny(img, low, high)


def pipeline(img):
    timings = {}
    t0 = time.perf_counter()
    noisy = add_gaussian_noise(img, 15)
    timings["шум"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    filtered = gaussian_cv(noisy, 5, 1.0)
    timings["фильтрация"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    edges = canny_cv(filtered, 100, 200)
    timings["границы"] = (time.perf_counter() - t0) * 1000
    timings["всего"] = sum(timings.values())
    return noisy, filtered, edges, timings


def variant_1(img):
    blurred = gaussian_cv(img, 5, 1.0)
    edges_original = canny_cv(img, 50, 150)
    edges_blurred = canny_cv(blurred, 50, 150)
    return blurred, edges_original, edges_blurred


print("=== Задание 1. Папки ===")
print("Созданы:", INPUT, OUT_CV, OUT_PIL)

image_path = INPUT / "image.png"
image_cv_color = cv2.imread(str(image_path))
if image_cv_color is None:
    raise FileNotFoundError("Положите изображение в input/image.png")
image_pil = Image.open(image_path)

print("\n=== Задание 2. Свойства и оттенки серого ===")
h, w = image_cv_color.shape[:2]
print(
    "OpenCV: размер",
    w,
    "x",
    h,
    "| каналов:",
    image_cv_color.shape[2],
    "| dtype:",
    image_cv_color.dtype,
)
print(
    "PIL: размер",
    image_pil.size,
    "| режим:",
    image_pil.mode,
    "| формат:",
    image_pil.format,
)

gray_cv = cv2.cvtColor(image_cv_color, cv2.COLOR_BGR2GRAY)
gray_pil = np.array(image_pil.convert("L"))
save_cv("gray_opencv.png", gray_cv)
save_pil("gray_pil.png", gray_pil)
print(
    "Серое OpenCV:",
    gray_cv.shape,
    "| PIL:",
    gray_pil.shape,
    "| максимальная разница пикселей:",
    int(np.abs(gray_cv.astype(int) - gray_pil.astype(int)).max()),
)

original = gray_cv
print("Резкость исходного (дисперсия Лапласиана):", round(sharpness(original), 1))

print("\n=== Задание 3. Шум ===")
noisy_g = add_gaussian_noise(original, 15)
noisy_sp = add_salt_pepper(original, 0.03)
save_cv("noise_gaussian_sigma15.png", noisy_g)
save_cv("noise_salt_pepper_3pct.png", noisy_sp)
changed = float((noisy_sp != original).mean() * 100)
print("PSNR гауссовского шума:", round(psnr(noisy_g, original), 2), "дБ")
print("PSNR шума соль и перец:", round(psnr(noisy_sp, original), 2), "дБ")
print("Изменено пикселей при соль и перец, %:", round(changed, 2))
show_grid(
    [original, noisy_g, noisy_sp],
    ["Исходное", "Гауссовский шум, сигма 15", "Соль и перец, 3 %"],
    "task3_noise.png",
)

print("\n=== Задание 4. Медианный (OpenCV) и BoxBlur (PIL) ===")
for noise_name, noisy in (("гауссовский", noisy_g), ("соль и перец", noisy_sp)):
    print("Шум:", noise_name)
    results, titles = [], []
    for k in (3, 5, 9):
        res = median_cv(noisy, k)
        save_cv(f'median_{k}x{k}_{"g" if noisy is noisy_g else "sp"}.png', res)
        print(
            "  OpenCV медианный",
            f"{k}x{k}",
            ": PSNR",
            round(psnr(res, original), 2),
            "| резкость",
            round(sharpness(res), 1),
        )
        results.append(res)
        titles.append(f"Median {k}x{k}")
    for r in (1, 2, 4):
        res = box_blur_pil(noisy, r)
        save_pil(f'boxblur_r{r}_{"g" if noisy is noisy_g else "sp"}.png', res)
        print(
            "  PIL BoxBlur радиус",
            r,
            ": PSNR",
            round(psnr(res, original), 2),
            "| резкость",
            round(sharpness(res), 1),
        )
        results.append(res)
        titles.append(f"BoxBlur r={r}")
    show_grid(results, titles, f'task4_{"g" if noisy is noisy_g else "sp"}.png')

print("\n=== Задание 5. Гауссово размытие ===")
g_results, g_titles = [], []
for k, s in ((5, 1.0), (9, 2.0)):
    res = gaussian_cv(noisy_g, k, s)
    save_cv(f"gaussian_{k}x{k}_sigma{s}.png", res)
    print(
        "OpenCV GaussianBlur",
        f"{k}x{k}",
        "sigma",
        s,
        ": PSNR",
        round(psnr(res, original), 2),
        "| резкость",
        round(sharpness(res), 1),
    )
    g_results.append(res)
    g_titles.append(f"OpenCV {k}x{k}, sigma {s}")
for r in (1, 3):
    res = gaussian_pil(noisy_g, r)
    save_pil(f"gaussian_radius{r}.png", res)
    print(
        "PIL GaussianBlur радиус",
        r,
        ": PSNR",
        round(psnr(res, original), 2),
        "| резкость",
        round(sharpness(res), 1),
    )
    g_results.append(res)
    g_titles.append(f"PIL radius {r}")
show_grid(g_results, g_titles, "task5_gaussian.png", cols=2)

print("\n=== Задание 6. Соль и перец: медианные фильтры 3, 5, 7 ===")
best = None
m_results, m_titles = [], []
for size in (3, 5, 7):
    res_cv = median_cv(noisy_sp, size)
    res_pil = median_pil(noisy_sp, size)
    save_cv(f"sp_median_cv_{size}.png", res_cv)
    save_pil(f"sp_median_pil_{size}.png", res_pil)
    p_cv, p_pil = psnr(res_cv, original), psnr(res_pil, original)
    print(
        "Размер", size, ": OpenCV PSNR", round(p_cv, 2), "| PIL PSNR", round(p_pil, 2)
    )
    m_results += [res_cv, res_pil]
    m_titles += [f"OpenCV median {size}", f"PIL median {size}"]
    for lib, val in (("OpenCV", p_cv), ("PIL", p_pil)):
        if best is None or val > best[2]:
            best = (lib, size, val)
print(
    "Лучший результат по PSNR:",
    best[0],
    "размер",
    best[1],
    "PSNR",
    round(best[2], 2),
    "дБ",
)
show_grid(m_results, m_titles, "task6_median.png", cols=2)

print("\n=== Задание 7. Двусторонний фильтр и гауссов ===")
bil = bilateral_cv(noisy_g, 9, 75, 75)
gau = gaussian_cv(noisy_g, 9, 2.0)
save_cv("bilateral_d9_75_75.png", bil)
save_cv("gaussian_9x9_for_compare.png", gau)
print("PSNR зашумлённого:", round(psnr(noisy_g, original), 2))
print(
    "PSNR двусторонний:",
    round(psnr(bil, original), 2),
    "| резкость",
    round(sharpness(bil), 1),
)
print(
    "PSNR гауссов 9x9 sigma 2:",
    round(psnr(gau, original), 2),
    "| резкость",
    round(sharpness(gau), 1),
)
show_grid(
    [noisy_g, bil, gau],
    ["Зашумлённое", "Двусторонний", "Гауссов 9x9"],
    "task7_bilateral.png",
)

print("\n=== Задание 8. Повышение резкости ===")
sh_cv = sharpen_cv(original)
sh_pil = sharpen_pil(original)
un_pil = unsharp_pil(original)
save_cv("sharpen_kernel.png", sh_cv)
save_pil("sharpen_pil.png", sh_pil)
save_pil("unsharp_mask_pil.png", un_pil)
for name, res in (
    ("OpenCV ядро", sh_cv),
    ("PIL SHARPEN", sh_pil),
    ("PIL UnsharpMask", un_pil),
):
    print(
        name,
        ": резкость",
        round(sharpness(res), 1),
        "| PSNR к исходному",
        round(psnr(res, original), 2),
    )
show_grid(
    [original, sh_cv, sh_pil, un_pil],
    ["Исходное", "OpenCV ядро", "PIL SHARPEN", "PIL UnsharpMask"],
    "task8_sharpen.png",
    cols=2,
)

print("\n=== Задание 9. Собель и Шарр ===")
sx, sy, smag = sobel_cv(original)
cx, cy, cmag = scharr_cv(original)
save_cv("sobel_x.png", sx)
save_cv("sobel_y.png", sy)
save_cv("sobel_magnitude.png", smag)
save_cv("scharr_magnitude.png", cmag)
print(
    "Собель: средняя интенсивность границ",
    round(float(smag.mean()), 2),
    "| Шарр:",
    round(float(cmag.mean()), 2),
)
print(
    "Корреляция карт Собеля и Шарра:",
    round(float(np.corrcoef(smag.ravel(), cmag.ravel())[0, 1]), 4),
)
show_grid(
    [sx, sy, smag, cmag],
    ["Собель X", "Собель Y", "Собель |G|", "Шарр |G|"],
    "task9_sobel_scharr.png",
    cols=2,
)

print("\n=== Задание 10. Лаплас, Превитт, FIND_EDGES, CONTOUR ===")
lap = laplacian_cv(original)
px, py, pmag = prewitt_cv(original)
fe = find_edges_pil(original)
ct = contour_pil(original)
save_cv("laplacian.png", lap)
save_cv("prewitt_magnitude.png", pmag)
save_pil("find_edges.png", fe)
save_pil("contour.png", ct)
for name, res in (
    ("Лаплас", lap),
    ("Превитт", pmag),
    ("FIND_EDGES", fe),
    ("CONTOUR", ct),
):
    print(
        name,
        ": форма",
        res.shape,
        "| режим серый | средняя яркость",
        round(float(res.mean()), 2),
    )
show_grid(
    [lap, pmag, fe, ct],
    ["Лаплас", "Превитт", "PIL FIND_EDGES", "PIL CONTOUR"],
    "task10_edges.png",
    cols=2,
)

print("\n=== Задание 11. Кэнни после гауссова фильтра ===")
smooth = gaussian_cv(original, 5, 1.0)
canny_results, canny_titles = [], []
for low, high in ((50, 150), (100, 200), (150, 250)):
    edges = canny_cv(smooth, low, high)
    save_cv(f"canny_{low}_{high}.png", edges)
    count, pct = edge_stats(edges)
    print(
        "Пороги",
        low,
        high,
        ": граничных пикселей",
        count,
        "из",
        edges.size,
        "=",
        round(pct, 2),
        "%",
    )
    canny_results.append(edges)
    canny_titles.append(f"Canny {low}/{high}")
show_grid(canny_results, canny_titles, "task11_canny.png")

print("\n=== Задание 12. Общая последовательность ===")
noisy_p, filtered_p, edges_p, timings = pipeline(original)
for name, value in timings.items():
    print("Время", name, ":", round(value, 2), "мс")
save_cv("pipeline_noisy.png", noisy_p)
save_cv("pipeline_filtered.png", filtered_p)
save_cv("pipeline_edges.png", edges_p)
show_grid(
    [original, noisy_p, filtered_p, edges_p],
    ["Исходное", "Зашумлённое", "Отфильтрованное", "Границы Canny"],
    "pipeline_comparison.png",
    cols=4,
)

print("\n=== Вариант 1 ===")
blurred, edges_o, edges_b = variant_1(original)
save_cv("variant1_gaussian_5x5_sigma1.png", blurred)
save_cv("variant1_canny_original_50_150.png", edges_o)
save_cv("variant1_canny_blurred_50_150.png", edges_b)
print("PSNR размытого относительно исходного:", round(psnr(blurred, original), 2), "дБ")
print(
    "Резкость: исходное",
    round(sharpness(original), 1),
    "| размытое",
    round(sharpness(blurred), 1),
)
co, po = edge_stats(edges_o)
cb, pb = edge_stats(edges_b)
print("Canny 50/150 на исходном:", co, "пикселей", round(po, 2), "%")
print("Canny 50/150 после размытия:", cb, "пикселей", round(pb, 2), "%")
print("Граничных пикселей стало меньше на", round(100 * (co - cb) / co, 1), "%")
show_grid(
    [original, blurred, edges_o, edges_b],
    [
        "Исходное",
        "GaussianBlur 5x5, sigma 1.0",
        "Canny 50/150 (исходное)",
        "Canny 50/150 (после размытия)",
    ],
    "variant1_comparison.png",
    cols=2,
)
