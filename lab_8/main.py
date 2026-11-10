import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)
DATA = BASE / 'data' / 'coco128'
if not DATA.exists():
    with zipfile.ZipFile(BASE / 'coco128.zip') as z:
        z.extractall(BASE / 'data')
IMG_DIR = DATA / 'images' / 'train2017'
LBL_DIR = DATA / 'labels' / 'train2017'
CONF = 0.25
IOU_MATCH = 0.5


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


log = open(OUT / 'console.txt', 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log)


def read_labels(path, w, h):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])
                boxes.append((c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def draw(img, box, text, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def save_grid(images, titles, path, cols):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


model = YOLO(str(BASE / 'yolo11n.pt'))
names = model.names
images = sorted(IMG_DIR.glob('*.jpg'))

print('=== Шаг 1. Изучение набора данных ===')
counter = Counter()
for p in images:
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    for c, *_ in read_labels(LBL_DIR / (p.stem + '.txt'), w, h):
        counter[c] += 1
print(f'Набор данных: COCO128, изображений: {len(images)}, объектов: {sum(counter.values())}, классов в разметке: {len(counter)} из {len(names)}')
print('Самые частые классы:')
for c, n in counter.most_common(8):
    print(f'  {names[c]}: {n}')

top = counter.most_common(10)
plt.figure(figsize=(9, 4.5))
plt.bar([names[c] for c, _ in top], [n for _, n in top], color='#4c72b0')
plt.xticks(rotation=40, ha='right')
plt.ylabel('Число объектов')
plt.title('Распределение классов в наборе COCO128 (10 самых частых)')
plt.tight_layout()
plt.savefig(OUT / 'step1_classes.png', dpi=110)
plt.close()

samples = [images[i] for i in (0, 9, 24, 40)]
show, titles = [], []
for p in samples:
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    for c, x1, y1, x2, y2 in read_labels(LBL_DIR / (p.stem + '.txt'), w, h):
        draw(im, (x1, y1, x2, y2), names[c], (60, 160, 60))
    show.append(im)
    titles.append(f'Разметка: {p.name}')
save_grid(show, titles, OUT / 'step1_dataset.png', 2)

print('\n=== Шаг 2. Загрузка предобученной модели ===')
print(f'Модель: yolo11n, классов: {len(names)}, параметров: {sum(p.numel() for p in model.model.parameters()):,}')

print('\n=== Шаг 3. Обнаружение объектов на тестовых изображениях ===')
test_ids = [0, 9, 24, 40, 60, 90]
test_imgs = [images[i] for i in test_ids]
show, titles = [], []
for p in test_imgs:
    res = model.predict(str(p), conf=CONF, verbose=False)[0]
    im = cv2.imread(str(p))
    print(f'{p.name}: найдено объектов {len(res.boxes)}')
    for b in res.boxes:
        c = int(b.cls)
        conf = float(b.conf)
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        print(f'    {names[c]}: уверенность {conf:.2f}, рамка ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})')
        draw(im, (x1, y1, x2, y2), f'{names[c]} {conf:.2f}', (0, 120, 255))
    show.append(im)
    titles.append(f'{p.name}: {len(res.boxes)} объектов')
save_grid(show, titles, OUT / 'step3_detections.png', 2)

print('\n=== Шаг 4. Метрики Precision, Recall, mAP ===')
yaml_path = BASE / 'coco128.yaml'
yaml_path.write_text(
    f"path: {DATA.as_posix()}\ntrain: images/train2017\nval: images/train2017\nnames:\n" +
    ''.join(f'  {k}: {v}\n' for k, v in names.items()), encoding='utf-8')
m = model.val(data=str(yaml_path), conf=0.001, iou=0.6, plots=False, verbose=False,
              project=str(OUT), name='val', exist_ok=True, workers=0, batch=16)
P, R = float(m.box.mp), float(m.box.mr)
print(f'Precision: {P:.3f}')
print(f'Recall: {R:.3f}')
print(f'mAP@0.5: {float(m.box.map50):.3f}')
print(f'mAP@0.5:0.95: {float(m.box.map):.3f}')

print('\n=== Шаг 5. Правильные и ошибочные обнаружения ===')
tp = fp = fn = 0
fp_cases, fn_cases = [], []
per_image = []
for p in images:
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    gts = read_labels(LBL_DIR / (p.stem + '.txt'), w, h)
    res = model.predict(str(p), conf=CONF, verbose=False)[0]
    preds = [(int(b.cls), float(b.conf), *b.xyxy[0].tolist()) for b in res.boxes]
    used = set()
    ptp = pfp = 0
    pred_state = []
    for c, conf, x1, y1, x2, y2 in sorted(preds, key=lambda t: -t[1]):
        best, bi = 0, -1
        for i, g in enumerate(gts):
            if i in used or g[0] != c:
                continue
            v = iou((x1, y1, x2, y2), g[1:])
            if v > best:
                best, bi = v, i
        if best >= IOU_MATCH:
            used.add(bi)
            ptp += 1
            pred_state.append((c, conf, (x1, y1, x2, y2), True))
        else:
            pfp += 1
            pred_state.append((c, conf, (x1, y1, x2, y2), False))
    missed = [g for i, g in enumerate(gts) if i not in used]
    tp += ptp
    fp += pfp
    fn += len(missed)
    per_image.append((p, pred_state, missed, ptp, pfp, len(missed)))

prec = tp / (tp + fp)
rec = tp / (tp + fn)
print(f'При пороге уверенности {CONF} и IoU {IOU_MATCH}: верных {tp}, ложных срабатываний {fp}, пропущенных объектов {fn}')
print(f'Precision: {prec:.3f}, Recall: {rec:.3f}')

tp_conf = [c for _, st, _, _, _, _ in per_image for _, c, _, ok in st if ok]
fp_conf = [c for _, st, _, _, _, _ in per_image for _, c, _, ok in st if not ok]
print(f'Средняя уверенность верных обнаружений: {np.mean(tp_conf):.2f}')
print(f'Средняя уверенность ложных обнаружений: {np.mean(fp_conf):.2f}')

fn_names = Counter(names[g[0]] for _, _, ms, _, _, _ in per_image for g in ms)
fp_names = Counter(names[c] for _, st, _, _, _, _ in per_image for c, _, _, ok in st if not ok)
print('Чаще всего пропускаются: ' + ', '.join(f'{k} ({v})' for k, v in fn_names.most_common(5)))
print('Чаще всего ложно обнаруживаются: ' + ', '.join(f'{k} ({v})' for k, v in fp_names.most_common(5)))

plt.figure(figsize=(7, 4))
plt.hist([tp_conf, fp_conf], bins=10, range=(0.25, 1.0), label=['верные', 'ложные'], color=['#55a868', '#c44e52'])
plt.xlabel('Уверенность модели')
plt.ylabel('Число обнаружений')
plt.legend()
plt.title('Уверенность верных и ложных обнаружений')
plt.tight_layout()
plt.savefig(OUT / 'step5_confidence.png', dpi=110)
plt.close()

good = sorted(per_image, key=lambda t: (t[4] + t[5], -t[3]))[:2]
bad = sorted(per_image, key=lambda t: -(t[4] + t[5]))[:2]
show, titles = [], []
for group in (good, bad):
    for p, st, missed, ptp, pfp, pfn in group:
        im = cv2.imread(str(p))
        for c, conf, box, ok in st:
            draw(im, box, f'{names[c]} {conf:.2f}', (60, 160, 60) if ok else (0, 0, 220))
        for g in missed:
            draw(im, g[1:], f'пропуск: {names[g[0]]}', (200, 100, 0))
        show.append(im)
        titles.append(f'{p.name}: верных {ptp}, ложных {pfp}, пропусков {pfn}')
save_grid(show, titles, OUT / 'step5_compare.png', 2)
print('\nЗелёные рамки - верные обнаружения, красные - ложные, синие - пропущенные объекты')
print('\nГотово')
