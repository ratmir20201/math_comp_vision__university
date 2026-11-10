import hashlib
import os
import ssl
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)
DATA = BASE / 'data'
SIZE = 128
EPOCHS = 10
BATCH = 32
LR = 1e-3
NUM_CLASSES = 3
CLASS_NAMES = ['животное', 'фон', 'граница']
PALETTE = np.array([[230, 60, 60], [40, 40, 40], [250, 220, 60]], dtype=np.uint8)


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

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA.mkdir(exist_ok=True)
IMG_URL = 'https://zenodo.org/records/8067751/files/The%20Oxford-IIIT%20Pet.zip?download=1'
IMG_MD5 = '7f61d2a52733bce84d505617dd7e780a'
ANN_URL = 'https://huggingface.co/datasets/Shiym/ViT-FineTune/resolve/main/oxford-iiit-pet/annotations.tar.gz'


def download(url, dest, md5=None):
    tmp = dest.with_name(dest.name + '.part')
    try:
        r = urllib.request.urlopen(url, timeout=60)
    except (ssl.SSLError, urllib.error.URLError) as e:
        reason = e.reason if isinstance(e, urllib.error.URLError) else e
        if not isinstance(reason, ssl.SSLError):
            raise
        print('Ошибка проверки сертификата, повторная попытка без проверки:', reason)
        r = urllib.request.urlopen(url, timeout=60, context=ssl._create_unverified_context())
    with r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        last = -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done * 10 // total != last:
                last = done * 10 // total
                print(f'  загружено {done * 100 // total}%')
    if md5:
        h = hashlib.md5()
        with open(tmp, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        if h.hexdigest() != md5:
            raise RuntimeError('Контрольная сумма архива не совпала, загрузка повреждена')
    tmp.replace(dest)


def index_images():
    return {p.stem: p for p in DATA.rglob('*.jpg') if not p.name.startswith('._') and '__MACOSX' not in str(p)}


def find_annotations():
    for root, dirs, files in os.walk(DATA):
        if '__MACOSX' in root:
            continue
        p = Path(root)
        if p.name == 'trimaps' and any(f.endswith('.png') and not f.startswith('._') for f in files):
            ann = p.parent
            if (ann / 'trainval.txt').exists() and (ann / 'test.txt').exists():
                return ann
    return None


def ensure_dataset():
    imgs = index_images()
    if len(imgs) < 7000:
        zip_path = DATA / 'pets.zip'
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            print('Скачивание изображений (около 790 МБ)...')
            download(IMG_URL, zip_path, IMG_MD5)
        print('Распаковка изображений...')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)
        imgs = index_images()
    ann = find_annotations()
    if ann is None:
        arc = DATA / 'annotations.tar.gz'
        if not arc.exists():
            print('Скачивание масок сегментации (около 19 МБ)...')
            download(ANN_URL, arc)
        print('Распаковка масок...')
        with tarfile.open(arc) as t:
            t.extractall(DATA)
        ann = find_annotations()
    if ann is None or not imgs:
        raise RuntimeError('Не удалось найти изображения или маски')
    return imgs, ann


def load_split(split):
    cache = DATA / f'pets_{split}_{SIZE}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['x'], z['y']
    imgs, ann_dir = ensure_dataset()
    xs, ys = [], []
    for line in (ann_dir / f'{split}.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name = line.split()[0]
        ip, mp = imgs.get(name), ann_dir / 'trimaps' / f'{name}.png'
        if ip is None or not mp.exists():
            continue
        img = Image.open(ip).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
        mask = Image.open(mp).resize((SIZE, SIZE), Image.NEAREST)
        xs.append(np.asarray(img, dtype=np.uint8))
        ys.append(np.clip(np.asarray(mask, dtype=np.uint8), 1, 3) - 1)
    x, y = np.stack(xs), np.stack(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


print('=== Шаг 1. Загрузка набора данных Oxford-IIIT Pet ===')
x_train, y_train = load_split('trainval')
x_test, y_test = load_split('test')
print(f'Обучающая часть: {len(x_train)} изображений, тестовая часть: {len(x_test)} изображений')

print('\n=== Шаг 2. Единый размер изображений и масок ===')
print(f'Изображения: {x_train.shape[1:]} (высота, ширина, каналы), маски: {y_train.shape[1:]}')
print('Классы масок: 0 - животное, 1 - фон, 2 - граница животного')
freq = np.bincount(y_train.reshape(-1), minlength=NUM_CLASSES) / y_train.size
for i, n in enumerate(CLASS_NAMES):
    print(f'  Доля пикселей класса "{n}": {freq[i] * 100:.1f}%')

print('\n=== Шаг 3. Разделение на обучающую и тестовую части ===')
print(f'Обучение: {len(x_train)} ({len(x_train) / (len(x_train) + len(x_test)) * 100:.0f}%), тест: {len(x_test)} ({len(x_test) / (len(x_train) + len(x_test)) * 100:.0f}%)')

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(x):
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    return (t - mean) / std


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, classes, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = DoubleConv(3, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.bottom = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[3] * 2, c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[2] * 2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[1] * 2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[0] * 2, c[0])
        self.head = nn.Conv2d(c[0], classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def confusion(pred, true):
    idx = true.reshape(-1).astype(np.int64) * NUM_CLASSES + pred.reshape(-1).astype(np.int64)
    return np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_conf(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    acc = tp.sum() / cm.sum()
    return iou, dice, acc


@torch.no_grad()
def predict(model, x, batch=64):
    model.eval()
    preds = []
    for i in range(0, len(x), batch):
        logits = model(to_tensor(x[i:i + batch]).to(device))
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(preds)


print('\n=== Шаг 4. Архитектура U-Net ===')
model = UNet(NUM_CLASSES).to(device)
print(f'Архитектура: U-Net (4 уровня кодировщика, пропускные соединения), параметров: {sum(p.numel() for p in model.parameters()):,}')
print(f'Устройство: {device}')

print('\n=== Шаг 5. Обучение модели ===')
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
xt = to_tensor(x_train)
yt = torch.from_numpy(y_train).long()
sub = np.random.RandomState(1).choice(len(x_test), min(400, len(x_test)), replace=False)
history = {'loss': [], 'val_miou': [], 'val_pet_iou': []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    perm = torch.randperm(len(xt))
    total = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = xt[idx], yt[idx]
        if torch.rand(1).item() < 0.5:
            xb, yb = xb.flip(3), yb.flip(2)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    sched.step()
    pred = predict(model, x_test[sub])
    iou, dice, acc = metrics_from_conf(confusion(pred, y_test[sub]))
    history['loss'].append(total / len(xt))
    history['val_miou'].append(iou.mean())
    history['val_pet_iou'].append(iou[0])
    print(f'Эпоха {epoch}/{EPOCHS}: потери {total / len(xt):.4f}, mIoU на тесте {iou.mean():.3f}, IoU животного {iou[0]:.3f}, время {time.time() - t0:.0f} с')
torch.save(model.state_dict(), OUT / 'unet_pets.pt')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, EPOCHS + 1), history['loss'], marker='o')
ax[0].set_title('Функция потерь на обучении')
ax[0].set_xlabel('Эпоха')
ax[1].plot(range(1, EPOCHS + 1), history['val_miou'], marker='o', label='средний IoU')
ax[1].plot(range(1, EPOCHS + 1), history['val_pet_iou'], marker='s', label='IoU животного')
ax[1].set_title('Качество на тестовой выборке')
ax[1].set_xlabel('Эпоха')
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / 'step5_training.png', dpi=110)
plt.close()

print('\n=== Шаг 6. Оценка качества: IoU и коэффициент Дайса ===')
pred_test = predict(model, x_test)
cm = confusion(pred_test, y_test)
iou, dice, acc = metrics_from_conf(cm)
for i, n in enumerate(CLASS_NAMES):
    print(f'Класс "{n}": IoU = {iou[i]:.3f}, Dice = {dice[i]:.3f}')
print(f'Средний IoU (mIoU): {iou.mean():.3f}')
print(f'Средний коэффициент Дайса: {dice.mean():.3f}')
print(f'Точность по пикселям: {acc:.3f}')
fg_true = y_test != 1
fg_pred = pred_test != 1
inter = (fg_true & fg_pred).sum(axis=(1, 2)).astype(float)
union = (fg_true | fg_pred).sum(axis=(1, 2)).astype(float)
sum_area = (fg_true.sum(axis=(1, 2)) + fg_pred.sum(axis=(1, 2))).astype(float)
img_iou = inter / np.maximum(union, 1)
img_dice = 2 * inter / np.maximum(sum_area, 1)
print(f'Для области "животное + граница" против фона: IoU = {inter.sum() / union.sum():.3f}, Dice = {2 * inter.sum() / sum_area.sum():.3f}')
print(f'Доля тестовых изображений с IoU области животного выше 0,9: {(img_iou > 0.9).mean() * 100:.1f}%')
print(f'Доля тестовых изображений с IoU области животного ниже 0,7: {(img_iou < 0.7).mean() * 100:.1f}%')

plt.figure(figsize=(7, 4))
plt.hist(img_iou, bins=20, color='#4c72b0')
plt.xlabel('IoU области животного на изображении')
plt.ylabel('Число изображений')
plt.title('Распределение IoU по тестовым изображениям')
plt.tight_layout()
plt.savefig(OUT / 'step6_iou_hist.png', dpi=110)
plt.close()

print('\n=== Шаг 7. Примеры: изображение, истинная маска, предсказание ===')
ids = [3, 120, 400, 900, 1500, 2300]
fig, axes = plt.subplots(len(ids), 3, figsize=(9, 3 * len(ids)))
for r, i in enumerate(ids):
    axes[r, 0].imshow(x_test[i])
    axes[r, 1].imshow(PALETTE[y_test[i]])
    axes[r, 2].imshow(PALETTE[pred_test[i]])
    axes[r, 0].set_ylabel(f'№ {i}')
    if r == 0:
        axes[r, 0].set_title('Исходное изображение')
        axes[r, 1].set_title('Настоящая маска')
        axes[r, 2].set_title('Предсказанная маска')
    for c in range(3):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    print(f'Изображение {i}: IoU области животного = {img_iou[i]:.3f}, Dice = {img_dice[i]:.3f}')
plt.tight_layout()
plt.savefig(OUT / 'step7_examples.png', dpi=110)
plt.close()

order = np.argsort(img_iou)
worst = order[:3]
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for k, i in enumerate(worst):
    axes[k, 0].imshow(x_test[i])
    axes[k, 1].imshow(PALETTE[y_test[i]])
    axes[k, 2].imshow(PALETTE[pred_test[i]])
    axes[k, 3].axis('off')
    axes[k, 3].text(0, 0.5, f'IoU = {img_iou[i]:.2f}\nDice = {img_dice[i]:.2f}\nизображение № {i}', fontsize=13)
    for c in range(3):
        axes[k, c].set_xticks([])
        axes[k, c].set_yticks([])
axes[0, 0].set_title('Исходное изображение')
axes[0, 1].set_title('Настоящая маска')
axes[0, 2].set_title('Предсказанная маска')
plt.suptitle('Самые неудачные случаи сегментации')
plt.tight_layout()
plt.savefig(OUT / 'step8_worst.png', dpi=110)
plt.close()
print('Наименьший IoU: ' + ', '.join(f'№ {i}: {img_iou[i]:.2f}' for i in worst))
print('\nГотово')
