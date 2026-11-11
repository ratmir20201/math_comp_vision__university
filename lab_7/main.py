import sys
import tarfile
import time
import urllib.request
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
DATA = BASE / 'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SIZE = 128
EPOCHS = 10
BATCH = 32
LR = 1e-3

URLS = {
    'images.tar.gz': 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz',
    'annotations.tar.gz': 'https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz',
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = Tee(sys.__stdout__, open(OUT / 'console.txt', 'w', encoding='utf-8'))
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(max(1, torch.get_num_threads()))


def download_and_extract():
    for name, url in URLS.items():
        folder = DATA / name.split('.')[0]
        if folder.exists():
            continue
        arc = DATA / name
        if not arc.exists():
            print(f'Скачивание {name} ...')
            urllib.request.urlretrieve(url, arc)
        print(f'Распаковка {name} ...')
        with tarfile.open(arc) as t:
            t.extractall(DATA, filter='data')


def load_split(split):
    cache = DATA / f'{split}_{SIZE}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['x'], z['y']
    download_and_extract()
    names = [l.split()[0] for l in (DATA / 'annotations' / f'{split}.txt').read_text().splitlines() if l.strip()]
    xs, ys = [], []
    for n in names:
        img = Image.open(DATA / 'images' / f'{n}.jpg').convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
        tri = Image.open(DATA / 'annotations' / 'trimaps' / f'{n}.png').resize((SIZE, SIZE), Image.NEAREST)
        tri = np.asarray(tri)
        mask = (tri != 2).astype(np.uint8)
        xs.append(np.asarray(img, dtype=np.uint8))
        ys.append(mask)
    x, y = np.stack(xs), np.stack(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


print('=== Шаг 1. Загрузка набора данных Oxford-IIIT Pet ===')
x_train, y_train = load_split('trainval')
x_test, y_test = load_split('test')
print(f'Загружено изображений: {len(x_train) + len(x_test)}')

print('\n=== Шаг 2. Приведение изображений и масок к одному размеру ===')
print(f'Размер изображения: {x_train.shape[1:]}, размер маски: {y_train.shape[1:]}')
print(f'Классы: 0 - фон, 1 - животное. Доля пикселей животного: {y_train.mean() * 100:.1f}%')

print('\n=== Шаг 3. Разделение на обучающую и тестовую выборки ===')
print(f'Обучающая выборка: {len(x_train)}, тестовая выборка: {len(x_test)}')


def to_tensor(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.mid = block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


print('\n=== Шаг 4. Выбор архитектуры: U-Net ===')
model = UNet().to(device)
print(model)
print(f'Число параметров: {sum(p.numel() for p in model.parameters()):,}')
print(f'Устройство: {device}')


@torch.no_grad()
def predict(x):
    model.eval()
    res = []
    for i in range(0, len(x), 64):
        p = torch.sigmoid(model(to_tensor(x[i:i + 64]).to(device)))
        res.append((p[:, 0] > 0.5).cpu().numpy().astype(np.uint8))
    return np.concatenate(res)


def iou_dice(pred, true):
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    s = pred.sum() + true.sum()
    return inter / max(union, 1), 2 * inter / max(s, 1)


print('\n=== Шаг 5. Обучение модели ===')
opt = torch.optim.Adam(model.parameters(), lr=LR)
xt, yt = to_tensor(x_train), torch.from_numpy(y_train).float().unsqueeze(1)
hist_loss, hist_iou = [], []
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, total = time.time(), 0.0
    perm = torch.randperm(len(xt))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = xt[idx], yt[idx]
        if torch.rand(1).item() < 0.5:
            xb, yb = xb.flip(3), yb.flip(3)
        xb, yb = xb.to(device), yb.to(device)
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    val_iou, _ = iou_dice(predict(x_test[:500]), y_test[:500])
    hist_loss.append(total / len(xt))
    hist_iou.append(val_iou)
    print(f'Эпоха {epoch:2d}/{EPOCHS}: loss = {hist_loss[-1]:.4f}, IoU (тест) = {val_iou:.3f}, время {time.time() - t0:.0f} с')
torch.save(model.state_dict(), OUT / 'unet_pets.pt')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, EPOCHS + 1), hist_loss, marker='o')
ax[0].set_title('Функция потерь (обучение)')
ax[0].set_xlabel('Эпоха')
ax[1].plot(range(1, EPOCHS + 1), hist_iou, marker='o', color='green')
ax[1].set_title('IoU на тестовой выборке')
ax[1].set_xlabel('Эпоха')
plt.tight_layout()
plt.savefig(OUT / 'training.png', dpi=110)
plt.close()

print('\n=== Шаг 6. Оценка качества модели ===')
pred_test = predict(x_test)
iou_pet, dice_pet = iou_dice(pred_test, y_test)
iou_bg, dice_bg = iou_dice(1 - pred_test, 1 - y_test)
acc = (pred_test == y_test).mean()
print(f'Класс "животное": IoU = {iou_pet:.3f}, Dice = {dice_pet:.3f}')
print(f'Класс "фон":      IoU = {iou_bg:.3f}, Dice = {dice_bg:.3f}')
print(f'Средний IoU (mIoU) = {(iou_pet + iou_bg) / 2:.3f}, средний Dice = {(dice_pet + dice_bg) / 2:.3f}')
print(f'Точность по пикселям = {acc:.3f}')

print('\n=== Шаг 7. Примеры сегментации ===')
ids = [5, 250, 700, 1200, 1900, 3000]
fig, axes = plt.subplots(len(ids), 3, figsize=(9, 3 * len(ids)))
for r, i in enumerate(ids):
    iou_i, dice_i = iou_dice(pred_test[i], y_test[i])
    print(f'Изображение № {i}: IoU = {iou_i:.3f}, Dice = {dice_i:.3f}')
    for c, (img, title) in enumerate([(x_test[i], 'Исходное изображение'),
                                      (y_test[i], 'Настоящая маска'),
                                      (pred_test[i], f'Предсказание (IoU={iou_i:.2f})')]):
        axes[r, c].imshow(img, cmap='gray')
        axes[r, c].set_title(title)
        axes[r, c].axis('off')
plt.tight_layout()
plt.savefig(OUT / 'examples.png', dpi=100)
plt.close()
print('\nГотово. Результаты сохранены в папке output')
