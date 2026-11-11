import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)


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

CLASSES = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
           'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']
BATCH = 64
EPOCHS = 10
LR = 0.001

print('=== Задание 1. Подготовка набора данных Fashion-MNIST ===')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
train_set = datasets.FashionMNIST(BASE / 'data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(BASE / 'data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
print(f'Обучающая выборка: {len(train_set)} изображений, тестовая выборка: {len(test_set)} изображений')
x, y = next(iter(train_loader))
print(f'Размер одного батча: {tuple(x.shape)}, метки: {tuple(y.shape)}')
print(f'После нормализации: мин = {x.min():.3f}, макс = {x.max():.3f}, среднее = {x.mean():.3f}')

fig, axes = plt.subplots(10, 6, figsize=(7, 12))
targets = train_set.targets.numpy()
for c in range(10):
    idx = np.where(targets == c)[0][:6]
    for j, i in enumerate(idx):
        axes[c, j].imshow(train_set.data[i], cmap='gray')
        axes[c, j].axis('off')
    axes[c, 0].set_title(CLASSES[c], fontsize=9, loc='left')
plt.tight_layout()
plt.savefig(OUT / 'classes.png', dpi=100)
plt.close()


class CNN(nn.Module):
    def __init__(self, filters=(16, 32), kernel=3, dropout=0.0):
        super().__init__()
        layers, cin = [], 1
        for f in filters:
            layers += [nn.Conv2d(cin, f, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool2d(2)]
            cin = f
        self.features = nn.Sequential(*layers)
        size = 28
        for _ in filters:
            size //= 2
        self.flatten = nn.Flatten()
        head = [nn.Linear(cin * size * size, 128), nn.ReLU()]
        if dropout:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128, 10))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


print('\n=== Задание 2. Модель CNN ===')
model = CNN()
print(model)
print('\nИзменение размера тензора после каждого слоя:')
t = torch.zeros(1, 1, 28, 28)
print(f'  {"Вход":<12} {tuple(t.shape)}')
for layer in list(model.features) + [model.flatten] + list(model.classifier):
    t = layer(t)
    print(f'  {layer.__class__.__name__:<12} {tuple(t.shape)}')

print('\n=== Задание 3. Параметры модели ===')
print('Функция потерь: CrossEntropyLoss')
print(f'Оптимизатор: Adam, скорость обучения: {LR}')
print(f'Размер батча: {BATCH}, количество эпох: {EPOCHS}')
print(f'Общее количество параметров модели: {count_params(model):,}')


def evaluate(m, loader):
    m.eval()
    correct = total = 0
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            p = m(xb).argmax(1)
            preds.append(p)
            correct += (p == yb).sum().item()
            total += len(yb)
    return correct / total, torch.cat(preds)


def train(m, opt, epochs=EPOCHS, verbose=True):
    loss_fn = nn.CrossEntropyLoss()
    hist = {'loss': [], 'train_acc': [], 'test_acc': [], 'batch_loss': [], 'time': []}
    for epoch in range(1, epochs + 1):
        m.train()
        t0, total_loss, correct, n = time.time(), 0.0, 0, 0
        for xb, yb in train_loader:
            out = m(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
            hist['batch_loss'].append(loss.item())
        test_acc, _ = evaluate(m, test_loader)
        hist['loss'].append(total_loss / n)
        hist['train_acc'].append(correct / n)
        hist['test_acc'].append(test_acc)
        hist['time'].append(time.time() - t0)
        if verbose:
            print(f'Эпоха {epoch:2d}/{epochs}: средняя потеря = {total_loss / n:.4f}, '
                  f'точность (обучение) = {correct / n:.4f}, точность (тест) = {test_acc:.4f}, '
                  f'время = {time.time() - t0:.1f} с')
    return hist


print('\n=== Задание 4. Обучение модели ===')
hist = train(model, torch.optim.Adam(model.parameters(), lr=LR))
print(f'Средняя потеря за эпоху (по всем эпохам): {np.mean(hist["loss"]):.4f}')
print(f'Точность на обучающей выборке после обучения: {hist["train_acc"][-1]:.4f}')
torch.save(model.state_dict(), OUT / 'cnn_fashion.pt')

print('\n=== Задание 5. Тестирование модели ===')
acc, preds = evaluate(model, test_loader)
labels = test_set.targets
print(f'Общая точность на тестовой выборке: {acc:.4f} ({acc * 100:.2f}%)')
print('Точность по классам:')
per_class = []
for c in range(10):
    mask = labels == c
    a = (preds[mask] == c).float().mean().item()
    per_class.append(a)
    print(f'  {CLASSES[c]:<10} {a * 100:.1f}%')

cm = np.zeros((10, 10), dtype=int)
for t_, p_ in zip(labels.numpy(), preds.numpy()):
    cm[t_, p_] += 1
pairs = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j]
pairs.sort(reverse=True)
print('Самые частые ошибки (истинный класс -> предсказанный):')
for k, a, b in pairs[:5]:
    print(f'  {a} -> {b}: {k} раз')

wrong = (preds != labels).nonzero().flatten()[:8]
right = (preds == labels).nonzero().flatten()[:8]
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for row, ids, name in [(0, right, 'Верно'), (1, wrong, 'Ошибка')]:
    for j, i in enumerate(ids):
        i = i.item()
        axes[row, j].imshow(test_set.data[i], cmap='gray')
        axes[row, j].set_title(f'{name}\nфакт: {CLASSES[labels[i]]}\nпрогноз: {CLASSES[preds[i]]}',
                               fontsize=9, color='green' if row == 0 else 'red')
        axes[row, j].axis('off')
plt.tight_layout()
plt.savefig(OUT / 'predictions.png', dpi=100)
plt.close()

print('\n=== Задание 6. Визуализация результатов ===')
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist['batch_loss'], alpha=0.3, label='по батчам')
steps = len(hist['batch_loss']) // EPOCHS
ax[0].plot([steps * (e + 1) for e in range(EPOCHS)], hist['loss'], 'o-', label='среднее за эпоху')
ax[0].set_title('Функция потерь во время обучения')
ax[0].set_xlabel('Итерация')
ax[0].legend()
ax[1].plot(range(1, EPOCHS + 1), hist['train_acc'], 'o-', label='обучающая')
ax[1].plot(range(1, EPOCHS + 1), hist['test_acc'], 's-', label='тестовая')
ax[1].set_title('Точность')
ax[1].set_xlabel('Эпоха')
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / 'training.png', dpi=100)
plt.close()

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap='Blues')
plt.xticks(range(10), CLASSES, rotation=45, ha='right')
plt.yticks(range(10), CLASSES)
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')
plt.tight_layout()
plt.savefig(OUT / 'confusion.png', dpi=100)
plt.close()
print('Графики сохранены: training.png, predictions.png, confusion.png')

print('\n=== Задание 7. Сравнение архитектур ===')
experiments = [
    ('Исходная модель', dict(), 'Adam', LR),
    ('Мод. 1: 3 сверточных слоя', dict(filters=(16, 32, 64)), 'Adam', LR),
    ('Мод. 2: больше фильтров', dict(filters=(32, 64)), 'Adam', LR),
    ('Мод. 3: ядро 5x5', dict(kernel=5), 'Adam', LR),
    ('Мод. 4: Dropout 0.5', dict(dropout=0.5), 'Adam', LR),
    ('Мод. 5: SGD', dict(), 'SGD', 0.01),
    ('Мод. 6: lr = 0.01', dict(), 'Adam', 0.01),
]
rows = []
for name, cfg, opt_name, lr in experiments:
    if name == 'Исходная модель':
        test_acc = hist['test_acc'][-1]
        m = model
    else:
        torch.manual_seed(0)
        m = CNN(**cfg)
        opt = (torch.optim.Adam(m.parameters(), lr=lr) if opt_name == 'Adam'
               else torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9))
        test_acc = train(m, opt, verbose=False)['test_acc'][-1]
    filters = cfg.get('filters', (16, 32))
    rows.append((name, len(filters), '/'.join(map(str, filters)), f'{cfg.get("kernel", 3)}x{cfg.get("kernel", 3)}',
                 f'{opt_name}, lr={lr}', f'{test_acc * 100:.2f}%', f'{count_params(m):,}'))
    print(f'{name:<28} слоев: {rows[-1][1]}, фильтры: {rows[-1][2]:<9} ядро: {rows[-1][3]}, '
          f'{rows[-1][4]:<15} точность теста: {rows[-1][5]}')

with open(OUT / 'comparison.csv', 'w', encoding='utf-8') as f:
    f.write('Модель;Сверточных слоев;Фильтры;Ядро;Оптимизатор;Точность теста;Параметров\n')
    for r in rows:
        f.write(';'.join(map(str, r)) + '\n')
print('\nГотово. Результаты сохранены в папке output')
