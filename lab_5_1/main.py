import sys
from pathlib import Path

Path('output').mkdir(exist_ok=True)


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


sys.stdout = Tee(sys.stdout, open('output/console.txt', 'w', encoding='utf-8'))

import numpy as np
import torch
import matplotlib.pyplot as plt


def info(name, t):
    print(f'{name}: shape={tuple(t.shape)}, ndim={t.ndim}, numel={t.numel()}, dtype={t.dtype}, device={t.device}, requires_grad={t.requires_grad}')


print('=== Задание 1. Версия, seed, устройство ===')
print('Версия PyTorch:', torch.__version__)
torch.manual_seed(42)
print('CUDA доступна:', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Устройство:', device)

print('\n=== Задание 2. Создание тензоров ===')
scalar = torch.tensor(3.14)
vector = torch.arange(0, 10, 2)
lin = torch.linspace(0, 1, 5)
matrix_zeros = torch.zeros(3, 4)
matrix_ones = torch.ones(2, 3)
rand3d = torch.rand(2, 3, 4)
for name, t in (('Скаляр torch.tensor', scalar), ('Вектор torch.arange', vector), ('Вектор torch.linspace', lin),
                ('Матрица torch.zeros', matrix_zeros), ('Матрица torch.ones', matrix_ones), ('3D torch.rand', rand3d)):
    print(t)
    info(name, t)

print('\n=== Задание 3. Индексация и маски ===')
x = torch.randn(4, 5)
print('Тензор 4x5:\n', x)
print('Первая строка:', x[0])
print('Последний столбец:', x[:, -1])
print('Центр x[1:3, 1:4]:\n', x[1:3, 1:4])
mask = x > 0
print('Маска x > 0:\n', mask)
print('Положительные элементы:', x[mask])
print('Количество положительных:', int(mask.sum()))
copy = x.clone()
copy[copy < 0] = 0
print('Копия с отрицательными, заменёнными нулём:\n', copy)
print('Память копии отличается от оригинала:', copy.data_ptr() != x.data_ptr())
print('Отрицательные в оригинале остались:', bool((x < 0).any()))

print('\n=== Задание 4. Изменение формы ===')
t = torch.arange(24).reshape(2, 3, 4)
print('Исходный', tuple(t.shape))
r = t.reshape(4, 6)
print('reshape(4, 6):', tuple(r.shape))
v = t.view(6, 4)
print('view(6, 4):', tuple(v.shape))
f = t.flatten()
print('flatten():', tuple(f.shape))
tr = t.transpose(0, 1)
print('transpose(0, 1):', tuple(tr.shape), '| contiguous:', tr.is_contiguous())
try:
    tr.view(-1)
except RuntimeError as e:
    print('view после transpose вызывает ошибку:', str(e)[:70])
print('reshape после transpose работает:', tuple(tr.reshape(-1).shape))
p = t.permute(2, 0, 1)
print('permute(2, 0, 1):', tuple(p.shape))
u = t.unsqueeze(0)
print('unsqueeze(0):', tuple(u.shape))
s = u.squeeze()
print('squeeze():', tuple(s.shape))

print('\n=== Задание 5. Broadcasting и статистики ===')
A = torch.arange(12.).reshape(3, 4)
vec = torch.tensor([10., 20., 30., 40.])
c = torch.tensor([[100.], [200.], [300.]])
B = A + vec + c
print('A:\n', A)
print('B = A + v + c:\n', B)
info('B', B)
print('sum по строкам (dim=1):', B.sum(dim=1), '| по столбцам (dim=0):', B.sum(dim=0))
print('mean по строкам:', B.mean(dim=1), '| по столбцам:', B.mean(dim=0))
print('max по строкам:', B.max(dim=1).values, '| по столбцам:', B.max(dim=0).values)
print('std по строкам:', B.std(dim=1), '| по столбцам:', B.std(dim=0))

print('\n=== Задание 6. Умножение матриц и операции ===')
M1 = torch.arange(12.).reshape(3, 4)
M2 = torch.arange(8.).reshape(4, 2)
r1 = M1 @ M2
r2 = torch.matmul(M1, M2)
print('M1 @ M2:\n', r1)
print('allclose(@, matmul):', torch.allclose(r1, r2))
E1 = torch.arange(1., 7.).reshape(2, 3)
E2 = torch.full((2, 3), 2.)
print('Поэлементное умножение:\n', E1 * E2)
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print('torch.dot(a, b):', torch.dot(a, b))
print('torch.cat по dim=0:', tuple(torch.cat([E1, E2], dim=0).shape), '| по dim=1:', tuple(torch.cat([E1, E2], dim=1).shape))
print('torch.stack:', tuple(torch.stack([E1, E2]).shape), '| stack dim=1:', tuple(torch.stack([E1, E2], dim=1).shape))
parts = torch.split(torch.arange(10), [3, 3, 4])
print('torch.split:', [tuple(p.shape) for p in parts], [p.tolist() for p in parts])

print('\n=== Задание 7. NumPy и тензоры ===')
arr = np.arange(6, dtype=np.float32)
shared = torch.from_numpy(arr)
print('Исходно: arr =', arr, '| shared =', shared)
arr[0] = 100
print('После arr[0] = 100: shared =', shared)
shared[1] = -1
print('После shared[1] = -1: arr =', arr)
indep = shared.clone()
arr[2] = 555
print('После arr[2] = 555: shared[2] =', shared[2].item(), '| clone[2] =', indep[2].item())
on_device = indep.to(device)
print('Устройство после .to(device):', on_device.device)
back = on_device.detach().cpu().numpy()
print('Обратно в NumPy:', type(back), back.dtype, back)

print('\n=== Задание 8. Градиент (3x^2 + 2x - 5) ===')
x = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=True)
loss = (3 * x ** 2 + 2 * x - 5).sum()
loss.backward()
print('loss =', loss.item())
print('x.grad =', x.grad)
analytic = 6 * x.detach() + 2
print('Аналитически 6x + 2 =', analytic)
print('allclose:', torch.allclose(x.grad, analytic))

print('\n=== Задание 9. Градиенты функции двух переменных ===')
x = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
y = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
u = x * y + torch.sin(x)
loss = u ** 2 + torch.exp(y)
loss.backward()
print('u =', u.item(), '| loss =', loss.item())
print('autograd: dx =', x.grad.item(), '| dy =', y.grad.item())
dx = 2 * u.item() * (2.0 + np.cos(1.0))
dy = 2 * u.item() * 1.0 + np.exp(2.0)
print('вручную: dx =', dx, '| dy =', dy)
print('Совпадение:', np.isclose(x.grad.item(), dx), np.isclose(y.grad.item(), dy))

print('\n=== Задание 10. Листовые и нелистовые тензоры ===')
x = torch.tensor([1., 2., 3.], dtype=torch.float32, requires_grad=True)
h = 2 * x
h.retain_grad()
loss = (h ** 2).sum()
print('x: is_leaf =', x.is_leaf, '| requires_grad =', x.requires_grad, '| grad_fn =', x.grad_fn)
print('h: is_leaf =', h.is_leaf, '| requires_grad =', h.requires_grad, '| grad_fn =', h.grad_fn)
loss.backward()
print('x.grad =', x.grad, '| h.grad =', h.grad)
print('x.grad = 2 * h.grad:', torch.allclose(x.grad, 2 * h.grad))

print('\n=== Задание 11. Накопление градиентов, detach и no_grad ===')
x = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
(x ** 2).backward()
print('После backward x**2: x.grad =', x.grad.item())
(3 * x).backward()
print('После backward 3*x: x.grad =', x.grad.item(), '(накопление 4 + 3)')
x.grad.zero_()
print('После zero_(): x.grad =', x.grad.item())
(x ** 2).backward()
x.grad = None
print('После x.grad = None: x.grad =', x.grad)
z = x.detach()
print('detach: requires_grad =', z.requires_grad, '| общая память:', z.data_ptr() == x.data_ptr())
with torch.no_grad():
    w = x * 2
print('no_grad: requires_grad результата =', w.requires_grad, '| grad_fn =', w.grad_fn)
w2 = x * 2
print('без no_grad: requires_grad результата =', w2.requires_grad, '| grad_fn =', w2.grad_fn)

print('\n=== Задание 12. Линейная регрессия на PyTorch ===')
X = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
Y = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float32)
torch.manual_seed(42)
w = torch.randn(1, dtype=torch.float32, requires_grad=True)
b = torch.randn(1, dtype=torch.float32, requires_grad=True)
print('Начальные w, b:', w.item(), b.item())
lr = 0.01
losses = []
for i in range(100):
    y_hat = w * X + b
    loss = ((y_hat - Y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    losses.append(loss.item())
    if i % 10 == 0 or i == 99:
        print('Итерация', i + 1, ': loss =', round(loss.item(), 4), '| w =', round(w.item(), 4), '| b =', round(b.item(), 4))
print('Итог: w =', round(w.item(), 4), ', b =', round(b.item(), 4), '(истинные: w = 2, b = 1)')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(losses)
ax[0].set_title('Функция потерь MSE')
ax[0].set_xlabel('Итерация')
ax[0].set_ylabel('MSE')
ax[1].scatter(X.numpy(), Y.numpy(), label='Данные')
ax[1].plot(X.numpy(), (w * X + b).detach().numpy(), color='red', label='Модель')
ax[1].set_title('Результат обучения')
ax[1].legend()
plt.tight_layout()
plt.savefig('output/task12_regression.png', dpi=130)
plt.close()

print('\n=== Вариант 1 ===')
A = torch.arange(1, 13).reshape(3, 4)
print('A:\n', A)
info('A', A)
print('Среднее значение строк:', A.float().mean(dim=1))
print('A.T:\n', A.T)
info('A.T', A.T)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (2 * x ** 2 + 3 * x).sum()
loss.backward()
print('loss =', loss.item())
print('x.grad =', x.grad)
analytic = 4 * x.detach() + 3
print('Аналитически 4x + 3 =', analytic)
print('allclose:', torch.allclose(x.grad, analytic))
