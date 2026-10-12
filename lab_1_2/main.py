import numpy as np

v = np.array([2, 4, 6, 8, 10], dtype=float)

print("Вектор:", v)
print("Форма:", v.shape)
print("Размер:", v.size)
print("Сумма:", v.sum())
print("Среднее:", v.mean())
print("v[2] =", v[2])
print("v[-1] =", v[-1])
print("Наибольший:", v.max(), "Наименьший:", v.min())

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("2a =", 2 * a)
print("Скалярное произведение:", np.dot(a, b))
print("Сумма элементов a * b:", (a * b).sum())

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(A)
print("A[1, 2] =", A[1, 2])
print("Первая строка:", A[0, :])
print("Второй столбец:", A[:, 1])
print("Раздел 2×2:\n", A[:2, :2])

B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

print("A + B:\n", A + B)
print("A - B:\n", A - B)
print("A * B:\n", A * B)
print("A / 2:\n", A / 2)
print("A²:\n", A**2)

M = np.array([[1, 2], [3, 4]])
N = np.array([[5, 6], [7, 8]])

print("M * N:\n", M * N)
print("M @ N:\n", M @ N)
print("N @ M:\n", N @ M)
print("M @ N == N @ M:", np.array_equal(M @ N, N @ M))

C = np.array([[3, 7, 2], [8, 1, 6]])

print("C.T:\n", C.T)
print("Сумма всех элементов:", C.sum())
print("Сумма строк:", C.sum(axis=1))
print("Среднее значение столбцов:", C.mean(axis=0))
print("Наибольший элемент:", C.max())
print("Сумма по столбцам (axis=0):", C.sum(axis=0))

D = np.array([[4.0, 7.0], [2.0, 6.0]])

det_D = np.linalg.det(D)
inv_D = np.linalg.inv(D)

print("det(D) =", det_D)
print("D⁻¹:\n", inv_D)
print("Check D @ D⁻¹:\n", D @ inv_D)
print("Округлённо:\n", np.round(D @ inv_D, 10))

A_sys = np.array([[2.0, 1.0], [1.0, -1.0]])
b_sys = np.array([7.0, 2.0])

x = np.linalg.solve(A_sys, b_sys)
print("Solution [x, y] =", x)
print("Проверка A @ x =", A_sys @ x)

P = np.array([[2, 1, 3], [0, 4, 5], [1, 2, 2]], dtype=float)
Q = np.array([[1, 0, 2], [3, 1, 1], [2, 4, 0]], dtype=float)

print("P + Q:\n", P + Q)
print("P - Q:\n", P - Q)
print("P * Q:\n", P * Q)
print("P @ Q:\n", P @ Q)
print("Q @ P:\n", Q @ P)
print("P @ Q == Q @ P:", np.array_equal(P @ Q, Q @ P))
print("P.T:\n", P.T)

det_P = np.linalg.det(P)
print("det(P) =", det_P)

if abs(det_P) > 1e-12:
    inv_P = np.linalg.inv(P)
    print("P⁻¹:\n", inv_P)
    print("P @ P⁻¹:\n", np.round(P @ inv_P, 10))
