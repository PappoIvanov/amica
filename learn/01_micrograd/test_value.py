"""
Проверка на value.py: сравняваме градиента от backward()
с числена производна: (f(x+h) - f(x-h)) / 2h.

Пускане от папката на проекта:
    py learn/01_micrograd/test_value.py
"""
import sys

from value import Value

# Конзолата на Windows по подразбиране не е UTF-8 и Python гърми при печат на кирилица.
# Този ред оправя изхода само за този скрипт. Трайно решение: променлива PYTHONUTF8=1.
sys.stdout.reconfigure(encoding="utf-8")


def f(a, b, c):
    # произволна функция, която минава през всички операции, които написахме
    return ((a * b + c) ** 2 / (1 + c.exp())).tanh()


x0 = [0.7, -1.3, 0.4]

a, b, c = (Value(v) for v in x0)
out = f(a, b, c)
out.backward()
analytic = [a.grad, b.grad, c.grad]


def numeric(i, h=1e-6):
    plus = list(x0)
    plus[i] += h
    minus = list(x0)
    minus[i] -= h
    return (f(*map(Value, plus)).data - f(*map(Value, minus)).data) / (2 * h)


numeric_grads = [numeric(i) for i in range(3)]

print(f"f(a, b, c) = {out.data:.6f}")
for name, g1, g2 in zip("abc", analytic, numeric_grads):
    ok = abs(g1 - g2) < 1e-5
    status = "OK" if ok else "ГРЕШКА"
    print(f"d/d{name}: backward={g1:+.6f}  числено={g2:+.6f}  {status}")
