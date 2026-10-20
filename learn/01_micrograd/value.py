"""
Амика * стъпка 0 * micrograd
============================
Най-малкият възможен двигател за автоматично диференциране.
PyTorch прави същото с тензори. Тук го правим за едно число,
за да се види механизмът.

Идея: всяко число в изчислението е обект Value, което помни:
   - стойността си: data
   - от кои Value е получено и с каква операция: _prev, _op
   - производната на крайния резултат спрямо него: grad
     (с колко се променя резултатът, ако това число мръдне мъничко)

Обучение на модел = смятаме grad за всеки параметър и го бутаме
малко в посоката, която намалява грешката. grad идва от backward().
"""
import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0                # d(резултат)/d(това). Нула до извикване на backward().
        self._backward = lambda: None   # как този възел подава grad към децата си
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---- операции ---------------------------------------------------------------
    # Всяка операция прави нов Value и му закача функция _backward,
    # която знае местната производна на операцията (верижно правило).

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1 и d(a+b)/db = 1: градиентът минава непроменен.
            # += вместо =, защото едно Value може да участва на няколко места
            # в израза и приносите се събират.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b и d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "степента трябва да е обикновено число"
        out = Value(self.data ** n, (self,), f"**{n}")

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += n * self.data ** (n - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(e^a)/da = e^a, а това е точно out.data
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh a)/da = 1 - tanh(a)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # ---- удобства: за да работят -x, x - y, x / y, 2 * x, 2 + x ... --------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return other * self ** -1

    # ---- backpropagation ---------------------------------------------------------
    def backward(self):
        # 1. Подреждаме възлите така, че всеки да е след децата си
        # (топологична подредба на графа на изчислението).
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # 2. Производната на резултата спрямо самия себе си е 1.
        self.grad = 1.0

        # 3. Отзад напред: всеки възел раздава своя grad на децата си.
        for v in reversed(topo):
            v._backward()