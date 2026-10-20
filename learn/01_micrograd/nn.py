"""
Амика * стъпка 0 * част 2
Неврон, слой и мрежа (MLP), построени от Value.
"""
import random

from value import Value


class Neuron:
    def __init__(self, n_in):
        # по едно тегло за всеки вход плюс отместване (bias), случайни в [-1, 1]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # претеглена сума на входовете плюс bias, прекарана през tanh
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, n_outs):
        # n_outs е списък с размерите на слоевете, например [4, 4, 1]
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]