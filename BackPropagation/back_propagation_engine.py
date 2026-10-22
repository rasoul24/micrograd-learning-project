import numpy as np

class Value:

    def __init__(self, data, _children = (), _op = ''):

        self.data = data
        self.childre = _children

        self._backward = lambda: None
        self._prev = set(_children)

        self.grad = 0.0
        self._op = _op

    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op = "+")

        def _backward():
            """
            Backward function of the sum operator.
            """
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op = "x")

        def _backward():
            """
            Backward function of the multiplication operator.
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def exp(self):

        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward

        return out

    def log(self):
        eps = 1e-16

        assert self.data > 0, f"log reçu avec x={self.data}"
        out = Value(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1/(out.data+eps))*out.grad
        out._backward = _backward

        return out



    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):

        return f"Value(data={self.data})"


if __name__ == "__main__":

    a = Value(6)
    b = Value(7)

    c = a + b
    d = Value(5)

    e = c*d

    e = -e

    e = -d/2

    e = d - 25

    e.backward()

    print(e)
    print('de/de = ', e.grad)

    print("de/dc = ", c.grad)
    print("de/dd = ", d.grad)

