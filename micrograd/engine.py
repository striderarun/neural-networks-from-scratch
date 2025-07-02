# Data structure for maintaining expression graph of operators and values like a AST.
# Keep track of children values and operators involved in the expression computation.
# Wraps a single scalar value and it's gradient
import math

# A Scalar valued engine; In contrast to Pytorch which is built around Tensors (N dimensional arrays of scalars)
class Value:

    # Constructor
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # toString()
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # a + b => a.__add__(b)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Support expression Value(2.0) + 1
        out = Value(self.data + other.data, (self, other), '+')

        # Define the function that computes gradient of the children given the gradient of output (backwards propagation of gradients)
        # Consider e = d + c; out=e, d=self, other=c
        # Imagine final output L = f(e) and we have already computed dL/de or e.grad. Now given e.grad, how do we compute gradient for d and c
        # d.grad => dL/dd = dL/de * de/dd;
        # d.grad = e.grad * de/dd
        # Rewriting in terms of variables, since out = self + other
        # self.grad = out.grad * d(out)/d(self)
        # self.grad = out.grad * 1

        # other.grad = out.grad * d(out)/d(other)
        # other.grad = out.grad * 1
        # Basically, for addition, output gradient is copied to child's gradients
        def _backward():
            self.grad += out.grad * 1.0 # Gradients should be accumulated; When same variable is used more than once
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

     # Python will swap operands when 1 + Value(2.0) returns an error
    def __radd__(self, other):
        return self * other
    
    # a * b => a.__mul__(b)
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        # Given out.grad, compute gradients of children
        # out = self * other; let the final output be L = f(out)
        # dL/d(self) = dL/d(out) * d(out)/d(self)
        # self.grad = out.grad * other

        # dL/d(other) = dL/d(out) * d(out)/d(other)
        # other.grad = out.grad * self
        def _backward():
            self.grad += out.grad * other.data # Gradients should be accumulated; When same variable is used more than once
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    # Python will swap operands when 1 * Value(2.0) returns an error
    def __rmul__(self, other):
        return self * other

    # https://en.wikipedia.org/wiki/Hyperbolic_functions
    # tanh x = (e**2x-1)/(e**2x+1)
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        # out = tanh(self.data)
        # self.grad = out.grad * d(out)/d(self)
        # self.grad = out.grad * (1 - tanh**2 self.data) (Derivate of tanh: https://en.wikipedia.org/wiki/Hyperbolic_functions)
        def _backward():
            self.grad += out.grad * (1 - t**2) # Gradients should be accumulated; When same variable is used more than once
        out._backward = _backward # Assigning a function to a variable, don't call the function here : self._backward = _backward()
        return out
    
    # Implement e**x
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        # out = e**self.data
        # self.grad = out.grad * d(out)/d(e**self)
        # self.grad = out.grad * out.data since d/dx(ex) = ex
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    # other is a scalar here, unlike other operations where other was a Value
    # Value(2.0) ** 3
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self, ), f'**{other}')

        # Power rule
        # d/dx(x**n) = n * x**n-1
        def _backward():
            self.grad += out.grad * (other * (self.data ** (other-1)))
        out._backward = _backward
        return out

    # Implement division
    # a/b = a * b**-1
    # Relies on the power operation x**y
    def __truediv__(self, other):
        return self * other**-1
    
    # Operation for Negation. -Value(2.0)
    # Directly multiply by -1
    def __neg__(self):
        return self.data * -1
    
    # Implement subtraction
    # a - b = a + (-b)
    # Ends up calling add operation and the negation operation
    def __sub__(self, other):
        return self + (-other)
    
    # We want to call _backward() on every node starting from the output and going backwards, refer manual_backpropagation
    # First topologically sort the graph, then call _backward() on the reversed topological order (backwards from output)
    def backward(self):
        topo = []
        visited = set()
        def topological_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_order(child)
                topo.append(v)
        
        topological_order(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()




    
   