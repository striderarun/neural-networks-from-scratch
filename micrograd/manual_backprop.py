import visualizer.graph as draw
import torch
from engine import Value

# Manually calculate gradient by calling backward in correct order
def manual_backpropagation():
    # Inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # Weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # Bias of the neuron
    b = Value(6.8813735, label='b')

    # n = x1w1 + x2w2 + b
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'

    # o = tanh(n)
    # Forward pass output
    o = n.tanh(); o.label = 'o'

    # Manual Backpropagation 
    # Calculate derivatives at each node by calling backward()
    # Initialize output gradient to 1
    o.grad = 1.0

    # Gradient of o's child n
    o._backward()

    # Gradients of n's children, x1w1x2w2 and b
    n._backward()

    # Gradient of x1w1 and x2w2
    x1w1x2w2._backward()

    # Gradients of other nodes
    b._backward()
    x1w1._backward()
    x2w2._backward()
    x1._backward()
    x2._backward()
    w1._backward()
    w2._backward()

    # Check the gradient values
    draw.draw_dot(o).view()

def illustrate_backprop_bug():
    # Initially, we were doing self.grad = val instead of self.grad += val.
    # When we use same variable more than once, the gradient values are overwritten instead of being added.
    # Gradients at a node should be accumulated.

    # a = Value(3.0, label='a')
    # b = a + a
    # b.backward()

    # draw.draw_dot(b).view()

    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    d = a * b; d.label = 'd'
    e = a + b; d.label = 'e'
    f = d * e; d.label = 'f'
    f.backward()

    draw.draw_dot(f).view()

def auto_backpropagation():
    # Inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # Weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # Bias of the neuron
    b = Value(6.8813735, label='b')

    # n = x1w1 + x2w2 + b
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'

    # o = tanh(n)
    # Forward pass output
    # o = n.tanh(); o.label = 'o'

    # Breakup tanh into (e**2x - 1)/(e**2x + 1)
    e = (2 * n).exp()
    o = (e - 1)/(e + 1); o.label = 'o'
    o.backward()

    # Check the gradient values
    draw.draw_dot(o).view()

def compute_using_pytorch():
    # We have implemented a scalar valued grad engine which operates on a single scalar value for all operations
    # In Pytorch, everything is a Tensor which is an N dimensionsal array of scalar values

    # Create the same expression graph using pytorch
    # Create x1 as a Tensor with a single scalar value. By default Pytorch creates the tensor as a float32 dtype
    # Casting to double() to tell Pytorch to use float64 instead, in line with vanilla Python that we used in our engine
    # By default, pytorch does not compute the gradient for leaf nodes like x1,x2,w1,w2 etc. Instruct it to compute gradient specifically.
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
    b = torch.Tensor([6.8813735]).double(); b.requires_grad = True

    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)

    # Extract the value of o from it's tensor and print
    print(o.data.item())
    print(o.item()) # Same

    # Kickoff back propagation
    o.backward()

    # Once backward is called, each Tensor will have a 'grad' attribute with the computed gradient. This is also a Tensor.
    # Extract the gradient from the single valued grad tensor.
    print('--Gradients at each node--')
    print('x1', x1.grad.item(), 'shape', x1.grad.shape)
    print('x2', x2.grad.item(), 'shape', x2.grad.shape)
    print('w1', w1.grad.item(), 'shape', w1.grad.shape)
    print('w2', w2.grad.item(), 'shape', w2.grad.shape)

    # Creating a Tensor of 2X3
    tensor_x = torch.Tensor([[1,2,3],[4,5,6]])
    print(tensor_x.grad) # None since no grad computed
    print(tensor_x.shape)
    print(tensor_x.dtype)
    print(tensor_x.data)


if __name__ == '__main__':
    # manual_backpropagation()
    # auto_backpropagation()
    illustrate_backprop_bug()

    # Support expressions Value(val) + 1
    # a = Value(4.0)
    # b = Value(3.0)
    # print(1+a)
    # print(a+1)
    # print(a*3)
    # print(a/b)

    # compute_using_pytorch()