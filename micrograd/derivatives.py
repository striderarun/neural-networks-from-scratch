import math
import numpy as np
import matplotlib.pyplot as plt
from engine import Value
import visualizer.graph as draw

# A quadratic scalar valued function with one input
def f(x):
    return 3*x**2 - 4*x + 5

def computing_derivative_univariate_calculus():
    # Invoking function f with a scalar value
    result = f(3.0)

    # Generate array of scalar values from [-5, 5) in steps of 0.5
    xs = np.arange(-5, 5, 0.25)

    # Invoke function f with array of scalar values
    ys = f(xs)

    # Plot function f for scalar value array
    plt.plot(xs, ys)
    # A parabolic function
    # plt.show()

    # Derivative of a function answers the question:
    # If you slightly bump up the value of x at any point by a small value h, how does the function f(x) respond and with what sensitivity? Does f go up or go down and by how much?
    # Does f go up or go down and by how much == the slope of the function at that point. (The rise over the run)
    # Prime notation: f'(x) = (f(x+h) - f(x))/h
    # Leibniz notation: dy/dx = (f(x+h) - f(x))/h or dy/dx = d/dx(f(x))
    # Other definitions: 
    # Derivative is the rate of change of a function with respect to a variable.
    # Derivative quantifies the sensitivity to change of a function's output with respect to its input.
    # Derivative of a function of a single variable at a chosen input value, when it exists, is the slope of the tangent line to the graph of the function at that point.
    # Derivative is the instantaneous rate of change, the ratio of the instantaneous change in the dependent variable to that of the independent variable.
    # Derivative is also known as gradient.
    # Gradient = 1 means output changes at same rate as input.
    # Gradient = 0 means input has no effect at all on the output.

    # Calculating the derivative of f (Single Variable Calculus)
    h = 0.001
    x = 3.0
    # Applying power rule to f, we get derivative = 6*x - 4. Putting in x=3, we get derivative = 14
    # Derivative computed below using slope definition also converges to 14 for smaller and smaller values of h.
    derivative_or_slope = (f(x + h) - f(x))/h
    print(derivative_or_slope)

def computing_derivative_multivariate_calculus():
    # Function with three input variables
    a = 2.0
    b = -3.0
    c = 10.0
    d = a*b + c

    # Computing derivative of d with respect to a
    # Using differentiation, expected dd/da = b
    h = 0.001
    d1 = a*b + c
    a += h
    d2 = a*b + c 
    
    print('d1', d1)
    print('d2', d2)
    # This will be equal to -3.0 as indicated mathematically above
    # Slope=-3; Implies d will decrease 3 times as fast as a.
    print('slope', (d2 - d1)/h)

    # Computing derivative of d with respect to b
    # Using differentiation, expected dd/db = a
    d1 = a*b + c
    b += h
    d2 = a*b + c 

    print('d1', d1)
    print('d2', d2)
    # This will be equal to 2.0 as indicated mathematically above
    # Slope=2; Implies d will increase twice as fast as b.
    print('slope', (d2 - d1)/h)

    # Similarly, expected dd/dc = 1
    # Slope=1; Implies, d will increase at same rate as c.

    # Here, d = a*b + c is an expression with three parameters
    # Neural networks are composed of similar expressions (using vectors/tensors instead of scalars) with massive number of parameters 

def back_propagation():
    # Forward pass or propagation 
    # From the four input variables, we compute layers of intermediate expressions (d, e) which are used to compute final result L (represents Loss)
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    f = Value(-2.0, label='f')
    d = a*b; d.label = 'd'
    e = d + c; e.label='e'
    L = e * f; L.label='L'

    # End goal: We want to find out how to tweak the values of a,b,c,d,e,f so that the final output L is lowest or minimum.
    # For this, first we have to know what is the impact of changing these variables on the output L.
    # This is done by backpropagation. Start from the result or loss function and reverse. Compute the gradient at each node.
    # For each node, we are going to calculate the derivative of L with respect to that node. i.e dL/dL, dL/df, dL/de, dL/dd, dL/dc, dL/db, dL/da
    # In a neural network setting, we are interested in how the weights affect or impact the resultant loss L. We are not interested in derivative of L with respect to inputs a,b,c,f which are fixed.
    # Now based on the above function definitions, we can directly calculate the following local derivatives using differential calculus:
        # dL/de = f
        # dL/df = e
        # de/dd = 1
        # de/dc = 1
        # dd/da = b
        # dd/db = a
    # The local derivative at a point in the network with respect to it's inputs does not depend on the rest of the weights in the network. It just depends on that node's inputs.
    # Thus, derivative of d with respect to a, dd/da is a local derivative.
    # So, we already know how changing e and f impact the result L (defined by dL/de and dL/df)
    # Now, we need to calculate dL/dd, dL/dc, dL/db, dL/da to understand how changing d,c,b and a affect L.
    # d, c, b and a do not directly affect L, they indirectly affect L through the intermediate expressions.
    # We can calculate these derivatives using the chain rule in differential calculus.
    # dL/dd = dL/de * de/dd 
    # dL/dc = dL/de * de/dc
    # dL/db = dL/de * de/dd * dd/da
    # dL/da = dL/de * de/dd * dd/db

    # Chain Rule: dz/dx = dz/dy * dy/dx
    # Knowing the instantaneous rate of change of z relative to y and that of y relative to x allows one to calculate the instantaneous rate of change of z relative to x as the product of the two rates of change.
    # If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 × 4 = 8 times as fast as the man.

    # Thus using backprop, we can compute the gradient of L with respect to every weight in the network by applying the chain rule on local derivatives.
    # Knowing the gradient of L with respect to the weights gives us the power to control the end result by varying the values of the weights in the network.

def simple_manual_nn_calculation_single_pass():
    # Inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # Weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # Bias of the neuron
    b = Value(6.8813735, label='b')

    # Weighted sums
    n = x1*w1 + x2*w2 + b
    n.label = 'n'

    # Final output after passing weighted sum to activation function
    o = n.tanh(); o.label = 'o'

    # Backpropagation calculations
    # Start from output
    # o.grad = do/do; do/do = 1
    o.grad = 1

    # Derivative of tanh x = 1 - tanh x ** 2
    # do/dn = 1 - tanh n ** 2 
    # n.grad = do/dn
    n.grad = 1 - o.data ** 2

    # b.grad = do/db
    # do/db = do/dn * dn/db
    b.grad = n.grad * 1

    # w1.grad = do/dw1
    # do/dw1 = do/dn * dn/dw1 
    w1.grad = n.grad * x1.data

    # w2.grad = do/dw2
    # do/dw2 = do/dn * dn/dw2
    # Intuitively, gradient of w2 will be 0 because x2 is 0. Which means weight w2 has no effect on the output
    w2.grad = n.grad * x2.data

    # x1.grad = do/dx1
    # do/dx1 = do/dn * dn/dx1
    x1.grad = n.grad * w1.data

    # x2.grad = do/dx2
    # do/dx2 = do/dn * dn/dx2
    x2.grad = n.grad * w2.data

    # Observe how tanh is squashing the output value n
    draw.draw_dot(o).view()

    # Final gradient values for weights
    gradient_w1 = 1.0
    gradient_w2 = 0

    # Now we can conclude that if we want to increase output o, we have to increase weight w1 proportionally, while w2 has no effect.

if __name__ == '__main__':
    # computing_derivative_univariate_calculus()
    # computing_derivative_multivariate_calculus()
    simple_manual_nn_calculation_single_pass()