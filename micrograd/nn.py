# Neural Networks
# https://cs231n.github.io/neural-networks-1/
# A neuron receives input signals(x1, x2, x3), these input signals are carried to the neuron through dendrites. The strength of the dendrites are represented by 
# w0, w1, w2 and called as weights. The idea is that these weights are learnable and control the strength of influence of one neuron on another.
# In the neuron, the product of the inputs and weights (w*x) are summed along with a bias(innate trigger happiness of a neuron). 
# The weighted sum is passed to an activation function f and if the output is above a threshold, the neuron fires, sending a spike along it's output dendrites.
# Activation functions (also called non-linearity) are squashing functions. Eg: Sigmoid, Tanh, Relu. They take a real valued input and squash it to a range.
# Sigmoid: 
#   Squashes a real valued number (the weighted sum) to range [0, 1] (Between 0 and 1)
#   Large negative numbers become 0 and large positive numbers become 1. 
#   Frequently used historically since it has a nice interpretation as the firing rate of a neuron: from not firing at all (0) to fully-saturated firing at an assumed maximum frequency (1)
#   Fallen out of favor now because of 2 major drawbacks (read the course notes): 
#        Sigmoids saturate and kill gradients
#        Sigmoid outputs are not zero-centered
# Tanh: 
#   Squashes a real-valued number to the range [-1, 1].
# ReLU:
#   It computes the function f(x)=max(0,x). In other words, the activation is simply thresholded at zero.

# Insights: 
# Backpropagation is just an algorithm and by itself is not related to Neural Networks at all. We are using backpropagation to compute the gradients of parameters and use gradient descent to nudge the paraneter values in the right direction. 
# Challenges
# If you have a bug in your code, your NN may still perform well on small or simple problems, but lead to large losses for large or complex problems.
# Put another way, for small simple problems, it is very easy for Neural Nets to fit the problem despite bugs in the implementation like forgetting to zero grad.

import random

from engine import Value
import visualizer.graph as draw

class Neuron:

    # nin: Number of inputs to this neuron
    def __init__(self, nin):

        # randon.uniform(): Draw samples from a uniform distribution. Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
        # Uniform distribution means any value within the given interval is equally likely to be drawn by uniform.
        # For every input to the neuron, create a random float as weight
        # If nin=3, self.w = [0.34, -0.5, 0.9]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # Random value for bias
        self.b = Value(random.uniform(-1, 1))
    
    # n = Neuron(2)
    # x = [2.0, 3.0]
    # n(x) => Python will invoke the __call__ function on n.
    # Forward pass computation for a single neuron with input tensor x (x1, x2, ...)
    def __call__(self, x):
        # Forward pass through a single Neuron with x inputs: w*x + b
        # Dot product of w and x where w and x are 1D tensors
        # x = [2.0, 3.0], w = [-0.5, 0.5]
        # Iterate over w and x simultaneously using zip (Zip is used to iterate over multiple sequences simultaneously)
        # Get an array of weighted sums
        # One liner: activation = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        weighted_sums = [wi*xi for wi, xi in zip(self.w, x)]
        forward_pass_output = sum(weighted_sums) + self.b

        # Pass through tanh non-linearity
        out = forward_pass_output.tanh()
        return out

    # Collect all the parameters of a single neuron, weights + bias
    def parameters(self):
        return self.w + [self.b]


class Layer:

    # Create a layer of neurons
    # nin: Number of inputs to a single neuron in the network
    # nout: Number of neurons in this layer
    # To create a Layer with 3 Neurons, where each Neuron has 2 inputs (x1 and x2), 
    # layer = Layer(2, 3)
    def __init__(self, nin, nout):
        # Create nout Neurons in this layer with each Neuron having nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    # Trigger the forward pass computation for each Neuron in this layer given the inputs x to this layer
    # Output is array of forward pass computation values for each Neuron in the layer
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    
    # Collect all the parameters of a layer
    def parameters(self):
        # Python list comprehension for the below
        return [param for neuron in self.neurons for param in neuron.parameters()]
        # parameters = []
        # for neuron in self.neurons:
        #     params = neuron.parameters
        #     parameters.extend(params)
        # return parameters

class MLP:

    # Create a Multi Layer Perceptron
    # nin: Number of inputs to a single neuron in the network
    # nouts: An Array of nouts, indicating the number of neurons in each layer (hidden layers + output layer)
    # nouts = [2, 3, 3, 1] means we have three hidden layers, with 2, 3 and 3 neurons each respectively and an output layer with 1 neuron
    # In an MLP, the number of inputs given by nin, form the input layer 
    # Size of all layers = [input layer] + hidden and output layers  (nouts) 
    # Now, the number of inputs to a neuron in layer i+1 = nuber of neurons in the previous layer i
    # Consider an MLP with layers [2, 3, 3, 1]. 
    # This means an MLP with two inputs (nin=2, input layer size =2), 3 neurons in first hidden layer and another 3 neurons in second input layer. One neurin in output layer.
    # Each Neuron in first hidden layer will have 2 inputs since input layer size=2
    # Each Neuron in the second hidden layer will have three inputs since first hidden layer size = 3
    # The single Neuron in the output layer will have three inputs since second hidden layer size = 3
    # In general, each neuron in layer i+1 will have i inputs to each neuron 
    def __init__(self, nin, nouts):
        # nin: Number of inputs to the MLP network
        # nouts: An array indicating number of hidden and output layers and number of neurons in each layer
        # Combine input layer size with hidden layer sizes 
        layer_sizes = [nin] + nouts
        
        # Create an array of Layers in the MLP. Layer(nin, nout)
        # NOTE: Iterate only till the penultimate layer, i+1 will be out of range for the output layer
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(nouts))]

    # Trigger the forward pass computation for the MLP
    # Each Neuron in the output layer will have a forward pass output value
    # x = the input to the MLP, x = [x1, x2, x3]
    # For the first hidden layer, the input will be the input layer of the network.
    def __call__(self, x):
        # For layers = [3, 4, 4, 1], we have three layers
        # self.layers = [Layer(3,4), Layer(4,4), Layer(4,1)]
        # MLP input layer x = [x1, x2, x3]
        # First loop iteration, x = [x1, x2, x3], compute forward pass on HiddenLayer=1, now,  x = [x11, x12, x13, x14] (Four neurons in HL=1)
        # Second loop iteration, x = [x11, x12, x13, x14], compute forward pass on HiddenLayer=2, now, x = [x21, x22, x23, x24] (Four neurons in HL=2)
        # Third loop iteration, x = [x21, x22, x23, x24], compute forward pass on OutputLayer, now x = [x0] (1 neuron in output layer)
        for layer in self.layers:
            x = layer(x)
        return x
    
    # Collect all the parameters of the MLP
    def parameters(self):
        # List comprehension magic
        return [params for layer in self.layers for params in layer.parameters()]


def forward_pass_computation_runs():
    # Forward pass through a single neuron with two inputs x and two random weights
    x = [2.0, 3.0]
    n = Neuron(2)
    forward_pass_out = n(x)
    # print(forward_pass_out)

    # Create a layer of 3 Neurons with two inputs going into each Neuron
    # x = input to this layer
    layer = Layer(2, 3)
    x = [2.0, 3.0]
    forward_pass_values = layer(x)
    # print(forward_pass_values)

    # MLP with 3 inputs, two hidden layers of size 4 and output layer with 1 neuron
    # Draw and visualize this network  
    # Since output layer size = 1, the single output neuron's forward pass value is the MLP output
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    mlp_output = n(x)
    # print('MLP: ', mlp_output)

    # Draw the graph, note, take mlp_output[0] since we have a single output neuron
    draw.draw_dot(mlp_output[0]).view()
    
    # Now let's try to make some predictions using this mlp
    MLP_training_run(n)

def MLP_training_run(mlp):
    # Supervised Training data labels for the dataset where x consists of 3 values [x1, x2, x3] producing a single output [ys]
    # xs[0] has expected prediction ys[0]
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 0.5, 1],
        [1.0, 1.0, -1.0]
    ]
    # Desired outputs
    ys = [1.0, -1.0, -1.0, 1.0]

    # Prediction of the mlp for the given input xs
    # The prediction of the mlp is bad at this point, it's far from the expected result of [1.0, -1.0, -1.0, 1.0]
    # Here we made four forward passes through the network since there were 4 inputs
    ypred = [mlp(x) for x in xs]
    print(ypred) # [[Value(data=0.4039733942797753)], [Value(data=0.8660063620804366)], [Value(data=0.8759343529699677)], [Value(data=0.35105790624569133)]]

    # Let's quantify the performance of the MLP using the Mean Squared Error(MSE) loss. The loss tells us how good the network is at predicting the result.
    # MSE is one of the loss functions used in Deep Learning. Cross Entropy loss is more commonly used. Discard the sign.
    # MSE loss = (yactual - ypred)**2
    # Initially loss will be high, end goal is to minimize the loss by adjusting the weights of the network.

    # Loss for every output in ys
    individual_losses = [(yout[0]-yactual)**2 for yactual, yout in zip(ys, ypred)]

    # Total loss = Sum(MSE loss)
    # yout[0] because we have a single value from the 1 output neuron wrapped in an array. Look at __call__ of MLP module
    loss = sum((yout[0]-yactual)**2 for yactual, yout in zip(ys, ypred))

    # So far, the weights in the network were randonmly generated values. How do we tune these weights so that the loss is minimized?
    # Backpropagte from the loss and calculate the gradient at each parameter(weight/bias) in the network with respect to the loss 
    # Knowing the gradient at each parameter tells us how to nudge the weight so that the loss goes down.
    # The gradient of an intermediate parameter(weight or bias) with respect to the loss tells us the direction of the slope with respect to the parameter.
    # The gradient points in the direction of increased loss. Or we can say that the "gradient vector"(vector of all gradients) points in the direction of increasing loss.
    # If gradient (dweight/dloss) is positive, it means increasing the weight will increase the loss, decreasing it will decrease the loss.
    # If gradient is negative, increasing the weight will decrease the loss and decreasing will increase the loss.

    # Let's run backpropagation and calculate the gradients for all the parameters in the network (weights and biases)
    # Note: Backpropagation will go all the way back to the inputs, which means gradients are computed for the inputs too, but input gradients are not used since inputs are a given and not changed
    loss.backward()

    # Collect all the parameters (weights + biases)
    # There are 41 parameters for the MLP defined by [3,4,4,1]
    # Number of weights = 12 + 16 + 4 = 32
    # HL1 = 3 * 4 = 12 (3 inputs going to 4 neurons in Layer 1)
    # HL2 = 4 * 4 = 16 (connections from 4 neurons in HL1 going to 4 neurons in HL2)
    # OutputLayer = 4 * 1 = 4 (connections from 4 neurons in HL2 going to 1 neurons in OL)
    # Number of biases = number of neurons = 4 + 4 + 1 = 9 {Every neuron has a bias, There are no neurons in input layer}
    # Total parameters = weights + biases = 32 + 9 = 41
    # Weights should be visualized as the edges going from one neuron in one layer to all the neurons in the succeeding layers
    # IMPORTANT: The nodes/neurons in the network are not important, instead the weights which are the connections between the neurons are what's tuned
    # Weights are the strength of the dendrites connecting the neurons in the network
    mlp_parameters = mlp.parameters()
    print('No of parameters', len(mlp_parameters))

    # GRADIENT DESCENT
    # Each of the parameter (weight or bias) is a Value object with data and grad
    # Let's nudge the parameter data using the knowledge of the gradient so that the loss is decreased.
    # Since the gradient vector points in the direction of increasing loss, we have to change the parameter value in the opposite direction to the gradient.
    # By multiplying by -1
    step_size = 0.001
    for parameter in mlp_parameters:
        # Nudging the parameter value in the opposite direction of the gradient
        parameter.data += -1 * step_size * parameter.grad
    
    # Now, with these updated weights and biases, do another forward pass and generate a new set of predictions for the same inputs xs
    # Now calculate new MSE loss, it will be slightly lower than before
    # Now, we can trigger backpropagation on the loss again and update the gradients of the weights and biases again
    # Now based on the new gradients, again nudge the parameter values by a small step size in the opposite direction of gradient.
    # Again do a forward pass with the updated weights and biases and generate new predictions which should be closer to the expected result.
    # We do this iteration again and again until the loss converges to a low value (no more improvements in loss after a point)
    # GRADIENT DESCENT EXPLANATION
    # The whole iteration of starting with initial parameter values(weights+biases), doing a forward pass to generate predictions, computing the MSE loss, backpropagating the loss and 
    # computing new gradients for the parameters(weights+biases) with respect to the loss, nudging/updating the parameter values(weights+biases) in the opposite direction of the gradient, then 
    # doing another forward pass with new parameter values over and over again until the loss is minimized and converges is called GRADIENT DESCENT.
    # One iteration of gradient descent is called a step. You may take 500 steps or 1000s of steps depending on your training data and hyperparameters.

    # Step size has to be just right during gradient descent. Step size is also called the learning rate.
    # If the step size is too big, we may end up taking too big a step, overstepping and increasing the loss. This destabilizes training by leading to a lot of loss increases and reductions
    # If the step size is too small, the loss reduces very slowly and it takes a long time for the loss to converge.
    
    # A gradient descent training iteration with 20 steps
    for k in range(20):
        # Forward pass and compute predictions
        ypred = [mlp(x) for x in xs]

        # MSE Loss
        loss = sum((yout[0]-yactual)**2 for yactual, yout in zip(ys, ypred))

        # IMPORTANT: Flush or reset the parameter gradients to 0 before computing new gradients during backpropagation.
        # During backward(), gradients are accumulated in the Value engine.
        # If we don't reset the gradient to 0, the gradients calculated for a step are added on to the gradients calculated in the previous step
        # During each step, gradient should be newly calculated from 0.0.
        # This is one of the common mistakes when working with Neural Networks as Karpathy called out.
        for parameter in mlp.parameters():
            parameter.grad = 0.0

        # Backward pass, backpropagation
        loss.backward()

        # Update the weights and biases
        learning_rate = 0.05
        for parameter in mlp.parameters():
            parameter.data += -1 * learning_rate * parameter.grad
        
        # Loss value. It should keep decreasing with every step.
        # Notice that loss reduces smoothly and gradually instead of abruptly and suddenly as was the case when gradients were not reset to 0.
        print(k, loss.data)

if __name__ == '__main__':
    forward_pass_computation_runs()









