import torch
# Tensors
# A tensor is a generalization of scalars, vectors, and matrices. It can have any number of dimensions:
# Scalar (0D Tensor): A single number → torch.tensor(5)
# Vector (1D Tensor): A 1D array → torch.tensor([1, 2, 3])
# Matrix (2D Tensor): A 2D (MxN) array → torch.tensor([[1, 2], [3, 4]]) 
# Higher-Dimensional Tensor (3D+): Example → torch.randn(2, 3, 4)
# You can reshape or convert between 1D and 2D tensors easily using .view(), .reshape(), or .flatten().

# torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None) → LongTensor
# Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
# If samples are drawn without replacement, which means that when a sample index is drawn for a row, it cannot be drawn again for that row.
def sampling_from_a_multinomial_distribtion():
    # A probability distribution summing to 1
    values = torch.tensor([0.6, 0.3, 0.1])

    # Ensures reproducibility by setting a fixed random seed. torch.Generator() creates a generator associated with the CPU.
    generator = torch.Generator()

    # Sample 10 values with replacement from the probability distribution
    # multinomial returns indexes from the values tensor
    sampled_indices = torch.multinomial(values, num_samples=10, replacement=True, generator=generator)
    
    # Observe that index=0 occurs around 6 times (60%), index=1 occurs around 3 times (30%) and index=2 occurs around 1 time(10%)
    # If you don't use a generator, then for small sample sizes, you will see some small variation in the sampling distribution, it won't exactly follow the probability distribution but close to it.
    print(sampled_indices)

def tensor_manipulations_broadcasting():
    # A 5x5 matrix
    matrixA = [
        [2, 2, 2, 2, 2],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
        [10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20]
    ]

    # Creating a tensor from the matrix
    tensorA = torch.tensor(matrixA)
    print(tensorA.dtype) # torch.int64 (Pytorch will create ints as int64 by default)
    print(tensorA.shape) # torch.Size([5, 5])

    # Convert to float32
    tensorA = tensorA.to(torch.float)
    print(tensorA.dtype)

    non_vectorized_normalization(tensorA)
    vectorized_normalization(tensorA)

# Normalize the rows as a probability distribution by dividing each element in a row by that row's sum
# Non-vectorized approach:
#   Iterate over every row
#   Divide the row by it's row sum and reassign 
# This is very inefficient and wasteful.
def non_vectorized_normalization(N):
    rows = N.shape[0]
    for i in range(rows):
        N[i] = N[i].float()
        N[i] = N[i]/N[i].sum() # N[i] is a Row vector. Calling sum() on the row vector computes the total sum of the row.
    
    # Output will be a 5x5 tensor with all values 0.2 
    print('Non-Vectorized Approach')
    print(N)


def vectorized_normalization(N):
    # Shape [5, 1]
    row_sum_tensor = N.sum(dim=1, keepdim=True)

    N = N/row_sum_tensor
    print('Vectorized Approach')
    print(N)

def torch_compute_tensor_sums():
    # A 5x5 matrix
    matrixA = [
        [2, 2, 2, 2, 2],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
        [10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20]
    ]

    # Creating a tensor from the matrix
    tensorA = torch.tensor(matrixA)
    print(tensorA)

    # Compute a single sum tensor by summing up all the values in the tensor
    total_sum_tensor = tensorA.sum()
    print('Total tensor sum: ', total_sum_tensor.item())

    # Compute Sum across columns
    # Compute the sum column by column to get a row vector
    column_sum_tensor = tensorA.sum(dim=0, keepdim=True) # First argument Dimension=0 for column sums

    # Output: tensor([[41, 41, 41, 41, 41]])
    # We get a row vector of shape torch.Size([1, 5])
    print('Column sum tensor: ', column_sum_tensor)
    print('Shape: ', column_sum_tensor.shape)

    # What happens if we don't set keepdim=True? (By default keepdim is False)
    column_sum_tensor_nodim = tensorA.sum(dim=0)
    print('No dimensions')

    # Output: tensor([41, 41, 41, 41, 41])
    # We get a row vector of shape torch.Size([5])
    # The dimension 1 has been squeezed out
    print('Column sum tensor: ', column_sum_tensor_nodim)
    print('Shape: ', column_sum_tensor_nodim.shape)

    # Compute Sum across Rows
    # Compute the sum row by row to get a column vector
    row_sum_tensor = tensorA.sum(dim=1, keepdim=True) # First argument Dimension=1 for row sums

    # Output: tensor([[ 10],
    #                 [ 20],
    #                 [ 25],
    #                 [ 50],
    #                 [100]])
    # We get a column vector of shape torch.Size([5, 1])
    print('Row sum tensor: ', row_sum_tensor)
    print('Shape: ', row_sum_tensor.shape)

    # Similarly, keepdim=False will result in the dimension being squeezed out

# Many pytorch operations support broadcast semantics
# If a PyTorch operation supports broadcast, then its Tensor arguments can be 
# automatically expanded to be of equal sizes (without making copies of the data).
# https://pytorch.org/docs/stable/notes/broadcasting.html

# Two tensors are “broadcastable” if the following rules hold:
# Each tensor has at least one dimension.
# When iterating over the dimension sizes, starting at the trailing dimension, 
# the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
def torch_broadcast_semantics():
    # A 5x5 matrix
    matrixA = [
        [2, 2, 2, 2, 2],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
        [10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20]
    ]

    # Creating a tensor from the matrix
    # Shape [5, 5]
    tensorA = torch.tensor(matrixA)

    # Compute another tensor of row sums as a column vector
    # Shape [5, 1]
    # tensor([
    #   [ 10],
    #   [ 20],
    #   [ 25],
    #   [ 50],
    #   [100]
    # ])
    # Because, keepdim=True, the second dimension of 1 is retained, otherwise it is squeezed out and we get a 1-D array
    row_sum_tensor = tensorA.sum(dim=1, keepdim=True)

    # Now, we want to divide each row in tensorA by it's row sum that is computed in row_sum_tensor
    # Expected result 
    #    [2/10, ....],
    #    [4/20, ....],
    #    [5/25, ....],
    #    [10/50, ....],
    #    [20/100, ....]

    # To correctly divide tensorA (5x5) by row_sum_tensor (5X1), we need to apply broadcasting semantics
    # 5X5 2D array or matrix
    # 5X1 2D array or matrix
    # Align from the right side
    # 5X5
    # 5X1
    # The column vector is expanded to all 5 columns of that row before piece by piece division
    # [2, 2, 2, 2, 2]/[10, 10, 10, 10, 10]
    # [4, 4, 4, 4, 4]/[20, 20, 20, 20, 20]
    # ... and so on ....
    normalized_tensorA = tensorA/row_sum_tensor
    print(normalized_tensorA)

    # BEWARE BROADCAST SEMANTICS
    # Now, imagine we forgot to keepdim=True when computing row sum
    # Then the row_sum_tensor will be shape [5]
    # Aligining for broadcast semantics, we get
    # 5x5
    #   5
    # Pytorch will add 1 when there is no dimension, so we get,
    # 5x5
    # 1x5 
    # Now, tensorA/row_sum_tensor will be a division of a (5x5) tensor by a (1x5) row vector
    # Pytorch will do a row by row division which is the wrong result
    # [2, 2, 2, 2, 2]/[10, 20, 25, 50, 100]
    # [4, 4, 4, 4, 4]/[10, 20, 25, 50, 100]


    



if __name__ == '__main__':
    #sampling_from_a_multinomial_distribtion()
    # torch_compute_tensor_sums()
    tensor_manipulations_broadcasting()
    # torch_broadcast_semantics()