# A character level language model is predicting the next character in a sequence given a sequence of characters before it.
# For such a language model, even a single name in the dataset is actually a bunch of examples packed together.
# What does the existence of a word 'isabella' in the dataset tell us? Statistically, 
# Character i is likely to come first in the sequence. 
# Character s is likely to come after character 'i'.
# Character a is likely to come after 'is'.
# Character b is likely to come after 'isa' and so on.
# Also, there is one more example packed at the end. After 'isabella', the sequence is likely to end. This can be modeled as an end token.
# Thus we have multiple sequence examples in a single name and we have 32000 such names.
# The model is building a statistical structure of likelihood or probabilities of characters appearing together.
# In a bigram language model, we are working with only 2 characters in a row at a time.
# For 'isabella', bigrams are <start>i, is, sa, ab, be, el,ll,la, a<end>
# Use first character to predict second character.

# Word to bigrams
# Compute count of each bigram to track most frequently appearing bigrams

import torch
import matplotlib.pyplot as plt

def word_to_bigrams_frequency_dictonary():
    words = open('sequencemodels/names.txt', 'r').read().splitlines()
    bigram_freq = {}

    # zip will iterate over two sequences, even two strings.
    # To get bigrams of a string, iterate over the characters in the same string from position 0 and 1 (two pointers)
    # Pointer 1 points to first character in bigram, second pointer points to second character in bigram
    for word in words:
        # Add special start and end tokens to the beginning and end of the word
        # List(word) converts a word to array of Characters
        chs = ['<S>'] + list(word) + ['<E>']
        for ch1,ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    
    # Sort bigram frequencies by highest value first
    print(sorted(bigram_freq.items(), key = lambda kv: -kv[1]))

# It is much more efficient to store bigram frequencies in a 2D 28X28 matrix instead of a Python dictionary
# 26 alphabets + 2 special tokens
# Manipulation of values in a matrix is much more efficient compared to a dictionary
# Matrix Vectorization
# The rows of the matrix form the first character of the bigram and the columns form the second character of the bigram
# We need to map characters to numbers using a lookup table
def word_to_bigrams_frequency_matrix():
    words = open('sequencemodels/names.txt', 'r').read().splitlines()

    # Initialize a 27x27 matrix of int32, since each alphabet is mapped to an integer + a '.' character for stop and end tokens
    N = torch.zeros((27, 27), dtype=torch.int32)
    
    # Get the set of all characters from the dataset. set(entire_dataset_as_a_string) will create a set with all unique characters
    chars = sorted(list(set(''.join(words))))

    # Create a lookup table of symbols or characters to integers. Start from index 1.
    stoi = {s:i+1 for i, s in enumerate(chars)}

    # Set start and end token '.' as the first element, index=0
    stoi['.'] = 0


    # Populate the 28X28 array with counts of all the bigrams
    # The rows of the matrix form the first character of the bigram and the columns form the second character of the bigram
    for word in words:
        # Add special start and end tokens to the beginning and end of the word
        # List(word) converts a word to array of Characters
        chs = ['.'] + list(word) + ['.']
        # For every bigram in a word, find its position in the matrix and increment the count
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    
    # Reverse lookup table of integer to symbol from stoi
    itos = {i:s for s, i in stoi.items()}

    # Visualize the matrix
    # visualize_matrix(N, stoi, itos)

    # Vectorized Normalization of the matrix into a bigram probability distribution
    # Leveraging broadcasting semantics indirectly via keepdim=True
    P = N.float()
    # P = (N+1).float() # Model Smoothing to prevent Infinity loss
    P /= P.sum(dim=1, keepdim=True) # In place operation is efficient than p = p/p.sum()

    # Visualize the bigram probability distribution matrix
    # visualize_matrix(P, stoi, itos)
    # print(P[0].sum()) # Sum=1

    # Row0 = N[0] = Counts of bigrams starting with .., a,b,c...z (.a, .b, .c, .d, ..., .z)
    # Row1 = N[1] = Count of bigrams starting with a (a., aa, ab, ac, ..., az)
    # Row2 = N[2] = Count of bigrams starting with b (b., ba, bb, bc, ..., bz)
    # N[0] is the first row, the count of bigrams starting with a, b, c, ... z. Values are bigram counts in int32 since tensor was created as such.
    # print(N[0])

    # Picking a single likely starting letter from the first row
    # Convert the counts to probability distribution by dividing each count by row sum. Convert to float before dividing.
    # Convert the first row to probability distribution by normalizing by the sum
    # P = N[0].float()
    # P = P/P.sum(dim=1, keepdim=True)

    # # Random fixed seed for reproducibility
    # generator = torch.Generator().manual_seed(2147483647)

    # # Draw a single sample value from the first row with replacement. We expect a frequent letter to be picked based on prob distribution.
    # ix = torch.multinomial(P, num_samples=1, replacement=True, generator=generator).item()

    # # Print the letter that was picked, K got picked, it appears 2963 times where it was ending character
    # print(itos[ix])


    # Now Generate or sample 10 new names using the bigram model
    # The matrix is already normalized into a probability distribution.
    # Step 1: Pick a likely starting letter from first row N[0]. (First row has starting letters .a, .b etc)
    # Step 2: For the letter that was picked, let's say m, go to the row which contains bigrams starting from m and pick a likely letter from this row which follows m.
    # Step 3: Let o be the letter which was picked following m. Now go to the row which contains bigrams starting from o and pick a likely letter from this row which follows o.
    # Continue until we pick the ending character . as the likely character. Join the characters to form the name.
    for i in range(20):
        name = []
        ix = 0
        while True:
            p = P[ix]
            # Call .item() to unwrap the tensor to an int
            # No generator, random samples each time
            ix = torch.multinomial(p, num_samples=1, replacement=True).item()
            name.append(itos[ix])
            # If . end token is picked as next character, stop and exit out of the loop
            if ix == 0:
                break
        print(''.join(name))

    # At this point, we have a bigram language model based on counts of bigrams.
    # The matrix P represents the weights or parameters of the model (bigram frequencies)
    # Now let's evaluate the quality of the model against training data (training loss) and quantify it as a single number.
    # What is the probability of the model predicting each possible bigram from the training dataset?
    # The bigram probability can be simply looked up from the matrix which contains the probability distributon of bigrams from training dataset.
    # Prob(b1b2) = P[b1, b2]
    # Printing the bigram probabilities for the first 3 words from the dataset.
    # Printing bigram probabilities for first 3 words
    for w in words[:3]:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            # print(f'{ch1}{ch2}: {prob: 0.4f}')

    # Estimating the quality of the model using Maximum Likelihood Estimation
    # MLE is a statistical method for estimating the parameters of an assumed probability distribution, given some observed data.
    # Likelihood is defined as the product of all the probabilities in the probability distribution (elements of the matrix)
    # Since each element of matrix is between 0 and 1, likelihood value will be very very small. 
    # For convenience, people work with Log Likelihood becuase we can add the log of the probabilties up instead of multiplying the probabilities
    # log(a*b*c) = log a + log b + log c
    # Since we have a probability distribution, log(x) where x [0,1] has bounds of [-infinity, 0]
    # https://www.wolframalpha.com/input?i=log%28x%29+from+0+to+1

    # GOAL: maximize likelihood of the data w.r.t. model parameters (product of the probabilities) (statistical modeling)
    # Equivalent to maximizing the product of the probabilities
    # equivalent to maximizing the log likelihood (because log is monotonic)
    # Now for an improved model, probabilities will be closer to 1, which will result in log likelihood increasing and going towards 0
    # Whereas for a bad model, probabililties will be closer to 0 which will result in log likelihood reducing and going towards -infinity
    # We want a loss function with semantics that model improves when loss function value is lower and is bad when values are higher
    # Therefore the loss function is the negative log likelihood and goal is to minimize the negative log likelihood (OR) equivalent to minimizing the average negative log likelihood
    log_likelihood = 0
    count = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            count += 1
            # print(f'{ch1}{ch2}: {prob:0.4f} {logprob: 0.4f}')
    print(f'{log_likelihood=}')
    # Negative log likelihood is a good loss function
    # Lowest possible value is 0 when probabilities are 1 and the higher it gets the worse off the model is
    nll = -log_likelihood
    print(f'{nll=}')
    # Normalized or average negative log likelihood
    # A single value that is a measure of the quality of the model
    # Note: This is the loss function of the training dataset assigned by the model
    # The goal of training is to find parameters that minimize the negative log likelihood.
    normalized_nll = nll/count
    print(f'{normalized_nll=}')

    # Problems with the above naive implementation 
    # We can evaluate the probability of the model generating any particular word 
    # Because the bigram jq has zero occurences in the dataset, P[j,q] = 0 and logprob = -Infinity
    # Hence log_likelihood also becomes -Infinity, indicating that the model has 0% chance of predicting this bigram
    # This is not desirable
    andrejq_log_likelihood = 0
    for w in ['andrejq']:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            andrejq_log_likelihood += logprob
            count += 1
            # print(f'{ch1}{ch2}: {prob:0.4f} {logprob: 0.4f}')
    print(f'Arun {andrejq_log_likelihood=}') # andrejq_log_likelihood = -Infinity

    # The simple way to fix this issue is to perform model smoothing with fake counts
    # Add some fake count=1 to every thing
    # See above during vectorized normalization of the matrix




def visualize_matrix(N, stoi, itos):
    plt.figure(figsize=(16, 16))
    # Plot the matrix first
    plt.imshow(N, cmap='Blues')

    # Iterate over every cell row by row
    for i in range(27):
        for j in range(27):
            # Print the Bigram string for every cell in the matrix
            chrstr = itos[i] + itos[j]
            plt.text(j, i, chrstr, ha='center', va='bottom', color='gray')
            # Print the count of bigram in the cell. N[i, j] is still a Tensor(dtype=int32), N[].item() will unwrap and extract the integer.
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
    plt.show()
            

if __name__ == '__main__':
    # word_to_bigrams_frequency_dictonary()
    word_to_bigrams_frequency_matrix()