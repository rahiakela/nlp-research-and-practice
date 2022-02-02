import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilayerPerceptron(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, output_dims)
        
    def forward(self, x_in, apply_softmax=False):
        """
        The forward pass of the MLP
        Args:
            x_in (torch.Tensor): an input data tensor x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation should be false if used with
            the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        # print(intermediate.size())
        output = self.fc2(intermediate)

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output


def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: {}".format(x))


if __name__ == '__main__':

    """
    To demonstrate, we use an input dimension of size 3, an output dimension of size 4, 
    and a hidden dimension of size 100.
    """
    batch_size = 2    # number of samples input at once
    input_dim = 3
    hidden_dim = 100
    output_dim = 4

    # Initialize model
    print("################# Model structure #####################")
    mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)
    print(mlp)

    # We can quickly test the “wiring” of the model by passing some random inputs
    print("################# Input information #####################")
    x_input = torch.rand(batch_size, input_dim)
    describe(x_input)

    print("################# Output information #####################")
    y_output = mlp(x_input, apply_softmax=False)
    describe(y_output)

    """
    However, if you want to turn the prediction vector into probabilities, an extra step is required. 
    Specifically, you require the softmax activation function, which is used to
    transform a vector of values into probabilities.
    """
    print("################# Output information #####################")
    y_output = mlp(x_input, apply_softmax=True)
    describe(y_output)
