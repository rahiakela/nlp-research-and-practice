from torch import nn
import torch.nn.functional as F


class SurnameClassifier(nn.Module):
    """A 2-layer multilayer perceptron for classifying surnames"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(SurnameClassifier, self).__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x_in, dropout=None, apply_softmax=False):
        """
        The forward pass of the classifier
        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            dropout (float): an input data for drop probability.
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        # The first layer maps the input vectors to an intermediate Linear vector,
        # and a nonlinearity is applied to this vector
        intermediate_vector = F.relu(self.fc1(x_in))
        if dropout is not None:
            intermediate_vector = F.dropout(intermediate_vector, p=dropout)
        # and second Linear layer maps the intermediate vector to the prediction vector
        prediction_vector = self.fc2(intermediate_vector)

        """
        In the last step, the softmax function is optionally applied to make sure the outputs
        sum to 1; that is, are interpreted as “probabilities.”
        """
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector
