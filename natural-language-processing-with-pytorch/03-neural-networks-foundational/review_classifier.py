import torch
import torch.nn as nn


class ReviewClassifier(nn.Module):
    """
    The ReviewClassifier inherits from PyTorch’s and creates a single layer with a single output.
    Because this Module Linear is a binary classification setting (negative or positive review),
    this is an appropriate setup. The sigmoid function is used as the final non-linearity.
    """
    def __init__(self, num_features):
        """num_features (int): the size of the input feature vector"""
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)
        
    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier
        Args:
            x_in (torch.Tensor): an input data tensor. x_in.shape should be (batch, num_features)
            apply_sigmoid (bool): a flag for the sigmoid activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out
    