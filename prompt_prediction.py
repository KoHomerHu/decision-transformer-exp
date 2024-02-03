import torch
import torch.nn.functional as F


"""Probabilistic neural network that outputs a multivariate normal distribution."""
class PNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PNN, self).__init__()
        layer_dim = [input_dim,] + hidden_dim + [output_dim * 2,]
        self.fc = torch.nn.ParameterList(
            [torch.nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)]
        )

    def forward(self, x):
        for layer in self.fc:
            x = F.relu(layer(x))
        mu, std = torch.split(x, x.size(-1) // 2, dim=-1)
        std = F.softplus(std) + 1e-6
        return mu, std
    
    """NLL loss function for the PNN."""
    def loss(self, mu, std, y):
        return (mu - y).dot(mu - y) / (std ** 2) + 2 * torch.log(std).sum()