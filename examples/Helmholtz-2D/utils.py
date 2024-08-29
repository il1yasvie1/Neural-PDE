import torch
import torch.nn as nn
from typing import List, Dict, Optional


def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
    grad = torch.autograd.grad(
        [y],
        x,
        grad_outputs=grad_outputs,
        create_graph=True,
        allow_unused=True,
    )
    if grad is None:
        grad = [torch.zeros_like(xx) for xx in x]
    assert grad is not None
    grad = [g if g is not None
            else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
    return grad


class FCNet(nn.Module):
    def __init__(self,
                 input_dimension=2,
                 output_dimension=1,
                 n_hidden_layers=6,
                 neurons=16,
                 ):
        super(FCNet, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Tanh()
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons)
             for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, invar: Dict[str, torch.tensor]):
        x = torch.hstack([invar[key] for key in invar])
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)


def generate_mesh(Lx, Ly, nx, ny):
    xs = torch.linspace(0, Lx, nx)
    ys = torch.linspace(0, Ly, ny)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    x = torch.flatten(x).reshape(-1, 1)
    y = torch.flatten(y).reshape(-1, 1)
    return x, y