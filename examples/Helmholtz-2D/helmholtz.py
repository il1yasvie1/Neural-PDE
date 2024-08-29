import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from utils import autograd, generate_mesh, FCNet


f = lambda x, y: \
     (1. + 8.*torch.pi*torch.pi)*torch.cos(2*torch.pi*x)*torch.cos(2*torch.pi*y)
u_true = lambda x, y: \
     torch.cos(2*torch.pi*x)*torch.cos(2*torch.pi*y)


def compute_loss(invar):
    x = invar['x']
    y = invar['y']
    u = model(invar)
    ux, uy = autograd(u, [x, y])
    uxx = autograd(ux, x)[0]
    uyy = autograd(uy, y)[0]
    loss_pde = torch.mean((-(uxx+uyy) + u - f(x, y))**2)

    mask = (x == 0) | (x == 1)
    loss_x = torch.mean(ux[mask]**2)
    mask = (y == 0) | (y == 1)
    loss_y = torch.mean(uy[mask]**2)
    loss_boundary = loss_x + loss_y
    return loss_pde + loss_boundary


x, y = generate_mesh(1, 1, 50, 50)
x.requires_grad = True
y.requires_grad = True
invar = {'x': x, 'y': y}

model = FCNet()


def train(epochs=10000, lr=1e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = compute_loss(invar)
        loss.backward()
        opt.step()
        print(f"\repoch: {epoch}| loss: {loss} | error: {torch.linalg.norm(u_true(x, y) - model(invar))/torch.linalg.norm(u_true(x, y)): .2f}", end='')


# model.load_state_dict(torch.load('saved_model.pth'))
# train(epochs=10000, lr=1e-3)
# torch.save(model.state_dict(), 'saved_model.pth')
