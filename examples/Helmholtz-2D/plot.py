from helmholtz import FCNet, generate_mesh
import torch
import matplotlib.pyplot as plt


model = FCNet()
model.load_state_dict(torch.load('saved_model.pth'))
x, y = generate_mesh(1, 1, 100, 100)
invar = {'x': x, 'y': y}
u_pred = model(invar)
u_true = torch.cos(2*torch.pi*x)*torch.cos(2*torch.pi*y)

plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u_pred.detach().numpy())
plt.savefig('./figures/u_pred.png')

plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u_true.detach().numpy())
plt.savefig('./figures/u_true.png')

plt.scatter(x.detach().numpy(), y.detach().numpy(), c=abs(u_pred.detach().numpy() - u_true.detach().numpy()))
plt.savefig('./figures/error.png')
