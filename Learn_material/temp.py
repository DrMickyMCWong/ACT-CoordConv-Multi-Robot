import torch

# Simulate target trajectory with sharp jumps
T = 5
x = torch.tensor([0., 0., 1., 1., 2.], requires_grad=False)

# Initialize prediction to zero smooth curve
hat_x = torch.zeros(T, requires_grad=True)

alpha = 0.5  # temporal weight

optimizer = torch.optim.SGD([hat_x], lr=0.1)

for i in range(100):
    # Value loss
    L_val = ((hat_x - x)**2).sum()

    # Temporal loss (velocity diff)
    pred_v = hat_x[1:] - hat_x[:-1]
    true_v = x[1:] - x[:-1]
    L_temp = ((pred_v - true_v)**2).sum()

    L = L_val + alpha * L_temp
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

print(hat_x.detach().numpy())
