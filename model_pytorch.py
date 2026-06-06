
import torch
import torch.nn as nn

torch.manual_seed(0)

# dataset
xs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=torch.float32)

ys = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Model — MLP: 2 inputs -> hidden(4, ReLU) -> output(1, linear)
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of parameters: {total_params}\n")

# training
lr     = 0.1
epochs = 200

loss_fn   = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    # forward pass
    preds = model(xs)
    loss  = loss_fn(preds, ys)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # SGD parameter update
    optimizer.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}  MSE = {loss.item():.6f}")

# predictions
print("\n--- Predictions after training ---")
with torch.no_grad():
    for i in range(len(xs)):
        pred   = model(xs[i]).item()
        target = ys[i].item()
        x0, x1 = xs[i][0].item(), xs[i][1].item()
        print(f"x=({x0:.0f},{x1:.0f})  pred={pred:.6f}  target={target:.1f}")
