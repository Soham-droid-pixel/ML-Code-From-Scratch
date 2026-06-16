import torch
import torch.nn as nn

# 1. Prepare Data (Tensors)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. Define the Model (A single linear layer: y = wx + b)
model = nn.Linear(in_features=1, out_features=1)

# 3. Define the Loss Function and Optimizer
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. The Standard Training Loop
epochs = 1000
for epoch in range(epochs):
    # Step A: Forward pass
    y_pred = model(x)
    
    # Step B: Calculate loss
    loss = criterion(y_pred, y)
    
    # Step C: Clear old gradients
    optimizer.zero_grad()
    
    # Step D: Backpropagation (PyTorch does the calculus here)
    loss.backward()
    
    # Step E: Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        # Extract the learned weight and bias just to view them
        w = model.weight.item()
        b = model.bias.item()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | w: {w:.4f} | b: {b:.4f}")