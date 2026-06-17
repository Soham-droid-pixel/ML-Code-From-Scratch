import torch
import torch.nn as nn

# 1. Define the Architecture
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_features):
        super(LogisticRegressionModel, self).__init__()
        # A single linear transformation (z = wx + b)
        self.linear = nn.Linear(input_features, 1)
        # The activation function to squish the output
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        z = self.linear(x)
        probability = self.sigmoid(z)
        return probability

# 2. Prepare Data (Tensors)
# 2 samples, 2 features
X = torch.tensor([[2.5, 1.2], [0.5, 0.2]], dtype=torch.float32) 
# Target classifications
y = torch.tensor([[1.0], [0.0]], dtype=torch.float32) 

# 3. Initialize Model, Loss, and Optimizer
model = LogisticRegressionModel(input_features=2)

# BCELoss = Binary Cross Entropy. 
# This is the standard loss function for predicting 1s and 0s.
criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. The Standard Training Loop
epochs = 1000
for epoch in range(epochs):
    # Step A: Forward pass
    y_pred = model(X)
    
    # Step B: Calculate loss
    loss = criterion(y_pred, y)
    
    # Step C: Clear old gradients
    optimizer.zero_grad()
    
    # Step D: Backpropagation (PyTorch does the calculus here)
    loss.backward()
    
    # Step E: Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")