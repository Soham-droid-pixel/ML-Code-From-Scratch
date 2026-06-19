import torch
import torch.nn as nn
import torch.optim as optim

class MLP_Torch(nn.Module):
    def __init__(self):
        super(MLP_Torch, self).__init__()
        # Creating sequential network layers
        self.network = nn.Sequential(
            nn.Linear(2, 3),   # Layer 1: 2 Inputs -> 3 Hidden Features
            nn.ReLU(),         # Structural non-linear activation element
            nn.Linear(3, 1),   # Layer 2: 3 Hidden Features -> 1 Target Projection
            nn.Sigmoid()       # Target mapping activation boundary
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    torch.manual_seed(42)
    
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0],      [1.0],      [1.0],      [0.0]], dtype=torch.float32)

    model = MLP_Torch()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.3)

    # Optimization Loop
    for epoch in range(15000):
        predictions = model(X)
        loss = criterion(predictions, y)
        
        optimizer.zero_grad()
        loss.backward()  # Automatic framework execution of internal backpropagation tracking
        optimizer.step()

    print("PyTorch Framework Verification Output:")
    final_preds = model(X).detach().numpy()
    for i in range(len(X)):
        print(f"Input: {X[i].numpy()} | Predicted Probability: {final_preds[i][0]:.4f}")