import torch
import torch.nn as nn

def softmax_torch(logits_tensor):
    # ==========================================
    # INTERVIEW NOTE: PyTorch implements its own 
    # highly optimized, stable version internally.
    # 'dim=0' means we compute the softmax across 
    # the elements of a 1D tensor vector.
    # ==========================================
    softmax_fn = nn.Softmax(dim=0)
    return softmax_fn(logits_tensor)

if __name__ == "__main__":
    # In PyTorch, inputs must be converted into Tensors with specified data types
    logits = torch.tensor([3.2, 1.1, -0.5], dtype=torch.float32)
    
    probabilities = softmax_torch(logits)
    
    print(f"Raw Logits Tensor: {logits}")
    # .detach().numpy() converts the tensor tracking graph back to a standard NumPy array for printing
    print(f"Probabilities: {probabilities.detach().numpy()}")
    print(f"Total Sum: {torch.sum(probabilities).item():.1f}")
    
    expected_jersey = 0.871
    
    # We use .item() to extract the scalar float value out of the 0-dimensional PyTorch tensor
    if abs(probabilities[0].item() - expected_jersey) < 0.01:
        print("Status: Passed")
    else:
        print("Status: Failed")