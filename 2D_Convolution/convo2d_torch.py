import torch
import torch.nn.functional as F

image = torch.tensor([[[
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
]]], dtype=torch.float32)

kernel = torch.tensor([[[
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
]]], dtype=torch.float32)

output = F.conv2d(image, kernel)

print("PyTorch 2D Convolution Output:")
print(output[0][0].numpy())