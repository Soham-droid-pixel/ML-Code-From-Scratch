import numpy as np

image = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
])

kernel = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
])

output = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        image_chunk = image[i : i+3, j : j+3]
        output[i, j] = np.sum(image_chunk * kernel)

print("NumPy 2D Convolution Output:")
print(output)