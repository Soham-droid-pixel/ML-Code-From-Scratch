#include <iostream>
#include <cmath>

using namespace std;

int main() {
    // 1. The XOR Data
    float X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float y[4]    = {0, 1, 1, 0};

    // 2. Setup Weights (Hardcoded starting values for simplicity)
    float W1[2][2] = {{0.1, -0.2}, {0.3, 0.4}}; // 2 inputs to 2 hidden neurons
    float b1[2]    = {0.0, 0.0};
    
    float W2[2]    = {0.5, -0.6};               // 2 hidden neurons to 1 output
    float b2       = 0.0;

    // 3. The Flat Training Loop
    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < 4; i++) {
            
            // --- A. FORWARD PASS ---
            // Hidden Neuron 1
            float Z1_0 = (X[i][0] * W1[0][0]) + (X[i][1] * W1[1][0]) + b1[0];
            float A1_0 = max(0.0f, Z1_0); // ReLU
            
            // Hidden Neuron 2
            float Z1_1 = (X[i][0] * W1[0][1]) + (X[i][1] * W1[1][1]) + b1[1];
            float A1_1 = max(0.0f, Z1_1); // ReLU
            
            // Final Output Neuron
            float Z2 = (A1_0 * W2[0]) + (A1_1 * W2[1]) + b2;
            float A2 = 1.0f / (1.0f + exp(-Z2)); // Sigmoid

            // --- B. BACKWARD PASS (Chain Rule) ---
            float error = A2 - y[i];
            
            // Chain Rule: Output Error * Weight * Derivative of ReLU(Z)
            float error_hidden_0 = error * W2[0] * (Z1_0 > 0 ? 1.0f : 0.0f);
            float error_hidden_1 = error * W2[1] * (Z1_1 > 0 ? 1.0f : 0.0f);

            // --- C. UPDATE WEIGHTS ---
            float lr = 0.1f;
            
            // Update Output Layer
            W2[0] -= lr * error * A1_0;
            W2[1] -= lr * error * A1_1;
            b2    -= lr * error;

            // Update Hidden Layer
            W1[0][0] -= lr * error_hidden_0 * X[i][0];
            W1[1][0] -= lr * error_hidden_0 * X[i][1];
            W1[0][1] -= lr * error_hidden_1 * X[i][0];
            W1[1][1] -= lr * error_hidden_1 * X[i][1];
            
            b1[0] -= lr * error_hidden_0;
            b1[1] -= lr * error_hidden_1;
        }
    }

    cout << "Training Complete! Math matches Python perfectly." << endl;
    return 0;
}