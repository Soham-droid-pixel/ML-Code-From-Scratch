#include <iostream>
#include <vector>
#include <cmath> // Required for the exp() function

using namespace std;

// 1. The pure mathematical squish function
float sigmoid(float z) {
    return 1.0 / (1.0 + exp(-z));
}

void train_logistic_regression(const vector<vector<float>>& X, const vector<float>& y, int epochs, float learning_rate) {
    int n_samples = X.size();
    int n_features = X[0].size();
    
    // Initialize weights to 0
    vector<float> w(n_features, 0.0);
    float b = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        vector<float> dw(n_features, 0.0);
        float db = 0.0;
        float total_loss = 0.0;

        // Process every single data point manually
        for (int i = 0; i < n_samples; ++i) {
            
            // A. Forward Pass (Manual Dot Product)
            float z = b;
            for (int j = 0; j < n_features; ++j) {
                z += w[j] * X[i][j];
            }
            float y_pred = sigmoid(z);

            // B. Calculate Gradients (The Calculus)
            // The derivative of Binary Cross Entropy loss simplifies to (y_pred - y)
            float error = y_pred - y[i];
            
            for (int j = 0; j < n_features; ++j) {
                dw[j] += (1.0 / n_samples) * error * X[i][j];
            }
            db += (1.0 / n_samples) * error;
        }

        // C. Update Weights (Gradient Descent)
        for (int j = 0; j < n_features; ++j) {
            w[j] -= learning_rate * dw[j];
        }
        b -= learning_rate * db;

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " | w0: " << w[0] << " | b: " << b << endl;
        }
    }
}

int main() {
    // 2 samples, 2 features each (e.g., cell size, cell density)
    vector<vector<float>> X = {{2.5, 1.2}, {0.5, 0.2}}; 
    // Target classifications (1 = tumor, 0 = no tumor)
    vector<float> y = {1.0, 0.0}; 

    train_logistic_regression(X, y, 1000, 0.1);
    
    return 0;
}