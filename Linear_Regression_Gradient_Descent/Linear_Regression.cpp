#include <iostream>
#include <vector>

using namespace std;

void train_linear_regression(const vector<float>& x, const vector<float>& y, int epochs, float learning_rate) {
    float w = 0.0; // Weight (slope)
    float b = 0.0; // Bias (intercept)
    int n = x.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float dw = 0.0; // Derivative of weight
        float db = 0.0; // Derivative of bias
        float total_loss = 0.0;

        // Manually calculate the gradients across all data points
        for (int i = 0; i < n; ++i) {
            float y_pred = w * x[i] + b;         // Forward pass
            float error = y_pred - y[i];         // Calculate error
            total_loss += error * error;         // Mean Squared Error (MSE)
            
            dw += (2.0 / n) * x[i] * error;      // Chain rule for weight
            db += (2.0 / n) * error;             // Chain rule for bias
        }

        // Update the weights (Gradient Descent)
        w -= learning_rate * dw;
        b -= learning_rate * db;

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " | Loss: " << total_loss / n << " | w: " << w << " | b: " << b << endl;
        }
    }
}

int main() {
    vector<float> x = {1.0, 2.0, 3.0, 4.0};
    vector<float> y = {2.0, 4.0, 6.0, 8.0}; // Perfect line y = 2x
    train_linear_regression(x, y, 1000, 0.01);
    return 0;
}