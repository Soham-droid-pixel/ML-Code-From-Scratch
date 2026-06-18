#include <iostream>
#include <vector>
#include <cmath>     // Required for exp() and max()
#include <numeric>   // Required for accumulate()
#include <algorithm> // Required for max_element()

using namespace std;

// Function to compute Softmax probabilities from raw scores (logits)
vector<float> softmax(const vector<float>& logits) {
    int n = logits.size();
    vector<float> probabilities(n);

    // STEP 1: Find the maximum logit for numerical stability
    // This prevents exp(z) from overflowing memory if scores are massive
    float max_logit = *max_element(logits.begin(), logits.end());

    // STEP 2: Compute exponentiated values and their total sum
    float sum_exponents = 0.0;
    vector<float> exponents(n);
    
    for (int i = 0; i < n; ++i) {
        exponents[i] = exp(logits[i] - max_logit); // Stability shift applied here
        sum_exponents += exponents[i];
    }

    // STEP 3: Normalize every element to sum up to 1.0
    for (int i = 0; i < n; ++i) {
        probabilities[i] = exponents[i] / sum_exponents;
    }

    return probabilities;
}

int main() {
    // Input logits representing classes: [Jersey, Holstein, Gir]
    vector<float> logits = {3.2, 1.1, -0.5};

    vector<float> probabilities = softmax(logits);

    // Print calculated probability array
    cout << "Probabilities: [ ";
    for (float p : probabilities) {
        cout << p << " ";
    }
    cout << "]" << endl;

    // Automated Grader (with float tolerance check)
    float expected_jersey = 0.871;
    if (abs(probabilities[0] - expected_jersey) < 0.01) {
        cout << "Status: Passed" << endl;
    } else {
        cout << "Status: Failed" << endl;
    }

    return 0;
}