#include <iostream>

using namespace std;

int main() {
    int image[4][4] = {
        {1, 2, 3, 0},
        {0, 1, 2, 3},
        {3, 0, 1, 2},
        {2, 3, 0, 1}
    };

    int kernel[3][3] = {
        { 1,  0, -1},
        { 1,  0, -1},
        { 1,  0, -1}
    };

    int output[2][2] = {0};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int current_sum = 0;
            for (int m = 0; m < 3; m++) {
                for (int n = 0; n < 3; n++) {
                    current_sum += image[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = current_sum;
        }
    }

    cout << "C++ 2D Convolution Output:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << output[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}