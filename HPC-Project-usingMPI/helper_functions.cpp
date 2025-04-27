#include "helper_functions.h"

// Applies kernel at a specific (x, y) location
int applyKernelAtPixel(const Mat& paddedImage, const vector<vector<int>>& kernel, int x, int y, int padding) {
    int sum = 0;
    for (int ky = -padding; ky <= padding; ky++) {
        for (int kx = -padding; kx <= padding; kx++) {
            int pixel = paddedImage.at<uchar>(y + ky, x + kx);
            int weight = kernel[ky + padding][kx + padding];
            sum += pixel * weight;
        }
    }

    return min(255, max(0, sum)); // Clamp
}


// Generates a high-pass kernel based on the specified size
vector<vector<int>> generateKernel(int k) {
    if (k % 2 == 0 || k < 3) {
        throw invalid_argument("Kernel size must be an odd number >= 3");
    }

    int center = k / 2;
    vector<vector<int>> kernel(k, vector<int>(k, 0));
    int sum = 0;

    for (int y = 0; y < k; ++y) {
        for (int x = 0; x < k; ++x) {
            int dx = abs(x - center);
            int dy = abs(y - center);
            int dist = dx + dy;

            if (dist == 0) continue; // center pixel, we'll set it later

            // Only fill in values along the cross (horizontal + vertical)
            if (dx == 0 || dy == 0) {
                int weight = -(center - dist + 1);
                kernel[y][x] = weight;
                sum += weight;
            }
        }
    }

    // Set the center value
    kernel[center][center] = -sum;

    return kernel;
}

void printKernel(const vector<vector<int>>& kernel) {
    cout << "Generated Kernel: " << endl;
    int k = kernel.size();
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            cout << setw(4) << kernel[i][j] << " ";
        }
        cout << endl;
    }
    cout << "\n------------------------\n\n";
}


// Modified kernel application function for RGB
int applyKernelAtPixelRGB(const Mat& paddedImage, const vector<vector<int>>& kernel,
    int centerX, int centerY, int padding, int channel) {
    int result = 0;
    for (int ky = 0; ky < kernel.size(); ky++) {
        for (int kx = 0; kx < kernel[ky].size(); kx++) {
            int pixelY = centerY + ky - padding;
            int pixelX = centerX + kx - padding;
            // Access specific channel using Vec3b
            result += paddedImage.at<Vec3b>(pixelY, pixelX)[channel] * kernel[ky][kx];
        }
    }
    return min(max(result, 0), 255);  // Ensure result is within valid range
}
