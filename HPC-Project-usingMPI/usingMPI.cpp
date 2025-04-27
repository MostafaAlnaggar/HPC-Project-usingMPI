#include "usingMPI.h"
#include "helper_functions.h"

// MPI High-Pass Filter Function
Mat MPIHighPassFilterRGB(const Mat& inputImage, int kernelSize) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (kernelSize % 2 == 0 || kernelSize < 3) {
        if (rank == 0) {
            cerr << "Kernel size must be odd and >= 3." << endl;
        }
        return inputImage;
    }

    int padding = (kernelSize - 1) / 2;
    vector<vector<int>> kernel = generateKernel(kernelSize);

    if (rank == 0) {
        printKernel(kernel);
    }

    // Pad input image
    Mat paddedImage;
    copyMakeBorder(inputImage, paddedImage, padding, padding, padding, padding, BORDER_REPLICATE);

    // Prepare output
    Mat outputImage = Mat::zeros(inputImage.size(), inputImage.type());

    // Calculate rows per process
    int totalRows = inputImage.rows;
    int rowsPerProcess = totalRows / size;
    int remainingRows = totalRows % size;

    // Create arrays for scatter/gather
    int* sendcounts = nullptr;
    int* displs = nullptr;

    // Only rank 0 needs these arrays
    if (rank == 0) {
        sendcounts = new int[size];
        displs = new int[size];

        for (int i = 0; i < size; i++) {
            sendcounts[i] = (rowsPerProcess + (i < remainingRows ? 1 : 0)) * inputImage.cols * 3;
            displs[i] = 0;
            if (i > 0) {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
        }
    }

    // Calculate local rows for this process
    int localRows = rowsPerProcess + (rank < remainingRows ? 1 : 0);
    int startRow = rank * rowsPerProcess + min(rank, remainingRows);

    // Create local buffer for processed data
    vector<uchar> localBuffer(localRows * inputImage.cols * 3);

    // Start timing
    double startTime = MPI_Wtime();

    // Process local rows
    for (int y = 0; y < localRows; y++) {
        int globalY = startRow + y;
        for (int x = 0; x < inputImage.cols; x++) {
            Vec3b result;
            for (int c = 0; c < 3; c++) {
                int channelResult = applyKernelAtPixelRGB(paddedImage, kernel,
                    x + padding,
                    globalY + padding,
                    padding, c);
                result[c] = static_cast<uchar>(channelResult);
            }
            // Store in local buffer
            int bufferIdx = (y * inputImage.cols + x) * 3;
            localBuffer[bufferIdx] = result[0];
            localBuffer[bufferIdx + 1] = result[1];
            localBuffer[bufferIdx + 2] = result[2];
        }
    }

    // Gather results from all processes
    MPI_Gatherv(localBuffer.data(), localRows * inputImage.cols * 3, MPI_UNSIGNED_CHAR,
        outputImage.data, sendcounts, displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // End timing
    double endTime = MPI_Wtime();

    if (rank == 0) {
        double totalTime = (endTime - startTime) * 1000.0;
        cout << "Execution time for MPI code: " << totalTime << " milliseconds" << endl;
        delete[] sendcounts;
        delete[] displs;

    }

    return outputImage;
}
