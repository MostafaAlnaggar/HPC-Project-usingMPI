#include <chrono>
#include <sstream>
#include <iomanip>
#include <direct.h>  // For Windows

#include "usingMPI.h"


// Function to get current timestamp
string getCurrentTimestamp() {
    auto now = chrono::system_clock::now();
    auto in_time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;

    // Using localtime_s for Windows
    struct tm timeinfo;
    localtime_s(&timeinfo, &in_time_t);
    ss << put_time(&timeinfo, "%Y%m%d_%H%M%S");

    return ss.str();
}

// Cross-platform directory creation function
bool createDirectory(const string& path) {
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
}


int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // to disable logging
     utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

    string inputFilename = "lena.png";
    Mat image;
    int imgRows, imgCols, imgType;

    // Only rank 0 loads the image
    if (rank == 0) {
        cout << "Number of threads is: " << size << endl;

        image = imread(inputFilename, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Could not open the image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cout << "Image has " << image.channels() << " channels." << endl;

        imgRows = image.rows;
        imgCols = image.cols;
        imgType = image.type();
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&imgRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgType, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create image in non-root processes
    if (rank != 0) {
        image = Mat(imgRows, imgCols, imgType);
    }

    // Broadcast image data to all processes
    MPI_Bcast(image.data, imgRows * imgCols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // input from user
    int kernel_size = 3;

    // Apply high-pass filter using MPI
    Mat sharpenedImage = MPIHighPassFilterRGB(image, kernel_size);

    // Only rank 0 handles display and saving
    if (rank == 0) {
        // Create output directory if it doesn't exist
        string outputDir = "sharpened_images";
        createDirectory(outputDir);

        // Extract original filename without extension
        size_t lastDot = inputFilename.find_last_of(".");
        string filenameWithoutExt = (lastDot != string::npos) ?
            inputFilename.substr(0, lastDot) :
            inputFilename;
        string extension = (lastDot != string::npos) ?
            inputFilename.substr(lastDot) :
            ".png";

        // Create output filename with timestamp
        string timestamp = getCurrentTimestamp();
        string outputFilename = outputDir + "/" + filenameWithoutExt + "_sharpened_mpi_" + timestamp + extension;

        // Display and save results
        imshow("Original", image);
        imshow("Sharpened", sharpenedImage);

        // Save the image
        if (imwrite(outputFilename, sharpenedImage)) {
            cout << "Image saved successfully: " << outputFilename << endl;
        }
        else {
            cerr << "Failed to save image: " << outputFilename << endl;
        }

        waitKey(0);
    }

    MPI_Finalize();
    return 0;
}
