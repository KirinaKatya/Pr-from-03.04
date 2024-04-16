#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;

int mandelbrot(const complex<double>& c, int max_iterations) {
    complex<double> z = 0;
    int iterations = 0;
    while (abs(z) < 2 && iterations < max_iterations) {
        z = z * z + c;
        iterations++;
    }
    return iterations;
}

int main() {
    const int WIDTH = 800; 
    const int HEIGHT = 600; 
    const int MAX_ITERATIONS = 1000; 

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows_per_process = HEIGHT / size;

    int* data = new int[WIDTH * rows_per_process]; 

    for (int row = rank * rows_per_process; row < (rank + 1) * rows_per_process; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            double x = (col - WIDTH / 2.0) * 4.0 / WIDTH; 
            double y = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;

            complex<double> c(x, y);
            data[(row - rank * rows_per_process) * WIDTH + col] = mandelbrot(c, MAX_ITERATIONS); 
        }
    }

    int* recv_buffer = NULL;
    if (rank == 0) {
        recv_buffer = new int[WIDTH * HEIGHT];
    }

    MPI_Gather(data, WIDTH * rows_per_process, MPI_INT, recv_buffer, WIDTH * rows_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Mat image(HEIGHT, WIDTH, CV_8UC3);
        for (int row = 0; row < HEIGHT; ++row) {
            for (int col = 0; col < WIDTH; ++col) {
                int iterations = recv_buffer[row * WIDTH + col];
                if (iterations == MAX_ITERATIONS) {
                    image.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                }
                else {
                    image.at<Vec3b>(row, col) = Vec3b((iterations % 17 + 1) / 2.0 * 255, 255, (iterations % 3 + 1) / 5.5 * 255);;
                }
            }
        }
        imshow("Mandelbrot Fractal", image);
        waitKey(0);
    }

    delete[] data;
    if (rank == 0) {
        delete[] recv_buffer;
    }

    MPI_Finalize();

    return 0;
}
