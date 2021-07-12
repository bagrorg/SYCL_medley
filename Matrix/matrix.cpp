#include <CL/sycl.hpp>
#include <memory>
#include <iostream>
#include <random>
#include <time.h>

typedef std::shared_ptr<int> PtInt;
int SIZE = 1024;
int CONST = 322;

void result(std::vector<std::vector<int>> &a, 
            std::vector<std::vector<int>> &b, 
            std::vector<std::vector<int>> &ans) {
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            int sum = 0;

            for (int k = 0; k < SIZE; k++) {
                sum += a[i][k] * b[k][j];
            }

            ans[i][j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc >= 2) {
        SIZE = atoi(argv[1]);
    }

    sycl::queue q;
    std::cout << q.get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "Work size = " << SIZE << '\n';

    auto gen_mt = [] (int i) {
        return rand() % 10;
    };


    auto initialize_matrix = [&q, &gen_mt] (int **matrix) {
        for(int i = 0; i < SIZE; i++) {
            matrix[i] = sycl::malloc_shared<int>(SIZE, q);
            for (int j = 0; j < SIZE; j++) {
                matrix[i][j] = gen_mt(i);
            }
        }
    };

    int **a = sycl::malloc_shared<int *>(SIZE, q);
    int **b = sycl::malloc_shared<int *>(SIZE, q);
    int **c = sycl::malloc_shared<int *>(SIZE, q);
    
    

    initialize_matrix(a);
    initialize_matrix(b);
    initialize_matrix(c);

    sycl::range<2> r(SIZE, SIZE);

    auto e = q.parallel_for<class matrix>(r, [=](sycl::id<2> i){ 
        int sum = 0;
        int row = i[0];
        int col = i[1];

        for(int i = 0; i < r[0]; i++) {
            sum += a[row][i] * b[i][col];
        }

        c[row][col] = sum;
    });
    e.wait();

    std::vector<std::vector<int>> a1(SIZE, std::vector<int>(SIZE));
    std::vector<std::vector<int>> b1(SIZE, std::vector<int>(SIZE));
    std::vector<std::vector<int>> c1(SIZE, std::vector<int>(SIZE));
    
    for(int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            a1[i][j] = a[i][j];
            b1[i][j] = b[i][j];
        }
    }
    result(a1, b1, c1);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (c[i][j] != c1[i][j]) {
                std::cerr << "ERROR c[" << i << "][" << j << "] = " << c[i][j];
                return 1;
            }
        }
    }

    std::cout << "All ok!\n";

    if (SIZE < 10) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                std::cout << a[i][j] << ' ';
            }

            if (i == SIZE / 2) {
                std::cout << "  X   ";
            } else {
                std::cout << "      ";
            }

            for (int j = 0; j < SIZE; j++) {
                std::cout << b[i][j] << ' ';
            }

            if (i == SIZE / 2) {
                std::cout << "  =   ";
            } else {
                std::cout << "      ";
            }

            for (int j = 0; j < SIZE; j++) {
                std::cout << c[i][j] << ' ';
            }

            std::cout << '\n';
        }
    }

    auto free_mem = [&q] (int **a) {
        for (int i = 0; i < SIZE; i++) {
            sycl::free(a[i], q);
        }
        sycl::free(a, q);
    };
    
    free_mem(a);
    free_mem(b);
    free_mem(c);

    return 0;
}