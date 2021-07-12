#include <CL/sycl.hpp>
#include <memory>
#include <iostream>

typedef std::shared_ptr<int> PtInt;
int SIZE = 1024;
int CONST = 322;

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        SIZE = atoi(argv[1]);
    }

    sycl::queue q;
    std::cout << q.get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "Work size = " << SIZE << '\n';

    int *a = sycl::malloc_shared<int>(SIZE, q);

    for(int i = 0; i < SIZE; i++) {
        a[i] = 0;
    }

    sycl::range<1> r(SIZE);

    auto e = q.parallel_for<class constant>(r, [=](auto i){ a[i] = CONST; });
    e.wait();

    for(int i = 0; i < SIZE; i++) {
        if (a[i] != CONST) {
            std::cerr << "ERROR: res[" << i << "] == " << a[i] << '\n';
            return 1;
        }
    }

    std::cout << "All ok\n";
    sycl::free(a, q);
    return 0;
}