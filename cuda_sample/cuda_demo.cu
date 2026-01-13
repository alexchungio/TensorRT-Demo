#include "iostream"
#include "cuda.h"
#include "cuda_runtime.h"


__global__ void cuda_hello()
{
    std::cout << "hello world from GPU" << std::endl;
    
}

int main()
{
    cuda_hello<<<1, 1>>>();
    return 0;
}