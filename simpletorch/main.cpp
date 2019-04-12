#include <iostream>

#include <torch/extension.h>

int main(int argc, char* argv[]) {
    torch::Tensor a = torch::ones({2, 2});
    torch::Tensor b = torch::randn({2, 2});

    auto c = a + b;
    c.backward();


    std::cout<< a.grad() <<std::endl;
}