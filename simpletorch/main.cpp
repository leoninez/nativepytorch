#include <iostream>

#include <torch/extension.h>

int main(int argc, char* argv[]) {
    torch::TensorOptions option = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .layout(torch::kStrided)
                                    .requires_grad(true);
    torch::Tensor a = torch::ones({2, 2}, option);
    torch::Tensor b = torch::randn({2, 2});

    auto c = a + b;
    c.backward();


    std::cout<< a.grad() <<std::endl;
}