#include <iostream>

#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include "torch/csrc/autograd/generated/VariableType.h"

int main(int argc, char* argv[]) {
    torch::TensorOptions option = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .requires_grad(true);

    torch::Tensor a = torch::rand({2, 2}, at::requires_grad(true));
    torch::Tensor b = torch::zeros({2, 2}, at::requires_grad(false));

    auto c = a.abs();
    auto d = b.neg();
    auto e = c + d;
    e.backward();


    std::cout << "----- a -----" << std::endl;
    std::cout << a.is_variable() << std::endl;
    std::cout << a.grad() << std::endl;

    std::cout << "----- b -----" << std::endl;
    std::cout << b.is_variable() << std::endl;
    std::cout << b.grad() << std::endl;

    std::cout << "----- c -----" << std::endl;
    std::cout << c.is_variable() << std::endl;
    std::cout << c.grad() << std::endl;

    std::cout << "----- d -----" << std::endl;
    std::cout << d.is_variable() << std::endl;
    std::cout << d.grad() << std::endl;

    std::cout << "----- e -----" << std::endl;
    std::cout << e.is_variable() << std::endl;
    std::cout << e.grad() << std::endl;

}