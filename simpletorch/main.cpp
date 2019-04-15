#include <iostream>

#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>

int main(int argc, char* argv[]) {
    torch::TensorOptions option = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .layout(torch::kStrided)
                                    .requires_grad(true);
    torch::Tensor a = torch::ones({2, 2}, option);
    torch::Tensor b = torch::zeros({2, 2}, option);

    auto c = a.abs() + b;
    auto d = c.neg();
    d.backward();

    auto cv = static_cast<torch::autograd::Variable>(c);
    auto av = static_cast<torch::autograd::Variable>(a);
    auto dv = static_cast<torch::autograd::Variable>(d);

    auto edge = cv.gradient_edge();
    auto func = dv.grad_fn_unsafe();
    auto deg = func->next_edge(0);

    std::cout<< "----------a" << std::endl;
    std::cout<< a.is_variable()<<std::endl;
    std::cout<< av.grad_fn_unsafe()<<std::endl;
    std::cout<< av.grad_accumulator()<<std::endl;

    std::cout<< "----------c" << std::endl;
    std::cout<< c.is_variable()<<std::endl;
    std::cout<< c.grad().is_variable()<<std::endl;
    std::cout<< cv.grad_fn_unsafe()->name()<<std::endl;
    std::cout<< cv.grad_fn_unsafe()->num_outputs() <<std::endl;
    std::cout<< cv.grad_fn_unsafe()->num_inputs() <<std::endl;
    //std::cout<< cv.grad_accumulator()<<std::endl;

    std::cout<< "----------d" << std::endl;
    std::cout<< dv.is_variable()<<std::endl;
    std::cout<< dv.grad_fn_unsafe()<<std::endl;
    std::cout<< dv.grad_fn_unsafe()->name()<<std::endl;
    std::cout<< dv.grad_accumulator()<<std::endl;

}