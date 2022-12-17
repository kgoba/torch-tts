#include "net.h"

constexpr int kConvChannels = 5;

// Define a new Module.
Net::Net()
{
    // Construct and register two Linear submodules.
    cn1 = register_module("cn1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, kConvChannels, 5).dilation(2) // .padding(2)
        ));
    mp1 = register_module("mp1", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2)
        ));
    // fc1 = register_module("fc1", torch::nn::Linear(3*6*6, 128));
    fc2 = register_module("fc2", torch::nn::Linear(kConvChannels*10*10, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, 10));
}

// Implement the Net's algorithm.
torch::Tensor Net::forward(torch::Tensor x)
{
    // Use one of many tensor manipulation functions.
    auto y = torch::gelu(mp1->forward(cn1->forward(x)));
    // std::cout << x.sizes() << " -> " << y.sizes() << std::endl;
    x = y.reshape({y.size(0), kConvChannels*10*10});
    // x = torch::gelu(fc1->forward(y));
    x = torch::gelu(fc2->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
}
