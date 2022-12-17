#ifndef NET_H_INCLUDED
#define NET_H_INCLUDED

#include <torch/torch.h>

// Define a new Module.
struct Net : torch::nn::Module {
  Net();

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x);

  // Use one of many "standard library" modules.
  torch::nn::Conv2d cn1{nullptr};
  torch::nn::MaxPool2d mp1{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif // NET_H_INCLUDED