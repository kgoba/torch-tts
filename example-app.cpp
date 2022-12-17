#include <torch/torch.h>

#include "net.h"

const char* kDataRoot = "./data"; // Where to find the MNIST dataset.
const int64_t kTrainBatchSize = 64; // The batch size for training.

int main() {
  // Create a new Net.
  auto net = std::make_shared<Net>();

  // Create a multi-threaded data loader for the MNIST dataset.

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           //.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());

  auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest);
                           // .map(torch::data::transforms::Stack<>());

  std::cout << "Train dataset size: " << train_dataset.size().value() << std::endl;
  std::cout << "Test dataset size: " << test_dataset.size().value() << std::endl;
  // auto train_loader = torch::data::make_data_loader(
  //     torch::data::datasets::MNIST("./data").map(
  //         torch::data::transforms::Stack<>()),
  //     /*batch_size=*/64);

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(train_dataset), kTrainBatchSize);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.03);

  for (size_t epoch = 1; epoch <= 100; ++epoch) {
    net->train();
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *train_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(batch.data);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(net, "net.pt");
      }
    }

    net->eval();
    torch::Tensor test_prediction = net->forward(test_dataset.images());
    // Compute a loss value to judge the prediction of our model.
    torch::Tensor test_loss = torch::nll_loss(test_prediction, test_dataset.targets());
    std::cout << "Epoch: " << epoch
              << " | Test loss: " << test_loss.item<float>() << std::endl;

  }
}
