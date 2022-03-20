#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


void Classfier(cv::Mat& image) {
    torch::Tensor img_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255);
    using torch::jit::script::Module;
    Module m = torch::jit::load("resnet.pt");
    torch::Tensor output = m.forward({ img_tensor }).toTensor();
    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    std::cout << max_index << std::endl;
}

int main() {
    cv::Mat image = cv::imread("dog.png");
    cv::resize(image, image, cv::Size(224, 224));
    std::cout << image.rows << " " << image.cols << " " << image.channels() << std::endl;
    Classfier(image);
    return 0;
}