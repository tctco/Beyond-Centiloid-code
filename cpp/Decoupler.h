#pragma once

#include <iostream>
#include <unordered_map>

#include "common.h"
#include "onnxruntime_cxx_api.h"

class DecoupledResult {
 public:
  ImageType::Pointer strippedImage;
  ImageType::Pointer strippedComponent;
  ImageType::Pointer ADprobMap;
  float ADprob;
  float fakeProb;
  float strippedADProb;
  float strippedFakeProb;
  float ADADscore;

  void SaveResults(const std::string& fpath);
  void printResult();
};

class Decoupler {
 public:
  Decoupler(const std::string& modelPath);
  ~Decoupler();
  std::unordered_map<std::string, std::vector<float>> _predict(
      std::vector<float> inputTensor);
  DecoupledResult predict(ImageType::Pointer inputImage);

 private:
  std::vector<int64_t> INPUT_SHAPE{1, 1, 160, 160, 96};
  std::vector<int64_t> IMAGE_SHAPE{160, 160, 96};
  std::size_t INPUT_TENSOR_SIZE = 160 * 160 * 96;
  Ort::Env env;
  Ort::Session* session;
};
