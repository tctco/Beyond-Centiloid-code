#include "Decoupler.h"

void DecoupledResult::SaveResults(const std::string& fpath) {
  Common::SaveImage(strippedImage,
                    Common::addSuffixToFilePath(fpath, "_stripped_image"));
  Common::SaveImage(strippedComponent, Common::addSuffixToFilePath(
                                           fpath, "_stripped_component.nii"));
  Common::SaveImage(ADprobMap,
                    Common::addSuffixToFilePath(fpath, "_AD_prob_map.nii"));
}

void DecoupledResult::printResult() {
  std::cout << "AI can make mistakes, please double check the results."
            << std::endl;
  std::cout << "AD probability: " << this->ADprob * 100 << "%" << std::endl;
  std::cout << "Fake probability: " << this->fakeProb * 100 << "%" << std::endl;
  std::cout << "Stripped AD probability: " << this->strippedADProb * 100 << "%" << std::endl;
  std::cout << "Stripped fake probability: " << this->strippedFakeProb * 100 << "%"
            << std::endl;
  std::cout << "ADAD score: " << this->ADADscore << std::endl;
}

Decoupler::Decoupler(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "Decouple"), session(nullptr) {
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
  try {
    this->session =
        new Ort::Session(this->env, w_modelPath.c_str(), sessionOptions);
    ;
  } catch (const Ort::Exception& e) {
    std::cerr << "Error loading mode: " << e.what() << std::endl;
    throw std::runtime_error("Filed to load model");
  }
}
Decoupler::~Decoupler() {
  if (this->session != nullptr) {
    delete this->session;
  }
}

std::unordered_map<std::string, std::vector<float>> Decoupler::_predict(
    std::vector<float> inputTensor) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  size_t inputTensorSize = inputTensor.size() * sizeof(float);
  Ort::Value inputTensorValue = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensor.data(), inputTensorSize, this->INPUT_SHAPE.data(),
      this->INPUT_SHAPE.size());
  const char* inputName = "PET";
  std::vector<const char*> outputNames = {
      "stripped_AD_images", "stripped_component",
      "AD_prob_map",        "AD_prob",
      "fake_prob",          "stripped_AD_prob",
      "stripped_fake_prob", "ADAD_scores"};
  std::vector<size_t> outputSizes = {
      INPUT_TENSOR_SIZE, INPUT_TENSOR_SIZE, INPUT_TENSOR_SIZE, 1, 1, 1, 1, 1};
  size_t numOutputs = outputNames.size();

  // Run inference

  std::vector<Ort::Value> outputTensorValues =
      this->session->Run(Ort::RunOptions{nullptr}, &inputName,
                         &inputTensorValue, 1, outputNames.data(), numOutputs);
  std::unordered_map<std::string, std::vector<float>> result;
  for (size_t i = 0; i < outputNames.size(); ++i) {
    std::string outputName(outputNames[i]);

    auto& outputTensor = outputTensorValues[i];
    float* outputData = outputTensor.GetTensorMutableData<float>();
    size_t outputSize = outputSizes[i];

    result[outputName].assign(outputData, outputData + outputSize);
  }
  return result;
}

DecoupledResult Decoupler::predict(ImageType::Pointer inputImage) {
  // Convert input image to vector
  std::vector<float> inputTensor;
  Common::ExtractImageData(inputImage, inputTensor);

  // Run inference
  std::unordered_map<std::string, std::vector<float>> outputTensors =
      this->_predict(inputTensor);

  // Convert output tensors to images
  DecoupledResult decoupledResult;
  auto size = inputImage->GetLargestPossibleRegion().GetSize();
  for (auto& [name, imgData] : outputTensors) {
    if (name == "stripped_AD_images") {
      ImageType::Pointer image = Common::CreateImageFromVector(imgData, size);
      image->SetOrigin(inputImage->GetOrigin());
      image->SetSpacing(inputImage->GetSpacing());
      image->SetDirection(inputImage->GetDirection());
      decoupledResult.strippedImage = image;
    } else if (name == "stripped_component") {
      ImageType::Pointer image =
          Common::CreateImageFromVector(imgData, {160, 160, 96});
      image->SetOrigin(inputImage->GetOrigin());
      image->SetSpacing(inputImage->GetSpacing());
      image->SetDirection(inputImage->GetDirection());
      decoupledResult.strippedComponent = image;
    } else if (name == "AD_prob_map") {
      ImageType::Pointer image =
          Common::CreateImageFromVector(imgData, {160, 160, 96});
      decoupledResult.ADprobMap = image;
      image->SetOrigin(inputImage->GetOrigin());
      image->SetSpacing(inputImage->GetSpacing());
      image->SetDirection(inputImage->GetDirection());
    } else if (name == "AD_prob") {
      decoupledResult.ADprob = imgData[0];
    } else if (name == "fake_prob") {
      decoupledResult.fakeProb = imgData[0];
    } else if (name == "stripped_AD_prob") {
      decoupledResult.strippedADProb = imgData[0];
    } else if (name == "stripped_fake_prob") {
      decoupledResult.strippedFakeProb = imgData[0];
    } else if (name == "ADAD_scores") {
      decoupledResult.ADADscore = imgData[0];
    } else {
      std::cerr << "Unknown output tensor: " << name << std::endl;
    }
  }

  return decoupledResult;
}