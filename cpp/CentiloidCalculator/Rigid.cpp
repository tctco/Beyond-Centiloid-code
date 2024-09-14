#include "Rigid.h"

std::vector<double> GenerateGaussianKernel(double sigma, int kernelRadius) {
  int size = 2 * kernelRadius + 1;
  std::vector<double> kernel(size);
  double sum = 0.0;
  for (int i = -kernelRadius; i <= kernelRadius; ++i) {
    kernel[i + kernelRadius] = std::exp(-0.5 * std::pow(i / sigma, 2));
    sum += kernel[i + kernelRadius];
  }
  // 归一化
  for (int i = 0; i < size; ++i) {
    kernel[i] /= sum;
  }
  return kernel;
}

ImageType::PointType getPhysicalPoint(const std::vector<float>voxelPoint,
  const ImageType::DirectionType& direction,
  const ImageType::PointType& origin,
  const ImageType::SpacingType& spacing) {
  ImageType::PointType physicalPoint;
  for (unsigned int i = 0; i < 3; i++) {
    physicalPoint[i] = origin[i];
    for (unsigned int j = 0; j < 3; j++) {
      physicalPoint[i] += direction[i][j] * voxelPoint[j] * spacing[j];
    }
  }
  return physicalPoint;
}

itk::Vector<double, 3> normalizeVector(const itk::Vector<double, 3>& vec) {
  double norm = vec.GetNorm();
  itk::Vector<double, 3> normalizedVec = vec;
  for (unsigned int i = 0; i < 3; ++i) {
    normalizedVec[i] /= norm;
  }
  return normalizedVec;
}

ImageType::Pointer Apply1DConvolution(ImageType::Pointer image, const std::vector<double>& kernel, unsigned int dimension) {
  using PixelType = ImageType::PixelType;

  ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetRegions(image->GetLargestPossibleRegion());
  outputImage->Allocate();
  outputImage->FillBuffer(0);

  itk::ImageRegionIterator<ImageType> inputIt(image, image->GetLargestPossibleRegion());
  itk::ImageRegionIterator<ImageType> outputIt(outputImage, outputImage->GetLargestPossibleRegion());

  int kernelRadius = kernel.size() / 2;
  for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++outputIt) {
    ImageType::IndexType index = inputIt.GetIndex();
    double sum = 0.0;
    for (int k = -kernelRadius; k <= kernelRadius; ++k) {
      ImageType::IndexType shiftedIndex = index;
      shiftedIndex[dimension] += k;
      if (image->GetLargestPossibleRegion().IsInside(shiftedIndex)) {
        sum += image->GetPixel(shiftedIndex) * kernel[k + kernelRadius];
      }
    }
    outputIt.Set(static_cast<PixelType>(sum));
  }
  return outputImage;
}

ImageType::Pointer CustomGaussianSmooth(ImageType::Pointer image, double sigma) {
  ImageType::Pointer smoothedImage = image;

  for (unsigned int dim = 0; dim < ImageType::ImageDimension; ++dim) {
    int kernelRadius = std::ceil(sigma * 3.0);
    std::vector<double> kernel = GenerateGaussianKernel(sigma, kernelRadius);
    smoothedImage = Apply1DConvolution(smoothedImage, kernel, dim);
  }
  return smoothedImage;
}

std::vector<double> GetSortedPixelValues(ImageType::Pointer image) {
  itk::ImageRegionIterator<ImageType> it(image, image->GetRequestedRegion());
  std::vector<double> pixelValues;

  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    pixelValues.push_back(it.Get());
  }

  std::sort(pixelValues.begin(), pixelValues.end());
  return pixelValues;
}

double GetPercentileValue(const std::vector<double> &sortedValues,
                          double percentile) {
  size_t index = static_cast<size_t>(percentile * (sortedValues.size() - 1));
  return sortedValues[index];
}

ImageType::Pointer ResampleImage(ImageType::Pointer image,
  const ImageType::SpacingType& newSpacing) {
  using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;

  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetInput(image);

  // 设置新的 spacing
  resampler->SetOutputSpacing(newSpacing);

  // 计算新的图像大小
  ImageType::SizeType inputSize = image->GetLargestPossibleRegion().GetSize();
  ImageType::SpacingType inputSpacing = image->GetSpacing();

  ImageType::SizeType newSize;
  for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
    newSize[i] = static_cast<unsigned int>((inputSize[i] * inputSpacing[i]) / newSpacing[i] + 0.5);
  }

  resampler->SetSize(newSize);
  resampler->SetOutputOrigin(image->GetOrigin());
  resampler->SetOutputDirection(image->GetDirection());
  resampler->Update();

  return resampler->GetOutput();
}

ImageType::Pointer ClipIntensityPercentiles(ImageType::Pointer image,
                                            double lowerPercentile,
                                            double upperPercentile) {
  auto sortedVoxelValue = GetSortedPixelValues(image);

  double lowerValue = GetPercentileValue(sortedVoxelValue, lowerPercentile);
  double upperValue = GetPercentileValue(sortedVoxelValue, upperPercentile);

  using IntensityWindowingImageFilterType =
      itk::IntensityWindowingImageFilter<ImageType, ImageType>;
  IntensityWindowingImageFilterType::Pointer windowingFilter =
      IntensityWindowingImageFilterType::New();
  windowingFilter->SetInput(image);
  windowingFilter->SetWindowMinimum(lowerValue);
  windowingFilter->SetWindowMaximum(upperValue);
  windowingFilter->SetOutputMinimum(lowerValue);
  windowingFilter->SetOutputMaximum(upperValue);
  windowingFilter->Update();

  using RescaleFilterType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;
  auto rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput(windowingFilter->GetOutput());
  rescaleFilter->SetOutputMinimum(0.0);
  rescaleFilter->SetOutputMaximum(1.0);
  rescaleFilter->Update();

  return rescaleFilter->GetOutput();
}

ImageType::Pointer GaussianSmooth(ImageType::Pointer image, double sigma) {
  using SmoothingFilterType =
    itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
  using BoundaryConditionType = itk::ConstantBoundaryCondition<ImageType>;
  SmoothingFilterType::Pointer smoothingFilter = SmoothingFilterType::New();
  BoundaryConditionType boundaryCondition;
  boundaryCondition.SetConstant(-0.1);
  smoothingFilter->SetInputBoundaryCondition(&boundaryCondition);
  smoothingFilter->SetMaximumKernelWidth(9);
  smoothingFilter->SetInput(image);
  smoothingFilter->SetUseImageSpacingOff();
  smoothingFilter->SetVariance(sigma * sigma);
  smoothingFilter->Update();
  return smoothingFilter->GetOutput();
}

ImageType::Pointer CropForeground(ImageType::Pointer image,
                                  float lowerThreshold) {
  // Step 1: Iterate through image to get bounding box
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::IndexType start = region.GetIndex();
  ImageType::SizeType size = region.GetSize();
  ImageType::IndexType minIndex = start;
  ImageType::IndexType maxIndex = start;
  bool initialized = false;

  itk::ImageRegionConstIterator<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    if (it.Get() > lowerThreshold) {
      ImageType::IndexType idx = it.GetIndex();
      if (!initialized) {
        minIndex = idx;
        maxIndex = idx;
        initialized = true;
      } else {
        for (unsigned int i = 0; i < 3; ++i) {
          if (idx[i] < minIndex[i]) minIndex[i] = idx[i];
          if (idx[i] > maxIndex[i]) maxIndex[i] = idx[i];
        }
      }
    }
  }

  if (!initialized) {
    std::cerr << "No foreground objects found!" << std::endl;
    return image;
  }

  // Step 2: Calculate bouding box
  ImageType::IndexType startIndex = minIndex;
  ImageType::SizeType roiSize;
  for (unsigned int i = 0; i < 3; ++i) {
    roiSize[i] = maxIndex[i] - minIndex[i] + 1;
  }

  ImageType::RegionType desiredRegion;
  desiredRegion.SetIndex(startIndex);
  desiredRegion.SetSize(roiSize);

  // Step 3: Crop the image using the bounding box
  using RegionOfInterestFilterType =
      itk::RegionOfInterestImageFilter<ImageType, ImageType>;
  RegionOfInterestFilterType::Pointer roiFilter =
      RegionOfInterestFilterType::New();
  roiFilter->SetInput(image);
  roiFilter->SetRegionOfInterest(desiredRegion);
  roiFilter->Update();

  return roiFilter->GetOutput();
}

ImageType::Pointer ResizeImage(ImageType::Pointer image,
                               const ImageType::SizeType &newSize) {
  using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();

  // 获取原始图像的尺寸和像素间距
  ImageType::SizeType originalSize =
      image->GetLargestPossibleRegion().GetSize();
  ImageType::SpacingType originalSpacing = image->GetSpacing();

  // 计算新的像素间距
  ImageType::SpacingType newSpacing;
  for (unsigned int i = 0; i < 3; ++i) {
    newSpacing[i] = originalSpacing[i] * static_cast<double>(originalSize[i]) /
                    static_cast<double>(newSize[i]);
  }

  // 配置重采样过滤器
  resampler->SetInput(image);
  resampler->SetSize(newSize);
  resampler->SetOutputSpacing(newSpacing);
  resampler->SetOutputOrigin(image->GetOrigin());
  resampler->SetOutputDirection(image->GetDirection());
  resampler->SetTransform(TransformType::New());
  resampler->SetInterpolator(
      itk::LinearInterpolateImageFunction<ImageType, double>::New());
  resampler->Update();

  return resampler->GetOutput();
}

ImageType::Pointer Rigid::preprocess(ImageType::Pointer image) {
  itk::Vector<double, 3> spacing;
  spacing.Fill(3.0);

  image = ClipIntensityPercentiles(image, 0.01, 0.99);
  image = GaussianSmooth(image, 1);
  image = ResampleImage(image, spacing);

  image = CropForeground(image, 0.35);

  itk::Size<3> outputSize;
  outputSize.Fill(64);
  image = ResizeImage(image, outputSize);
  return image;
}

Rigid::~Rigid() {
  if (session) delete session;
  if (voxelMorphSession) delete voxelMorphSession;
}

Rigid::Rigid(const std::string &modelPath, const std::string &voxelMorphPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "Rigid"), session(nullptr), voxelMorphSession(nullptr) {
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
  std::wstring w_voxelMorphPath = std::wstring(voxelMorphPath.begin(), voxelMorphPath.end());

  try {
    this->session = new Ort::Session(env, w_modelPath.c_str(), sessionOptions);
    this->voxelMorphSession = new Ort::Session(env, w_voxelMorphPath.c_str(), sessionOptions);
  } catch (const Ort::Exception &e) {
  
  std::cerr << "Error loading model:" << e.what() << std::endl;
    throw std::runtime_error("Failed to load model.");
  }
}

std::unordered_map<std::string, std::vector<float>> Rigid::predict(
  std::vector<float> inputTensor,
  const std::vector<int64_t> inputShape) {
  Ort::AllocatorWithDefaultOptions allocator;
  // Prepare input tensor
  // Input information
  auto input_name_allocated = session->GetInputNameAllocated(0, allocator);
  const char *input_name = input_name_allocated.get();
  // Create input tensor
  std::vector<int64_t> input_shape = {1, 1, 64, 64, 64};
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, inputTensor.data(), inputTensor.size(), input_shape.data(),
      input_shape.size());
  // Define output names
  const char *output_names[] = {"ac", "nose", "top"};
  size_t num_outputs = 3;

  // Run inference

  std::vector<Ort::Value> output_tensors;
  std::vector<std::vector<float>> output_buffers(num_outputs,
                                                 std::vector<float>(3));
  for (size_t i = 0; i < num_outputs; ++i) {
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, output_buffers[i].data(), output_buffers[i].size(),
        std::vector<int64_t>{1, 3}.data(), 2);
    output_tensors.push_back(std::move(output_tensor));
  }

  session->Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1,
              output_names, output_tensors.data(), 3);

  std::unordered_map<std::string, std::vector<float>> output;
  for (size_t i = 0; i < output_tensors.size(); i++) {
    Ort::TensorTypeAndShapeInfo output_info =
        output_tensors[i].GetTensorTypeAndShapeInfo();

    float *output_data = output_tensors[i].GetTensorMutableData<float>();
    std::vector<float> output_vector(
        output_data, output_data + output_info.GetElementCount());
    output[output_names[i]] = output_vector;
  }

  return output;
}

std::unordered_map<std::string, std::vector<float>> Rigid::predictVoxelMorph(
  std::vector<float> originalImg,
  std::vector<float> movingImg,
  std::vector<float> templateImg) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto inputNamesAllocated = session->GetInputNameAllocated(0, allocator);
  const char* inputNames[] = {"input", "template", "input_raw" };
  std::vector<int64_t> inputShape = { 1, 1, 79 + 17, 95 + 33, 79 + 17 };
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, movingImg.data(), movingImg.size(), inputShape.data(), inputShape.size()));
  inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, templateImg.data(), templateImg.size(), inputShape.data(), inputShape.size()));
  inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, originalImg.data(), originalImg.size(), inputShape.data(), inputShape.size()));
  const char* outputNames[] = { "warped" };
  size_t numOutputs = 1;

  std::vector<Ort::Value> outputTensors;
  std::vector<std::vector<float>> outputBuffers(numOutputs, std::vector<float>(1*3*96*128*96));
  for (size_t i = 0; i < numOutputs; ++i) {
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputBuffers[i].data(), outputBuffers[i].size(), std::vector<int64_t>{1, 1, 96, 128, 96}.data(), 5);
    outputTensors.push_back(std::move(outputTensor));
  }

  voxelMorphSession->Run(Ort::RunOptions{ nullptr }, inputNames, inputTensors.data(), 3, outputNames, outputTensors.data(), numOutputs);

  std::unordered_map<std::string, std::vector<float>> output;
  for (size_t i = 0; i < outputTensors.size(); i++) {
    Ort::TensorTypeAndShapeInfo outputInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
    float* outputData = outputTensors[i].GetTensorMutableData<float>();
    std::vector<float> outputVector(outputData, outputData + outputInfo.GetElementCount());
    output[outputNames[i]] = outputVector;
  }
  return output;
}

ImageType::Pointer Rigid::preprocessVoxelMorph(ImageType::Pointer image) {
  image = ClipIntensityPercentiles(image, 0.01, 0.99);
  return image;
}

std::tuple<ImageType::PointType, ImageType::DirectionType> Rigid::getNewOriginAndDirection(
  ImageType::Pointer image,
  std::vector<float> ac,
  std::vector<float> pa,
  std::vector<float> is
) {
  std::for_each(ac.begin(), ac.end(),
    [](float& x) { x *= 64; });
  std::for_each(pa.begin(), pa.end(),
    [](float& x) { x *= 99999; });
  std::for_each(is.begin(), is.end(),
    [](float& x) { x *= 99999; });
  const ImageType::DirectionType& direction = image->GetDirection();
  const ImageType::PointType& origin = image->GetOrigin();
  const ImageType::SpacingType& spacing = image->GetSpacing();

  // calculate ac nose top in physical space
  auto acPhysical = getPhysicalPoint(ac, direction, origin, spacing);
  auto nosePhysical = getPhysicalPoint(pa, direction, origin, spacing);
  auto topPhysical = getPhysicalPoint(is, direction, origin, spacing);
  itk::Vector<double, 3> noseVec, topVec;
  for (unsigned int i = 0; i < 3; i++) {
    noseVec[i] = nosePhysical[i] - acPhysical[i];
    topVec[i] = topPhysical[i] - acPhysical[i];
  }
  float projectionLength = 0;
  itk::Vector<double, 3> topNormalVec = normalizeVector(topVec);
  for (unsigned int i = 0; i < 3; i++) {
    projectionLength += noseVec[i] * topNormalVec[i];
  }
  for (unsigned int i = 0; i < 3; i++) noseVec[i] -= projectionLength * topNormalVec[i];
  itk::Vector<double, 3> noseNormalVec = normalizeVector(noseVec);
  itk::Vector<double, 3> orthoVec = itk::CrossProduct(noseNormalVec, topNormalVec);

  using TransformType = itk::AffineTransform<double, 3>;

  TransformType::MatrixType rotMat;
  rotMat[0][0] = -orthoVec[0]; rotMat[0][1] = -noseNormalVec[0]; rotMat[0][2] = topNormalVec[0];
  rotMat[1][0] = -orthoVec[1]; rotMat[1][1] = -noseNormalVec[1]; rotMat[1][2] = topNormalVec[1];
  rotMat[2][0] = -orthoVec[2]; rotMat[2][1] = -noseNormalVec[2]; rotMat[2][2] = topNormalVec[2];
  TransformType::MatrixType invRotMat = rotMat.GetTranspose();

  // update the image affine matrix

  ImageType::DirectionType newDirection = invRotMat * image->GetDirection();

  ImageType::PointType newOrigin = image->GetOrigin();
  for (unsigned int i = 0; i < 3; i++) {
    newOrigin[i] = newOrigin[i] - acPhysical[i];
  }
  itk::Vector<double, 3> originVec;
  for (unsigned int i = 0; i < 3; i++) {
    originVec[i] = newOrigin[i];
  }
  originVec = invRotMat * originVec;
  for (unsigned int i = 0; i < 3; i++) {
    newOrigin[i] = originVec[i];
  }
  return std::tuple<ImageType::PointType, ImageType::DirectionType>(newOrigin, newDirection);
}