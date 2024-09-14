﻿#include "CentiloidCalculator.h"
#include "Rigid.h"
#include <itkImageRegionConstIterator.h>
#include <algorithm>
#include <itkTranslationTransform.h>
#include <itkVector.h>
#include <itkWarpImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkCompositeTransform.h>
#include <itkDisplacementFieldTransform.h>
#include <itkComposeDisplacementFieldsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkVersion.h>
#include <filesystem> // C++17

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

std::string getExecutablePath() {
#ifdef _WIN32
  char buffer[MAX_PATH];
  GetModuleFileName(NULL, buffer, MAX_PATH);
  std::string executablePath(buffer);
#else
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  std::string executablePath(result, (count > 0) ? count : 0);
#endif
  // 移除文件名部分，只保留目录路径
  return std::filesystem::path(executablePath).parent_path().string();
}

double CalculateMeanInMask(ImageType::Pointer image, ImageType::Pointer mask) {
  using LabelStatisticsFilterType = itk::LabelStatisticsImageFilter<ImageType, ImageType>;
  LabelStatisticsFilterType::Pointer labelStatisticsFilter = LabelStatisticsFilterType::New();

  labelStatisticsFilter->SetInput(image);
  labelStatisticsFilter->SetLabelInput(mask);
  labelStatisticsFilter->Update();

  const unsigned char maskLabel = 1;
  if (labelStatisticsFilter->HasLabel(maskLabel)) {
    return labelStatisticsFilter->GetMean(maskLabel);
  }
  else {
    std::cerr << "Mask does not contain the specified label." << std::endl;
    return 0.0;
  }
}

void SaveImage(ImageType::Pointer image, const std::string &filename) {
  using WriterType = itk::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(filename);
  writer->SetInput(image);
  writer->Update();
}

ImageType::Pointer LoadImage(const std::string &filename) {
  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  reader->Update();
  return reader->GetOutput();
}

ImageType::Pointer ResampleToMatch(typename ImageType::Pointer referenceImage, typename ImageType::Pointer inputImage) {
  using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
  typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

  resampleFilter->SetInput(inputImage);
  resampleFilter->SetSize(referenceImage->GetLargestPossibleRegion().GetSize());
  resampleFilter->SetOutputSpacing(referenceImage->GetSpacing());
  resampleFilter->SetOutputOrigin(referenceImage->GetOrigin());
  resampleFilter->SetOutputDirection(referenceImage->GetDirection());

  using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  resampleFilter->SetInterpolator(interpolator);

  using TransformType = itk::AffineTransform<double, ImageType::ImageDimension>;
  typename TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  resampleFilter->SetTransform(transform);

  resampleFilter->Update();
  return resampleFilter->GetOutput();
}

void ExtractImageData(ImageType::Pointer image, std::vector<float>& imageData) {
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();

  imageData.resize(size[0] * size[1] * size[2]);
  for (size_t z = 0; z < size[0]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[2]; ++x) {
        ImageType::IndexType index = { {x, y, z} };
        float pixelValue = image->GetPixel(index);
        imageData[x * size[0] * size[1] + y * size[0] + z] = pixelValue;
      }
    }
  }
}

ImageType::Pointer CropMNI(const ImageType::Pointer img) {
  ImageType::RegionType cropRegion;
  ImageType::RegionType::IndexType start;
  start[0] = (96 - 79) / 2;  // x起始索引
  start[1] = (128 - 95) / 2; // y起始索引
  start[2] = (96 - 79) / 2;  // z起始索引

  ImageType::RegionType::SizeType size;
  size[0] = 79; // x尺寸
  size[1] = 95; // y尺寸
  size[2] = 79; // z尺寸
  cropRegion.SetSize(size);
  cropRegion.SetIndex(start);
  using FilterType = itk::RegionOfInterestImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetRegionOfInterest(cropRegion);
  filter->SetInput(img);
  filter->Update();
  return filter->GetOutput();
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <inputPath> <outputPath>"
              << std::endl;
    return EXIT_FAILURE;
  }
  std::string inputPath = argv[1];
  std::string outputPath = argv[2];
  std::string executableDir = getExecutablePath();
  try {
    ImageType::Pointer image = LoadImage(inputPath);
    Rigid rigidModel = Rigid(executableDir + "/2head-pib-noise-gelu-64channel.onnx", executableDir + "/affine_voxelmorph.onnx");
    image = rigidModel.preprocess(image);
    SaveImage(image, executableDir + "/preprocessed.nii");  
    
    ImageType::RegionType region = image->GetLargestPossibleRegion();

    ImageType::SizeType size = region.GetSize();
    std::vector<float>imageData;
    imageData.reserve(image->GetLargestPossibleRegion().GetNumberOfPixels());
    itk::ImageRegionIterator<ImageType> imageIterator(
        image, image->GetLargestPossibleRegion());
    for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator) {
      imageData.push_back(imageIterator.Get());
    }
    ExtractImageData(image, imageData);
    std::unordered_map<std::string, std::vector<float>>orientation = rigidModel.predict(imageData,
                       std::vector<int64_t>{1, 1, 64, 64, 64});
    ImageType::Pointer newImage = LoadImage(inputPath);
    std::tuple<ImageType::PointType, ImageType::DirectionType> newOriginAndDirection = rigidModel.getNewOriginAndDirection(
      newImage, 
      orientation["ac"], 
      orientation["nose"], 
      orientation["top"]
    );
    ImageType::PointType newOrigin = std::get<0>(newOriginAndDirection);
    ImageType::DirectionType newDirection = std::get<1>(newOriginAndDirection);
    newImage->SetDirection(newDirection);
    newImage->SetOrigin(newOrigin);

    // save the new image
    SaveImage(newImage, outputPath);

    ImageType::Pointer paddedTemplate = LoadImage(executableDir + "/paddedTemplate.nii");
    ImageType::Pointer paddedImage = ResampleToMatch(paddedTemplate, newImage);
    SaveImage(paddedImage, executableDir + "/padded.nii");
    paddedImage = rigidModel.preprocessVoxelMorph(paddedImage);
    ImageType::Pointer paddedOriginalImage = LoadImage(executableDir + "/padded.nii");

    std::vector<float> paddedImageData, paddedTemplateData, paddedOriginalData;
    ExtractImageData(paddedImage, paddedImageData);
    ExtractImageData(paddedTemplate, paddedTemplateData);
    ExtractImageData(paddedOriginalImage, paddedOriginalData);
    std::unordered_map<std::string, std::vector<float>> voxelMorphDDF = rigidModel.predictVoxelMorph(
      paddedOriginalData, 
      paddedImageData, 
      paddedTemplateData
    );
    paddedImage = CropMNI(paddedImage);
    SaveImage(paddedImage, outputPath);

    std::string voiTemplatePath = executableDir + "/voi_ctx_2mm.nii";
    std::string refTemplatePath = executableDir + "/voi_WhlCbl_2mm.nii";
    ImageType::Pointer voiTemplate = LoadImage(voiTemplatePath);
    ImageType::Pointer refTemplate = LoadImage(refTemplatePath);
    ImageType::Pointer resampledImage = ResampleToMatch(voiTemplate, paddedImage);
    double meanVoi = CalculateMeanInMask(resampledImage, voiTemplate);
    double meanRef = CalculateMeanInMask(resampledImage, refTemplate);
    std::cout << "meanVOI: " << meanVoi << std::endl;
    std::cout << "meanRef: " << meanRef << std::endl;
    std::cout << "SUVr: " << meanVoi / meanRef << std::endl;
    std::cout << "PiB: " << meanVoi / meanRef * 93.7 - 94.6 << std::endl;
    std::cout << "florbetapir / AV45: " << meanVoi / meanRef * 175.4 -182.3 << std::endl;
    std::cout << "florbetaben: " << meanVoi / meanRef * 153.4 -154.9 << std::endl;
    std::cout << "flutemetamol: " << meanVoi / meanRef * 93.7 - 94.6 << std::endl;
  }
  catch (itk::ExceptionObject& err) {
    std::cerr << "Exception caught in main: " << err << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}