#include "CentiloidCalculator.h"

#include <itkComposeDisplacementFieldsImageFilter.h>
#include <itkCompositeTransform.h>
#include <itkDisplacementFieldTransform.h>
#include <itkImageRegionConstIterator.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMaskImageFilter.h>
#include <itkTranslationTransform.h>
#include <itkVector.h>
#include <itkVersion.h>
#include <itkWarpImageFilter.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <filesystem>  // C++17

#include "Rigid.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
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
  using LabelStatisticsFilterType =
      itk::LabelStatisticsImageFilter<ImageType, ImageType>;
  LabelStatisticsFilterType::Pointer labelStatisticsFilter =
      LabelStatisticsFilterType::New();

  labelStatisticsFilter->SetInput(image);
  labelStatisticsFilter->SetLabelInput(mask);
  labelStatisticsFilter->Update();

  const unsigned char maskLabel = 1;
  if (labelStatisticsFilter->HasLabel(maskLabel)) {
    return labelStatisticsFilter->GetMean(maskLabel);
  } else {
    std::cerr << "Mask does not contain the specified label." << std::endl;
    return 0.0;
  }
}

void SaveImage(ImageType::Pointer image, const std::string& filename) {
  using WriterType = itk::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(filename);
  writer->SetInput(image);
  writer->Update();
}

ImageType::Pointer LoadNii(const std::string& filename) {
  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  reader->Update();
  return reader->GetOutput();
}

ImageType::Pointer ResampleToMatch(typename ImageType::Pointer referenceImage,
                                   typename ImageType::Pointer inputImage) {
  using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
  typename ResampleFilterType::Pointer resampleFilter =
      ResampleFilterType::New();

  resampleFilter->SetInput(inputImage); 
  resampleFilter->SetSize(referenceImage->GetLargestPossibleRegion().GetSize());
  resampleFilter->SetOutputSpacing(referenceImage->GetSpacing());
  resampleFilter->SetOutputOrigin(referenceImage->GetOrigin());
  resampleFilter->SetOutputDirection(referenceImage->GetDirection());

  using InterpolatorType =
      itk::LinearInterpolateImageFunction<ImageType, double>;
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
        ImageType::IndexType index = {{x, y, z}};
        float pixelValue = image->GetPixel(index);
        imageData[x * size[0] * size[1] + y * size[0] + z] = pixelValue;
      }
    }
  }
}

ImageType::Pointer CreateImageFromVector(const std::vector<float>& imageData, ImageType::SizeType size) {
  ImageType::Pointer image = ImageType::New();
  ImageType::IndexType start = { {0, 0, 0} };
  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);
  for (size_t z = 0; z < size[0]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[2]; ++x) {
        ImageType::IndexType index = { {x, y, z} };
        size_t vectorIndex = x * size[0] * size[1] + y * size[0] + z;
        image->SetPixel(index, imageData[vectorIndex]);
      }
    }
  }

  return image;
}

ImageType::Pointer CropMNI(const ImageType::Pointer img) {
  ImageType::RegionType cropRegion;
  ImageType::RegionType::IndexType start;
  start[0] = (96 - 79) / 2;   // x起始索引
  start[1] = (128 - 95) / 2;  // y起始索引
  start[2] = (96 - 79) / 2;   // z起始索引

  ImageType::RegionType::SizeType size;
  size[0] = 79;  // x尺寸
  size[1] = 95;  // y尺寸
  size[2] = 79;  // z尺寸
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
  argparse::ArgumentParser program("CentiloidCalculator");
  program.add_argument("inputPath").help("Input image path");
  program.add_argument("outputPath")
      .help("Output (spatially normalized) PET image path");
  program.add_argument("-m", "--manual-fov-placement")
      .help(
          "Use manual field of view (FOV) placement by skipping the deep "
          "learning based rigid transformation."
          " You can use this option when automatic spatial normalization "
          "fails.")
      .default_value(false)
      .implicit_value(true);
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(EXIT_FAILURE);
  }
  std::string inputPath = program.get<std::string>("inputPath");
  std::string outputPath = program.get<std::string>("outputPath");
  bool manualFovPlacement = program.get<bool>("--manual-fov-placement");
  std::string executableDir = getExecutablePath();
  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  try {
    ImageType::Pointer newImage{nullptr};
    Rigid rigidModel =
        Rigid(executableDir + "/2head-pib-noise-gelu-64channel.onnx",
              executableDir + "/affine_voxelmorph.onnx");
    if (!manualFovPlacement) {
      ImageType::Pointer image = LoadNii(inputPath);
      image = rigidModel.preprocess(image);
      SaveImage(image, executableDir + "/preprocessed.nii");

      ImageType::RegionType region = image->GetLargestPossibleRegion();

      ImageType::SizeType size = region.GetSize();
      std::vector<float> imageData;
      imageData.reserve(image->GetLargestPossibleRegion().GetNumberOfPixels());
      itk::ImageRegionIterator<ImageType> imageIterator(
          image, image->GetLargestPossibleRegion());
      for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd();
           ++imageIterator) {
        imageData.push_back(imageIterator.Get());
      }
      ExtractImageData(image, imageData);
      std::unordered_map<std::string, std::vector<float>> orientation =
          rigidModel.predict(imageData, std::vector<int64_t>{1, 1, 64, 64, 64});
      newImage = LoadNii(inputPath);
      std::tuple<ImageType::PointType, ImageType::DirectionType>
          newOriginAndDirection = rigidModel.getNewOriginAndDirection(
              image, newImage, orientation["ac"], orientation["nose"],
              orientation["top"]);
      ImageType::PointType newOrigin = std::get<0>(newOriginAndDirection);
      ImageType::DirectionType newDirection =
          std::get<1>(newOriginAndDirection);
      newImage->SetDirection(newDirection);
      newImage->SetOrigin(newOrigin);

      // save the new image
      SaveImage(newImage, executableDir + "/rigid.nii");
    } else {
      newImage = LoadNii(inputPath);
      SaveImage(newImage, executableDir + "/rigid.nii");
    }

    ImageType::Pointer paddedTemplate =
        LoadNii(executableDir + "/paddedTemplate.nii");
    ImageType::Pointer paddedImage = ResampleToMatch(paddedTemplate, newImage);
    SaveImage(paddedImage, executableDir + "/padded.nii");
    paddedImage = rigidModel.preprocessVoxelMorph(paddedImage);
    SaveImage(paddedImage, executableDir + "/paddedProcessed.nii");
    ImageType::Pointer paddedOriginalImage =
        LoadNii(executableDir + "/padded.nii");

    std::vector<float> paddedImageData, paddedTemplateData, paddedOriginalData;
    ExtractImageData(paddedImage, paddedImageData);
    ExtractImageData(paddedTemplate, paddedTemplateData);
    ExtractImageData(paddedOriginalImage, paddedOriginalData);
    std::unordered_map<std::string, std::vector<float>> warpedImageData =
        rigidModel.predictVoxelMorph(paddedOriginalData, paddedImageData,
                                     paddedTemplateData);
    ImageType::Pointer warpedImage = CreateImageFromVector(warpedImageData["warped"], paddedImage->GetLargestPossibleRegion().GetSize());
    warpedImage->SetDirection(paddedTemplate->GetDirection());
    warpedImage->SetOrigin(paddedTemplate->GetOrigin());
    warpedImage->SetSpacing(paddedTemplate->GetSpacing());
    warpedImage = CropMNI(warpedImage);
    SaveImage(warpedImage, outputPath);

    std::string voiTemplatePath = executableDir + "/voi_ctx_2mm.nii";
    std::string refTemplatePath = executableDir + "/voi_WhlCbl_2mm.nii";
    ImageType::Pointer voiTemplate = LoadNii(voiTemplatePath);
    ImageType::Pointer refTemplate = LoadNii(refTemplatePath);
    ImageType::Pointer resampledImage =
        ResampleToMatch(voiTemplate, warpedImage);
    double meanVoi = CalculateMeanInMask(resampledImage, voiTemplate);
    double meanRef = CalculateMeanInMask(resampledImage, refTemplate);
    std::cout << "meanVOI: " << meanVoi << std::endl;
    std::cout << "meanRef: " << meanRef << std::endl;
    std::cout << "SUVr: " << meanVoi / meanRef << std::endl;
    std::cout << "PiB: " << meanVoi / meanRef * 93.7 - 94.6 << std::endl;
    std::cout << "Florbetapir / FBP / AV45: " << meanVoi / meanRef * 175.4 - 182.3
              << std::endl;
    std::cout << "Florbetaben / FBB: " << meanVoi / meanRef * 153.4 - 154.9
              << std::endl;
    std::cout << "Flutemetamol / FBP: " << meanVoi / meanRef * 93.7 - 94.6
              << std::endl;
  } catch (itk::ExceptionObject& err) {
    std::cerr << "Exception caught in main: " << err << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
