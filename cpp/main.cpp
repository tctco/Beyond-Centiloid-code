#include <itkComposeDisplacementFieldsImageFilter.h>
#include <itkCompositeTransform.h>
#include <itkDisplacementFieldTransform.h>
#include <itkImageRegionConstIterator.h>
#include <itkMaskImageFilter.h>
#include <itkTranslationTransform.h>
#include <itkVector.h>
#include <itkVersion.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <filesystem>  // C++17
#include <iostream>

#include "Calculator.h"
#include "Decoupler.h"
#include "Rigid.h"
#include "common.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

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

ImageType::Pointer rigidAlign(const std::string& inputPath,
                              const std::string& executableDir,
                              Rigid& rigidModel,
                              ImageType::Pointer templateImage,
                              bool resampleFirst = false) {
  ImageType::Pointer image = Common::LoadNii(inputPath);
  if (resampleFirst) {
    image = Common::ResampleToMatch(templateImage, image);
  }
  image = rigidModel.preprocess(image);
  Common::SaveImage(image, executableDir + "/preprocessed.nii");

  ImageType::RegionType region = image->GetLargestPossibleRegion();

  ImageType::SizeType size = region.GetSize();
  std::vector<float> imageData;
  imageData.reserve(image->GetLargestPossibleRegion().GetNumberOfPixels());
  itk::ImageRegionIterator<ImageType> imageIterator(
      image, image->GetLargestPossibleRegion());
  for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator) {
    imageData.push_back(imageIterator.Get());
  }
  Common::ExtractImageData(image, imageData);
  std::unordered_map<std::string, std::vector<float>> orientation =
      rigidModel.predict(imageData, std::vector<int64_t>{1, 1, 64, 64, 64});
  ImageType::Pointer newImage = Common::LoadNii(inputPath);
  std::tuple<ImageType::PointType, ImageType::DirectionType>
      newOriginAndDirection = rigidModel.getNewOriginAndDirection(
          image, newImage, orientation["ac"], orientation["nose"],
          orientation["top"]);
  ImageType::PointType newOrigin = std::get<0>(newOriginAndDirection);
  ImageType::DirectionType newDirection = std::get<1>(newOriginAndDirection);
  newImage->SetDirection(newDirection);
  newImage->SetOrigin(newOrigin);
  return newImage;
}

ImageType::Pointer iterativeRigidAlign(const std::string& inputPath,
                                       const std::string& executableDir,
                                       Rigid& rigidModel, int maxIter,
                                       float acDiffThreshhold,
                                       ImageType::Pointer templateImage) {
  ImageType::Pointer newImage =
      rigidAlign(inputPath, executableDir, rigidModel, templateImage, false);
  ImageType::PointType lastOrigin = newImage->GetOrigin();
  Common::SaveImage(newImage, executableDir + "/rigid.nii");
  for (int i = 0; i < maxIter; ++i) {
    newImage = rigidAlign(executableDir + "/rigid.nii", executableDir,
                          rigidModel, templateImage, true);
    float originShift = 0;
    for (int j = 0; j < 3; ++j) {
      originShift += std::pow(newImage->GetOrigin()[j] - lastOrigin[j], 2);
    }
    originShift = std::sqrt(originShift);
    if (originShift < acDiffThreshhold) {
      break;
    }
  }
  return newImage;
}

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("CentiloidCalculator", "2.0.0");
  program.add_argument("inputPath").help("Input image path");
  program.add_argument("outputPath")
      .help("Output (spatially normalized) PET image path");
  program.add_argument("--ADNI-PET-core")
      .help(
          "Automatically perform and output ADNI PET core style rigid "
          "registration, intensity normalization, and resolution adjustment "
          "(Co-reg, Avg, Standardized Image and Voxel Size). See official ADNI "
          "PET preprocessing webpage for details. Note that we use Centiloid "
          "Project's cerebellar gray matter for the reference region.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-i", "--iterative")
      .help(
          "Use iterative rigid transformation to improve the accuracy of the "
          "spatial normalization. This option is useful when the automatic "
          "spatial normalization fails. This may costs a little more time.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-m", "--manual-fov-placement")
      .help(
          "Use manual field of view (FOV) placement by skipping the deep "
          "learning based rigid transformation."
          " You can use this option when automatic spatial normalization "
          "fails.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-d", "--decouple")
    .help("Decouple the input image to extract AD related components. You need to specify the modality of the input image, which is necessary for image "
      "decouple. Currently we only support Abeta and Tau.")
    .default_value("")
    .choices("Abeta", "abeta", "Tau", "tau", "");
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
  bool iterativeRigid = program.get<bool>("--iterative");
  bool ADNIstyle = program.get<bool>("--ADNI-PET-core");
  std::string modality = program.get<std::string>("--decouple");
  modality = Common::toLower(modality);

  ReaderType::Pointer reader = ReaderType::New();
  try {
    ImageType::Pointer rigidImage{nullptr};
    ImageType::Pointer paddedTemplate =
        Common::LoadNii(Common::EXECUTABLE_DIR + "/" + Common::TEMPLATE_PADDED);
    Rigid rigidModel =
        Rigid(Common::EXECUTABLE_DIR + "/" + Common::MODEL_RIGID,
              Common::EXECUTABLE_DIR + "/" + Common::MODEL_AFFINE_VOXELMORPH);
    if (!manualFovPlacement) {
      if (iterativeRigid) {
        rigidImage = iterativeRigidAlign(inputPath, Common::EXECUTABLE_DIR,
                                         rigidModel, 5, 2, paddedTemplate);
      } else {
        rigidImage = rigidAlign(inputPath, Common::EXECUTABLE_DIR, rigidModel,
                                paddedTemplate);
      }
      Common::SaveImage(rigidImage, Common::EXECUTABLE_DIR + "/rigid.nii");
    } else {
      rigidImage = Common::LoadNii(inputPath);
      Common::SaveImage(rigidImage, Common::EXECUTABLE_DIR + "/rigid.nii");
    }

    ImageType::Pointer paddedImage =
        Common::ResampleToMatch(paddedTemplate, rigidImage);
    Common::SaveImage(paddedImage, Common::EXECUTABLE_DIR + "/padded.nii");
    paddedImage = rigidModel.preprocessVoxelMorph(paddedImage);
    Common::SaveImage(paddedImage,
                      Common::EXECUTABLE_DIR + "/paddedProcessed.nii");
    ImageType::Pointer paddedOriginalImage =
        Common::LoadNii(Common::EXECUTABLE_DIR + "/padded.nii");

    std::vector<float> paddedImageData, paddedTemplateData, paddedOriginalData;
    Common::ExtractImageData(paddedImage, paddedImageData);
    Common::ExtractImageData(paddedTemplate, paddedTemplateData);
    Common::ExtractImageData(paddedOriginalImage, paddedOriginalData);
    std::unordered_map<std::string, std::vector<float>> warpedImageData =
        rigidModel.predictVoxelMorph(paddedOriginalData, paddedImageData,
                                     paddedTemplateData);
    ImageType::Pointer warpedImage = Common::CreateImageFromVector(
        warpedImageData["warped"],
        paddedImage->GetLargestPossibleRegion().GetSize());
    warpedImage->SetDirection(paddedTemplate->GetDirection());
    warpedImage->SetOrigin(paddedTemplate->GetOrigin());
    warpedImage->SetSpacing(paddedTemplate->GetSpacing());
    warpedImage = CropMNI(warpedImage);
    Common::SaveImage(warpedImage, outputPath);
    CentiloidCalculator clCalc;
    MetricResult clResult = clCalc.calculate(warpedImage);
    clResult.printResult();

    CenTauRCalculator ctrCalc;
    MetricResult ctrResult = ctrCalc.calculate(warpedImage);
    ctrResult.printResult();

    ImageType::Pointer ADNIPETCoreProcessed;
    double meanCerebralGray;
    if (ADNIstyle || !modality.empty()) {
      ImageType::Pointer refCerebralGray = Common::LoadNii(
          Common::EXECUTABLE_DIR + "/" + Common::MASK_CEREBRAL_GRAY);
      ImageType::Pointer resampledImage =
          Common::ResampleToMatch(refCerebralGray, warpedImage);
      meanCerebralGray =
          Common::CalculateMeanInMask(resampledImage, refCerebralGray);
      ImageType::Pointer ADNITemplate = Common::LoadNii(
          Common::EXECUTABLE_DIR + "/" + Common::TEMPLATE_ADNI_PET_CORE);
      ADNIPETCoreProcessed =
          Common::ResampleToMatch(ADNITemplate, rigidImage);
      Common::DivideVoxelsByValue(ADNIPETCoreProcessed, meanCerebralGray);
      Common::SaveImage(ADNIPETCoreProcessed,
        Common::EXECUTABLE_DIR + "/ADNI_normalized.nii");
      if (ADNIstyle)
        Common::SaveImage(ADNIPETCoreProcessed, outputPath);
    }

    if (modality.empty()) {
      return EXIT_SUCCESS;
    }
    std::string decouplerModelPath;
    if (modality == "abeta")
      decouplerModelPath = Common::MODEL_ABETA_DECOUPLER;
    else if (modality == "tau")
      decouplerModelPath = Common::MODEL_TAU_DECOUPLER;
    else
      throw std::runtime_error("Invalid modality.");
    Decoupler decoupler(Common::EXECUTABLE_DIR + "/" + decouplerModelPath);    
    DecoupledResult decoupledResult = decoupler.predict(ADNIPETCoreProcessed);
    decoupledResult.SaveResults(outputPath);
    decoupledResult.printResult();

  } catch (itk::ExceptionObject& err) {
    std::cerr << "Exception caught in main: " << err << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
