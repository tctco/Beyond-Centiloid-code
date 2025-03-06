#include "common.h"

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
  return std::filesystem::path(executablePath).parent_path().string();
}

namespace Common {
std::string TEMPLATE_ADNI_PET_CORE = "ADNI_empty.nii";
std::string TEMPLATE_PADDED = "paddedTemplate.nii";
std::string MASK_CEREBRAL_GRAY = "voi_CerebGry_2mm.nii";
std::string MASK_CENTILOID_VOI = "voi_ctx_2mm.nii";
std::string MASK_WHOLE_CEREBRAL = "voi_WhlCbl_2mm.nii";
std::string MASK_CENTAUR_VOI = "CenTauR.nii";
std::string MASK_CENTAUR_REF = "voi_CerebGry_tau_2mm.nii";
std::string MODEL_ABETA_DECOUPLER = "abeta.onnx";
std::string MODEL_TAU_DECOUPLER = "tau.onnx";
std::string MODEL_AFFINE_VOXELMORPH = "affine_voxelmorph.onnx";
std::string MODEL_RIGID = "rigid.onnx";
std::string EXECUTABLE_DIR = getExecutablePath();

void DivideVoxelsByValue(ImageType::Pointer image, float divisor) {
  itk::ImageRegionIterator<ImageType> it(image,
                                         image->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    it.Set(it.Get() / divisor);
  }
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

ImageType::Pointer CreateImageFromVector(const std::vector<float>& imageData,
                                         ImageType::SizeType size) {
  ImageType::Pointer image = ImageType::New();
  ImageType::IndexType start = {{0, 0, 0}};
  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);

  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        ImageType::IndexType index = {{x, y, z}};
        size_t vectorIndex = x * size[1] * size[2] + y * size[2] + z;
        image->SetPixel(index, imageData[vectorIndex]);
      }
    }
  }

  return image;
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

void SaveImage(ImageType::Pointer image, const std::string& filename) {
  using WriterType = itk::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(filename);
  writer->SetInput(image);
  writer->Update();
}

ImageType::Pointer LoadNii(const std::string& filename) {
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  reader->Update();
  return reader->GetOutput();
}

void ExtractImageData(ImageType::Pointer image, std::vector<float>& imageData) {
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();

  imageData.resize(size[0] * size[1] * size[2]);
  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        ImageType::IndexType index = {{x, y, z}};
        float pixelValue = image->GetPixel(index);
        imageData[x * size[1] * size[2] + y * size[2] + z] = pixelValue;
      }
    }
  }
}

std::string addSuffixToFilePath(const std::string& filePath,
                                const std::string& suffix) {
  std::filesystem::path path(filePath);
  std::string stem = path.stem().string();
  std::string extension = path.extension().string();
  std::string parentPath = path.parent_path().string();

  std::string newFilePath = parentPath + "/" + stem + suffix + extension;
  return newFilePath;
}

void debugLog(const std::string& message) { std::cout << message << std::endl; }

std::string toLower(const std::string& str) {
  std::string lowerStr = str;
  std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
  return lowerStr;
}
}  // namespace Common
