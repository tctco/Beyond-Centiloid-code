#pragma once
#ifndef COMMON_H
#define COMMON_H
#include <itkAffineTransform.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkWarpImageFilter.h>

#include <filesystem>

using ImageType = itk::Image<float, 3>;
using DDFType = itk::Image<itk::Vector<double, 3>, 3>;
using BinaryImageType = itk::Image<unsigned char, 3>;
using ReaderType = itk::ImageFileReader<ImageType>;

namespace Common {
extern std::string EXECUTABLE_DIR;

extern std::string TEMPLATE_ADNI_PET_CORE;
extern std::string TEMPLATE_PADDED;

extern std::string MASK_CEREBRAL_GRAY;
extern std::string MASK_WHOLE_CEREBRAL;
extern std::string MASK_CENTILOID_VOI;
extern std::string MASK_CENTAUR_VOI;
extern std::string MASK_CENTAUR_REF;

extern std::string MODEL_ABETA_DECOUPLER;
extern std::string MODEL_TAU_DECOUPLER;
extern std::string MODEL_AFFINE_VOXELMORPH;
extern std::string MODEL_RIGID;

void SaveImage(ImageType::Pointer image, const std::string& filename);
ImageType::Pointer LoadNii(const std::string& filename);
void DivideVoxelsByValue(ImageType::Pointer image, float divisor);
double CalculateMeanInMask(ImageType::Pointer image, ImageType::Pointer mask);
ImageType::Pointer ResampleToMatch(typename ImageType::Pointer referenceImage,
                                   typename ImageType::Pointer inputImage);
ImageType::Pointer CreateImageFromVector(const std::vector<float>& imageData,
                                         ImageType::SizeType size);
void ExtractImageData(ImageType::Pointer image, std::vector<float>& imageData);
std::string addSuffixToFilePath(const std::string& filePath,
                                const std::string& suffix);
void debugLog(const std::string& message);
std::string toLower(const std::string& str);
}  // namespace Common

#endif