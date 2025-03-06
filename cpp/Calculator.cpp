#include "Calculator.h"

void MetricResult::printResult() {
  std::cout << "Metric: " << metricName << std::endl;
  std::cout << "SUVr: " << suvr << std::endl;
  for (auto const& [key, val] : result) {
    std::cout << key << ": " << val << std::endl;
  }
  std::cout << std::endl;
}

MetricResult CentiloidCalculator::calculate(
    ImageType::Pointer spatialNormalizedImage) {
  std::string voiTemplatePath =
      Common::EXECUTABLE_DIR + "/" + Common::MASK_CENTILOID_VOI;
  std::string refTemplatePath =
      Common::EXECUTABLE_DIR + "/" + Common::MASK_WHOLE_CEREBRAL;
  ImageType::Pointer voiTemplate = Common::LoadNii(voiTemplatePath);
  ImageType::Pointer refTemplate = Common::LoadNii(refTemplatePath);
  ImageType::Pointer resampledImage =
      Common::ResampleToMatch(voiTemplate, spatialNormalizedImage);
  double meanVoi = Common::CalculateMeanInMask(resampledImage, voiTemplate);
  double meanRef = Common::CalculateMeanInMask(resampledImage, refTemplate);
  double suvr = meanVoi / meanRef;
  std::map<std::string, float> result;
  result["PiB"] = suvr * 93.7 - 94.6;
  result["FBP"] = suvr * 175.4 - 182.3;
  result["FBB"] = suvr * 153.4 - 154.9;
  result["FMM"] = suvr * 121.4 - 121.2;
  result["NAV"] = suvr * 85.2 - 87.6;
  MetricResult cl{result, suvr, "Centiloid"};
  return cl;
}

MetricResult CenTauRCalculator::calculate(
    ImageType::Pointer spatialNormalizedImage) {
  std::string voiTemplatePath =
      Common::EXECUTABLE_DIR + "/" + Common::MASK_CENTILOID_VOI;
  std::string refTemplatePath =
      Common::EXECUTABLE_DIR + "/" + Common::MASK_CENTAUR_REF;
  ImageType::Pointer voiTemplate = Common::LoadNii(voiTemplatePath);
  ImageType::Pointer refTemplate = Common::LoadNii(refTemplatePath);
  ImageType::Pointer resampledImage =
      Common::ResampleToMatch(voiTemplate, spatialNormalizedImage);
  double meanVoi = Common::CalculateMeanInMask(resampledImage, voiTemplate);
  double meanRef = Common::CalculateMeanInMask(resampledImage, refTemplate);
  double suvr = meanVoi / meanRef;
  std::map<std::string, float> result;
  result["RO948"] = 13.05 * suvr - 15.57;
  result["FTP"] = 13.63 * suvr - 15.85;
  result["MK6240"] = 10.08 * suvr - 10.06;
  result["PM-PBB3"] = 16.73 * suvr - 15.34;
  result["PI2620"] = 8.45 * suvr - 9.61;
  MetricResult ctr{result, suvr, "CenTauRz"};
  return ctr;
}