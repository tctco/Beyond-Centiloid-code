#include "common.h"

class MetricResult {
 public:
  void printResult();
  std::map<std::string, float> result;  // tracer - centiloid value
  double suvr;
  std::string metricName;
};

class CentiloidCalculator {
 public:
  MetricResult calculate(ImageType::Pointer spatialNormalizedImage);
};

class CenTauRCalculator {
 public:
  MetricResult calculate(ImageType::Pointer spatialNormalizedImage);
};