#ifndef UTILS_INCLUDE_MODEL_TOOL_AUC_CALCULATOR_H_
#define UTILS_INCLUDE_MODEL_TOOL_AUC_CALCULATOR_H_

#include <vector>
#include <string>
#include <mutex>

namespace ps {
namespace model {

struct AUCResult{
  double auc;
  double actual_ctr;
  double predict_ctr;
};

class AUCCalculator {
 public:
  AUCCalculator(const AUCCalculator&) = delete;
  AUCCalculator();
  AUCCalculator(const std::string& name);
  ~AUCCalculator() = default;

  void clear();
  void add(double pred, double label);
  void add(size_t n, const double *preds, const double *labps);
  void compute();
  void print_all_measures();

  inline void set_name(const std::string& name) {
    name_ = name;
  }

  inline double auc() const {
    return auc_;
  }

  inline double actual_ctr() const {
    return actual_ctr_;
  }

  inline double predicted_ctr() const {
    return predicted_ctr_;
  }

  inline double size() const {
    return size_;
  }

  // 返回auc计算结果的接口，外部可以用来做判断
  inline AUCResult get_auc_result() const {
    AUCResult result;
    result.auc = auc_;
    result.actual_ctr = actual_ctr_;
    result.predict_ctr = predicted_ctr_;
    return std::move(result);
  }

 private:
  static const int TABLE_SIZE = 1000000;

  std::string name_;
  double auc_;
  double mae_;
  double rmse_;
  double actual_ctr_;
  double predicted_ctr_;
  double local_err1_;
  double local_err2_;
  double local_pred_;
  double size_;

  std::vector<double> table_[2];
  std::mutex mutex_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_TOOL_AUC_CALCULATOR_H_

