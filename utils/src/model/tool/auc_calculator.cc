#include "model/tool/auc_calculator.h"

#include <math.h>
#include <butil/logging.h>
#include "toolkit/mpi_agent.h"

using ps::toolkit::MPIAgent;

namespace ps {
namespace model {

static inline bool real_eq(const double x1, const double x2) {
  static const double eps = 1e-9;
  return fabs(x1 - x2) < eps;
}

static inline int rounding(const double x) {
  return (x < 0) ? (int)(x - 0.5) : (int)(x + 0.5);
}

AUCCalculator::AUCCalculator() :
  name_(),
  auc_(0.0),
  mae_(0.0),
  rmse_(0.0),
  actual_ctr_(0.0),
  predicted_ctr_(0.0),
  local_err1_(0.0),
  local_err2_(0.0),
  local_pred_(0.0),
  size_(0.0),
  mutex_() {
  table_[0].assign(TABLE_SIZE, 0.0);
  table_[1].assign(TABLE_SIZE, 0.0);
}

AUCCalculator::AUCCalculator(const std::string& name) :
  name_(name),
  auc_(0.0),
  mae_(0.0),
  rmse_(0.0),
  actual_ctr_(0.0),
  predicted_ctr_(0.0),
  local_err1_(0.0),
  local_err2_(0.0),
  local_pred_(0.0),
  size_(0.0),
  mutex_() {
  table_[0].assign(TABLE_SIZE, 0.0);
  table_[1].assign(TABLE_SIZE, 0.0);
}

void AUCCalculator::clear() {
  mutex_.lock();

  auc_  = 0.0;
  mae_  = 0.0;
  rmse_ = 0.0;
  actual_ctr_ = 0.0;
  predicted_ctr_ = 0.0;
  local_err1_ = 0.0;
  local_err2_ = 0.0;
  local_pred_ = 0.0;
  size_ = 0.0;

  table_[0].assign(TABLE_SIZE, 0.0);
  table_[1].assign(TABLE_SIZE, 0.0);

  mutex_.unlock();
}

void AUCCalculator::add(size_t n, const double *preds, const double *labps) {
  mutex_.lock();

  for (size_t i = 0; i < n; ++i) {
    CHECK(preds[i] >= 0.0 && preds[i] <= 1.0) << "pred = " << preds[i];
    CHECK(real_eq(labps[i], 0) || real_eq(labps[i], 1)) << "label = " << labps[i];

    ++table_[rounding(labps[i])][std::min((int)(preds[i] * TABLE_SIZE), (int)(TABLE_SIZE - 1))];
    local_err1_ += fabs(preds[i] - labps[i]);
    local_err2_ += (preds[i] - labps[i]) * (preds[i] - labps[i]);
    local_pred_ += preds[i];
  }

  mutex_.unlock();
}

void AUCCalculator::add(double pred, double label) {
  mutex_.lock();

  CHECK(pred >= 0.0 && pred <= 1.0) << "pred = " << pred;
  CHECK(real_eq(label, 0) || real_eq(label, 1)) << "label = " << label;

  ++table_[rounding(label)][std::min((int)(pred * TABLE_SIZE), (int)(TABLE_SIZE - 1))];
  local_err1_ += fabs(pred - label);
  local_err2_ += (pred - label) * (pred - label);
  local_pred_ += pred;

  mutex_.unlock();
}

void AUCCalculator::compute() {
  CHECK(0 == MPI_Allreduce(MPI_IN_PLACE, table_[0].data(), TABLE_SIZE, MPI_DOUBLE, MPI_SUM, MPIAgent::mpi_comm_group()));
  CHECK(0 == MPI_Allreduce(MPI_IN_PLACE, table_[1].data(), TABLE_SIZE, MPI_DOUBLE, MPI_SUM, MPIAgent::mpi_comm_group()));

  double area = 0.0;
  double fp = 0.0;
  double tp = 0.0;

  for (int i = TABLE_SIZE - 1; i >= 0; --i) {
    double newfp = fp + table_[0][i];
    double newtp = tp + table_[1][i];
    area += (newfp - fp) * (tp + newtp) / 2.0;
    fp = newfp;
    tp = newtp;
  }

  auc_ = area / (fp * tp);

  mae_ = MPIAgent::mpi_allreduce_group(local_err1_, MPI_SUM) / (fp + tp);
  rmse_ = sqrt(MPIAgent::mpi_allreduce_group(local_err2_, MPI_SUM) / (fp + tp));
  actual_ctr_ = tp / (fp + tp);
  predicted_ctr_ = MPIAgent::mpi_allreduce_group(local_pred_, MPI_SUM) / (fp + tp);
  size_ = fp + tp;
}

void AUCCalculator::print_all_measures() {
  fprintf(stdout, "%s: AUC=%.6f MAE=%.6f RMSE=%.6f Actual CTR=%.6f Predicted CTR=%.6f\n",
          name_.c_str(), auc_, mae_, rmse_, actual_ctr_, predicted_ctr_);
}

} // namespace model
} // namespace ps

