#ifndef UTILS_INCLUDE_MODEL_TOOL_EIGEN_IMPL_H_
#define UTILS_INCLUDE_MODEL_TOOL_EIGEN_IMPL_H_

#include <Eigen>
#include <butil/logging.h>

namespace ps {
namespace model {

template<class A, class B>
bool matrix_size_equals(const Eigen::DenseBase<A>& a, const Eigen::DenseBase<B>& b) {
  return a.rows() == b.rows() && a.cols() == b.cols();
}

template<class A>
bool matrix_size_equals(const Eigen::DenseBase<A>& a, Eigen::DenseIndex rows, Eigen::DenseIndex cols) {
  return a.rows() == rows && a.cols() == cols;
}

template<class A, class B, class C>
void matrix_multiply(bool transa, bool transb, bool transc, const A& a, const B& b, C& c) {
  if (transc) {
    return matrix_multiply(!transb, !transa, false, b, a, c);
  }
  if (!transa) {
    if (!transb) {
      CHECK(a.cols() == b.rows());
      c.noalias() = a * b;
    } else {
      CHECK(a.cols() == b.cols());
      c.noalias() = a * b.transpose();
    }
  } else {
    if (!transb) {
      CHECK(a.rows() == b.rows());
      c.noalias() = a.transpose() * b;
    } else {
      CHECK(a.rows() == b.cols());
      c.noalias() = a.transpose() * b.transpose();
    }
  }
}

template<class A, class B, class C>
void matrix_multiply_add(bool transa, bool transb, bool transc, const A& a, const B& b, C& c) {
  if (transc) {
    return matrix_multiply_add(!transb, !transa, false, b, a, c);
  }
  if (!transa) {
    if (!transb) {
      CHECK(a.cols() == b.rows());
      CHECK(matrix_size_equals(c, a.rows(), b.cols()));
      c.noalias() += a * b;
    } else {
      CHECK(a.cols() == b.cols());
      CHECK(matrix_size_equals(c, a.rows(), b.rows()));
      c.noalias() += a * b.transpose();
    }
  } else {
    if (!transb) {
      CHECK(a.rows() == b.rows());
      CHECK(matrix_size_equals(c, a.cols(), b.cols()));
      c.noalias() += a.transpose() * b;
    } else {
      CHECK(a.rows() == b.cols());
      CHECK(matrix_size_equals(c, a.cols(), b.rows()));
      c.noalias() += a.transpose() * b.transpose();
    }
  }
}

// realize C = A .* B
template<class A, class B, class C>
void matrix_product(bool transa, bool transb, bool transc, const A& a, const B& b, C& c) {
  if (transc) {
    return matrix_multiply(!transa, !transb, false, a, b, c);
  }

  if (!transa) {
    if (!transb) {
      CHECK(a.rows() == b.rows() && a.cols() == b.cols());
      c.noalias() = a.cwiseProduct(b);
    } else {
      CHECK(a.rows() == b.cols() && a.cols() == b.rows());
      c.noalias() = a.cwiseProduct(b.transpose());
    }
  } else {
    if (!transb) {
      CHECK(a.rows() == b.cols() && a.cols() == b.rows());
      c.noalias() = a.transpose().cwiseProduct(b);
    } else {
      CHECK(a.rows() == b.rows() && a.cols() == b.cols());
      c.noalias() = a.transpose().cwiseProduct(b.transpose());
    }
  }
}

// realize C += A .* B
template<class A, class B, class C>
void matrix_product_add(bool transa, bool transb, bool transc, const A& a, const B& b, C& c) {
  if (transc) {
    return matrix_product_add(!transa, !transb, false, a, b, c);
  }

  if (!transa) {
    if (!transb) {
      CHECK(a.rows() == b.rows() && c.rows() == b.rows() && a.cols() == b.cols() && c.cols() == b.cols());
      c.noalias() += a.cwiseProduct(b);
    } else {
      CHECK(a.rows() == b.cols() && c.rows() == a.rows() && a.cols() == b.rows() && c.cols() == a.cols());
      c.noalias() += a.cwiseProduct(b.transpose());
    }
  } else {
    if (!transb) {
      CHECK(a.cols() == b.rows() && c.rows() == b.rows() && a.rows() == b.cols() && c.cols() == b.cols());
      c.noalias() += a.transpose().cwiseProduct(b);
    } else {
      CHECK(a.cols() == b.cols() && c.rows() == b.cols() && a.rows() == b.rows() && c.cols() == b.rows());
      c.noalias() += a.transpose().cwiseProduct(b.transpose());
    }
  }
}

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_TOOL_EIGEN_IMPL_H_

