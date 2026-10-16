#ifndef STAN_MATH_TORSTEN_PMX_EXP_DIVIDED_DIFFERENCE_HPP
#define STAN_MATH_TORSTEN_PMX_EXP_DIVIDED_DIFFERENCE_HPP

#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/expm1.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/rev/fun/expm1.hpp>
#include <cmath>

namespace torsten {
namespace internal {

  /**
   * phi1(z) = (exp(z) - 1) / z, extended continuously by phi1(0) = 1.
   *
   * Near zero <code>expm1(z) / z</code> is 0/0 at z = 0 and its
   * derivative loses precision, so there the Taylor series
   * sum_{n=0}^{11} z^n / (n + 1)! is used; for |z| < 0.1 its
   * truncation error is below 1e-21 in value and derivative.
   *
   * @tparam T scalar type
   * @param z argument
   * @return phi1(z)
   */
  template <typename T>
  inline T pmx_phi1(const T& z) {
    using stan::math::expm1;
    if (std::abs(stan::math::value_of_rec(z)) < 0.1) {
      T s = 1.0;
      for (int n = 12; n >= 2; --n) {
        s = 1.0 + z * s / n;
      }
      return s;
    }
    return expm1(z) / z;
  }

  /**
   * (1 - exp(-x t)) / x, the integral of exp(-x s) over s in [0, t].
   * Accurate for small x t, and equal to t at x = 0.
   *
   * @param x rate
   * @param t time span
   */
  template <typename Tx, typename Tt>
  inline stan::return_type_t<Tx, Tt> pmx_exp_integral(const Tx& x, const Tt& t) {
    return t * pmx_phi1(-x * t);
  }

  /**
   * (exp(-k t) - exp(-a t)) / (a - k), the integral of
   * exp(-k (t - s)) exp(-a s) over s in [0, t]: the divided difference of
   * exp(-x t) over the rates a and k. It is symmetric in (a, k) and
   * smooth across a == k, where it equals t exp(-k t). Factoring out the
   * slower decay keeps the argument of phi1 non-positive, so the result
   * neither cancels nor overflows; derivatives are those of the full
   * function, including at a == k.
   *
   * @param a first rate
   * @param k second rate
   * @param t time span
   */
  template <typename Ta, typename Tk, typename Tt>
  inline stan::return_type_t<Ta, Tk, Tt>
  pmx_exp_divided_difference(const Ta& a, const Tk& k, const Tt& t) {
    using stan::math::exp;
    if (stan::math::value_of_rec(a) >= stan::math::value_of_rec(k)) {
      return t * exp(-k * t) * pmx_phi1(-(a - k) * t);
    }
    return t * exp(-a * t) * pmx_phi1(-(k - a) * t);
  }

}  // namespace internal
}  // namespace torsten

#endif
