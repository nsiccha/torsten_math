#ifndef STAN_MATH_TORSTEN_PMX_DDE_SOLVER_HPP
#define STAN_MATH_TORSTEN_PMX_DDE_SOLVER_HPP

#include <stan/math/prim/functor/dde_retard.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/torsten/dsolve/pmx_dde_system.hpp>
#include <stan/math/torsten/pmx_dde_schedule.hpp>
#include <Eigen/Dense>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torsten {

  /**
   * Control parameters of <code>pmx_solve_dde_rk45</code>: tolerances and
   * maximum number of steps of the DDE integrator, and the same for the
   * steady-state solves (in units of dosing periods for the burn-in).
   */
  struct PMXDdeControls {
    double rtol = 1.e-6;
    double atol = 1.e-6;
    long int max_num_steps = 1000000;  // NOLINT(runtime/int)
    double as_rtol = 1.e-6;
    double as_atol = 1.e-6;
    long int as_max_num_steps = 100;  // NOLINT(runtime/int)
    std::ostream* msgs = nullptr;
  };

  /**
   * Solves a PMX DDE over the events of one subject.
   *
   * One resumable integrator (<code>dde_retard_integrator</code>) lives
   * across all events, so the history of the solution, and of its
   * sensitivities, survives the doses. The events map onto the integrator's
   * operations:
   *  - between two events the integrator advances to the next event time,
   *    after a change of the right hand side if the new interval has another
   *    parameter row, other rates or other data;
   *  - a bolus dose is a jump of the state;
   *  - turning a compartment off (EVID 2, negative CMT) and overwriting it
   *    are jumps to the new value;
   *  - a reset (EVID 3, and EVID 4 before its dose) restarts the integration
   *    from the history <code>y_hist</code>, forgetting the earlier solution;
   *  - a steady-state dose (SS = 1) restarts from the periodic steady state
   *    of the dosing regimen.
   *
   * The forward sensitivities are taken with respect to the global
   * parameters of the schedule. The times of jumps, of changes of the right
   * hand side and of restarts may depend on them; the integrator applies the
   * jump conditions of a moving discontinuity, and a restart at time u
   * starts the sensitivities from dy_hist/dp - f(u) du/dp.
   *
   * @tparam F type of the user's right hand side
   */
  template <typename F>
  class PMXDdeSolver {
    using system_t = dsolve::PMXDdeSystem<F>;
    using integrator_t = stan::math::internal::dde_retard_integrator<system_t>;

  public:
    static constexpr const char* function_name = "pmx_solve_dde_rk45";

    /**
     * @param f the user's right hand side
     * @param ncmt number of compartments
     * @param tau delays
     * @param delay_sensitivities P x D derivatives of the delays
     * @param y_hist history: the state before the first event and after a
     *   reset
     * @param y_hist_sensitivities nCmt x P derivatives of the history
     * @param controls tolerances and step limits
     */
    PMXDdeSolver(const F& f, int ncmt, const std::vector<double>& tau,
                 const Eigen::MatrixXd& delay_sensitivities,
                 const Eigen::VectorXd& y_hist,
                 const Eigen::MatrixXd& y_hist_sensitivities,
                 const PMXDdeControls& controls)
      : f_(f),
        n_(ncmt),
        p_(y_hist_sensitivities.cols()),
        tau_(tau),
        delay_sensitivities_(delay_sensitivities),
        controls_(controls),
        y_hist_(y_hist),
        z_hist_(n_ * (1 + p_)),
        system_(f, ncmt, p_, delay_sensitivities, true, controls.msgs),
        integrator_(function_name, system_, tau, controls.rtol, controls.atol,
                    controls.max_num_steps) {
      z_hist_.head(n_) = y_hist;
      for (size_t p = 0; p < p_; ++p) {
        z_hist_.segment(n_ * (1 + p), n_) = y_hist_sensitivities.col(p);
      }
    }

    /**
     * Solve the events.
     *
     * @param events the events of the subject, in order
     * @param[out] values nCmt x (number of kept events) state after each
     *   kept event
     * @param[out] sensitivities for each kept event, the nCmt x P
     *   derivatives of its state with respect to the global parameters
     */
    void solve(const std::vector<PMXDdeEvent>& events, Eigen::MatrixXd& values,
               std::vector<Eigen::MatrixXd>& sensitivities) {
      int nkeep = 0;
      for (const PMXDdeEvent& e : events) {
        nkeep += e.keep;
      }
      values.resize(n_, nkeep);
      sensitivities.clear();
      sensitivities.reserve(nkeep);

      system_.set_rhs(events[0].rhs);
      restart(events[0].time, events[0].time_sensitivities);
      int ikeep = 0;
      for (size_t i = 0; i < events.size(); ++i) {
        const PMXDdeEvent& e = events[i];
        const bool restarts = e.id >= 1 && e.id <= 4;
        if (i > 0) {
          if (restarts) {
            system_.set_rhs(e.rhs);
          } else {
            if (system_.rhs() != e.rhs) {
              integrator_.change_rhs(events[i - 1].time_sensitivities,
                                     [&] { system_.set_rhs(e.rhs); });
            }
            integrator_.advance_to(e.time);
          }
        }

        switch (e.id) {
        case 1:  // EVID 3: reset
          restart(e.time, e.time_sensitivities);
          break;
        case 2:  // EVID 4: reset and dose
          restart(e.time, e.time_sensitivities);
          dose(e);
          break;
        case 3:  // SS = 2
          throw std::domain_error(
              std::string(function_name)
              + ": steady-state dosing with SS = 2 (adding a steady state to "
                "the current state) is not supported: superposition does not "
                "hold for nonlinear systems, and is undefined for the history "
                "of a DDE");
        case 4:  // SS = 1
          steady_state(e);
          dose(e);
          break;
        case 5:  // EVID 2, negative CMT: turn the compartment off
          overwrite(-e.cmt - 1, 0.0, Eigen::VectorXd(), e.time_sensitivities);
          break;
        case 6:  // overwrite a compartment
          overwrite(e.cmt - 1, e.amount, e.amount_sensitivities,
                    e.time_sensitivities);
          break;
        default:  // transient dose or no action
          dose(e);
        }

        if (e.keep) {
          const Eigen::VectorXd& z = integrator_.state();
          values.col(ikeep) = z.head(n_);
          Eigen::MatrixXd s = Eigen::Map<const Eigen::MatrixXd>(z.data() + n_, n_, p_);
          if (e.time_sensitivities.size() > 0) {
            // the state at the event moves with the event time
            s += current_derivative("the time of an event in the schedule")
                 * e.time_sensitivities.transpose();
          }
          sensitivities.push_back(s);
          ++ikeep;
        }
      }
    }

  private:
    /**
     * Restart the integration at time t from the history, forgetting the
     * earlier solution. The solution after t depends on t: its derivative
     * with respect to t just after the restart is -f(t, y_hist, y_hist), so
     * the sensitivities start from dy_hist/dp - f dt/dp. The history before
     * t is constant and keeps the sensitivities dy_hist/dp.
     *
     * The right hand side used for f is the current one. Should it change at
     * t (the next interval has another one), the integrator's jump condition
     * of that change replaces it by the new one.
     */
    void restart(double t, const Eigen::VectorXd& dt_dtheta) {
      Eigen::VectorXd z0 = z_hist_;
      if (dt_dtheta.size() > 0) {
        const Eigen::MatrixXd y_lag = y_hist_.replicate(1, tau_.size());
        const Eigen::VectorXd f = system_.base(t, y_hist_, y_lag);
        for (size_t p = 0; p < p_; ++p) {
          z0.segment(n_ * (1 + p), n_) -= dt_dtheta.coeff(p) * f;
        }
      }
      integrator_.start(t, z0, z_hist_);
    }

    /**
     * Add the bolus amount of an event to its compartment.
     */
    void dose(const PMXDdeEvent& e) {
      if (e.amount == 0 && e.amount_sensitivities.size() == 0) {
        return;
      }
      const int c = e.cmt - 1;
      Eigen::VectorXd dz = Eigen::VectorXd::Zero(integrator_.state().size());
      dz.coeffRef(c) = e.amount;
      if (e.amount_sensitivities.size() > 0) {
        for (size_t p = 0; p < p_; ++p) {
          dz.coeffRef(n_ * (1 + p) + c) = e.amount_sensitivities.coeff(p);
        }
      }
      integrator_.jump(dz, e.time_sensitivities);
    }

    /**
     * Set compartment c to the given amount.
     *
     * The jump y_c -> amount depends on the state, so if its time u moves
     * with the parameters the sensitivity of compartment c just after it is
     * d(amount)/dp - f_c(u+) du/dp. The integrator adds
     * (f(u-) - f(u+)) du/dp to every component, so the jump of the
     * sensitivity passed to it also removes S_c + f_c(u-) du/dp.
     */
    void overwrite(int c, double amount, const Eigen::VectorXd& amount_sensitivities,
                   const Eigen::VectorXd& dt_dtheta) {
      const Eigen::VectorXd& z = integrator_.state();
      Eigen::VectorXd dz = Eigen::VectorXd::Zero(z.size());
      dz.coeffRef(c) = amount - z.coeff(c);
      for (size_t p = 0; p < p_; ++p) {
        dz.coeffRef(n_ * (1 + p) + c) = -z.coeff(n_ * (1 + p) + c);
      }
      if (amount_sensitivities.size() > 0) {
        for (size_t p = 0; p < p_; ++p) {
          dz.coeffRef(n_ * (1 + p) + c) += amount_sensitivities.coeff(p);
        }
      }
      if (dt_dtheta.size() > 0) {
        const double f_c = current_derivative(
            "the time of an event that turns off or overwrites a compartment")(c);
        for (size_t p = 0; p < p_; ++p) {
          dz.coeffRef(n_ * (1 + p) + c) -= f_c * dt_dtheta.coeff(p);
        }
      }
      integrator_.jump(dz, dt_dtheta);
    }

    /**
     * The derivative of the base state at the current time and state.
     *
     * Needs the right hand side with the integrator's own lagged values,
     * which the DDE core does not expose yet (math snag
     * dde-retard-integ-a0e8d0a8).
     *
     * @param what the input whose gradient needs it, for the error message
     */
    Eigen::VectorXd current_derivative(const std::string& what) const {
      throw std::invalid_argument(
          std::string(function_name) + ": the gradient with respect to " + what
          + " is not available yet: it needs the derivative of the solution "
            "at the current time from the DDE core (math snag "
            "dde-retard-integ-a0e8d0a8). Pass that time as data.");
    }

    /**
     * Restart the integration at a steady-state dose from the periodic (or,
     * for a constant infusion, constant) steady state of its regimen.
     */
    void steady_state(const PMXDdeEvent& e) {
      throw std::invalid_argument(
          std::string(function_name)
          + ": steady-state dosing (SS = 1) is not implemented yet");
    }

    const F& f_;
    const size_t n_;
    const size_t p_;
    const std::vector<double> tau_;
    const Eigen::MatrixXd delay_sensitivities_;
    const PMXDdeControls controls_;
    const Eigen::VectorXd y_hist_;
    Eigen::VectorXd z_hist_;
    system_t system_;
    integrator_t integrator_;
  };

}

#endif
