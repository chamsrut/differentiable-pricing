#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dp/option.hpp"

namespace dp {

// Operational ceilings. They exist so an accidental request fails immediately
// instead of allocating for minutes. None of them is a numerical claim.
inline constexpr std::size_t maximum_pde_spot_intervals = 200'000U;
inline constexpr std::size_t maximum_pde_time_steps = 400'000U;
inline constexpr std::size_t maximum_pde_cash_dividends = 512U;
inline constexpr std::size_t maximum_pde_curve_nodes = 4'096U;
inline constexpr std::size_t maximum_pde_rannacher_steps = 64U;

// Half-width in nodes of the regime-consistency stencil used by the task 9C-C1
// structural Greek-eligibility rule: node i is regime-consistent only when
// nodes i-2 .. i+2 all carry the same exercise classification. Five nodes is the
// smallest window that contains the three-node derivative stencil of both
// immediate neighbours, so a node cannot be declared eligible while a neighbour
// whose value entered its own stencil sits in a different regime. It is a fixed
// constant, not a tuning knob: the rule was fixed before any surface numbers
// were inspected.
inline constexpr std::size_t pde_regime_stencil_radius = 2U;

// Times are ACT/365F year fractions measured from the discount curve's origin.
// One second is about 3.17e-8 years, so this tolerance is far below any
// economically distinguishable instant. It is used only to decide whether two
// declared times are the same knot, never to repair an ordering.
inline constexpr double pde_time_tolerance = 1.0e-12;

// Settlement is carried because the task 9B input contract lists it. Task 9C-A
// prices per share and reads neither this nor the contract multiplier in any
// arithmetic; both are reporting metadata that must still be stated.
enum class SettlementConvention {
    cash,
    physical,
};

[[nodiscard]] SettlementConvention parse_settlement_convention(std::string_view value);

// Times and log discount factors of a piecewise-linear log-discount curve.
//
// The curve is carried in log-discount space so positivity holds by
// construction and a zero rate is a zero slope rather than a quotient. Linear
// interpolation of the log discount makes the implied instantaneous rate
// piecewise constant, equal on each segment to the negative slope. The curve
// must begin at (0, 0) and must bracket the whole valuation interval:
// evaluation outside [times.front(), times.back()] is rejected, never
// extrapolated.
struct PiecewiseLogDiscountCurve {
    std::vector<double> times;
    std::vector<double> log_discounts;

    void validate() const;

    // Linear interpolation in log-discount space. Throws std::invalid_argument
    // outside the declared knot range.
    [[nodiscard]] double log_discount(double time) const;

    // exp(log_discount(to_time) - log_discount(from_time)).
    [[nodiscard]] double discount_factor(double from_time, double to_time) const;

    // Piecewise-constant instantaneous rate on segment [times[i], times[i+1]].
    [[nodiscard]] double segment_rate(std::size_t segment_index) const;

    // Index of the segment containing `time`; the last segment is used at the
    // final knot. Throws std::invalid_argument outside the knot range.
    [[nodiscard]] std::size_t segment_index_for(double time) const;
};

struct CashDividend {
    double ex_time;
    double amount;
};

// A dividend schedule is a declaration, not a default.
//
// "This contract pays no cash dividend" and "the caller never said" are
// different statements, and a default-constructed vector cannot distinguish
// them. A default-constructed schedule is therefore undeclared and every entry
// point rejects it; callers reach the empty case only through
// `declared_none()`.
class CashDividendSchedule {
public:
    CashDividendSchedule() = default;

    [[nodiscard]] static CashDividendSchedule declared_none();
    [[nodiscard]] static CashDividendSchedule declared(std::vector<CashDividend> dividends);

    [[nodiscard]] bool is_declared() const noexcept {
        return declared_;
    }
    [[nodiscard]] const std::vector<CashDividend>& dividends() const noexcept {
        return dividends_;
    }

    // Rejects an undeclared schedule, a nonpositive or non-finite amount, an
    // ex-time outside the open interval (valuation_time, expiry_time), an
    // unsorted schedule, and duplicated ex-times.
    void validate(double valuation_time, double expiry_time) const;

private:
    CashDividendSchedule(bool declared, std::vector<CashDividend> dividends);

    bool declared_ = false;
    std::vector<CashDividend> dividends_;
};

// The economic state of a contract with the spot removed.
//
// One backward induction solves the whole valuation-time slice, so the spot is
// not part of what a solve is about: it is a query against the solved slice.
// Keeping it out of this struct is what makes the solved domain provably
// independent of any requested spot.
struct PdeSurfaceContract {
    OptionType option_type;
    ExerciseStyle exercise_style;
    double strike;
    double valuation_time;
    double expiry_time;
    // Constant for task 9C-A.
    double volatility;
    // The continuous carry c subtracted from the risk-neutral drift, so the
    // drift is (r(t) - c). std::nullopt means the caller has no carry
    // information; it is rejected rather than silently read as zero, because a
    // zero that reads as an observation is exactly the failure mode task 9B
    // wrote this field to prevent.
    std::optional<double> continuous_carry;
    PiecewiseLogDiscountCurve discount_curve;
    CashDividendSchedule dividends;
    SettlementConvention settlement;
    // Reporting metadata only. Prices are per share and this value never enters
    // the pricing arithmetic.
    double contract_multiplier;

    void validate() const;
};

// The contract priced by the finite-difference oracle, in the field order of
// the task 9B proposed input contract. Nothing here has a default: a missing
// rate, carry, dividend list or convention is an error, not a zero.
struct PdeContract {
    OptionType option_type;
    ExerciseStyle exercise_style;
    double spot;
    double strike;
    double valuation_time;
    double expiry_time;
    // Constant for task 9C-A.
    double volatility;
    // The continuous carry c subtracted from the risk-neutral drift, so the
    // drift is (r(t) - c). std::nullopt means the caller has no carry
    // information; it is rejected rather than silently read as zero, because a
    // zero that reads as an observation is exactly the failure mode task 9B
    // wrote this field to prevent.
    std::optional<double> continuous_carry;
    PiecewiseLogDiscountCurve discount_curve;
    CashDividendSchedule dividends;
    SettlementConvention settlement;
    // Reporting metadata only. Prices are per share and this value never enters
    // the pricing arithmetic.
    double contract_multiplier;

    void validate() const;

    // The same economic state without the spot. The scalar price is this
    // surface queried at `spot`, and both paths run the identical solve.
    [[nodiscard]] PdeSurfaceContract surface_state() const;
};

// Discretization settings. `spot_intervals` and `spot_maximum` are targets: the
// solver places a node exactly on the strike, which perturbs both slightly. The
// values actually used are reported in PdeResult and are what a refinement
// ladder must quote.
struct PdeGrid {
    std::size_t spot_intervals;
    std::size_t time_steps;
    double spot_maximum;
    // Number of time steps after the terminal payoff and after each dividend
    // jump that are replaced by two fully implicit half steps (Rannacher
    // damping). Zero disables damping and is permitted so its effect can be
    // measured.
    std::size_t rannacher_steps;
    double psor_tolerance;
    double psor_relaxation;
    std::size_t psor_maximum_iterations;

    void validate() const;
};

// Settings that belong to the valuation-time surface rather than to the solve.
// They change nothing about the discretization or the marching loop; they
// decide which solved nodes are structurally admissible as Greek labels.
struct PdeSurfaceSettings {
    // Nodes with index < buffer or index > spot_intervals - buffer are declared
    // Greek-ineligible. This is a node count, not a spot distance, and it is a
    // structural exclusion of the truncated domain edge. The separate, wider
    // interior harvesting window that a label dataset needs is a task 9C-C2
    // decision and is not implemented here. Must be at least
    // `pde_regime_stencil_radius` so that every node surviving it has a
    // complete regime stencil.
    std::size_t boundary_exclusion_nodes;

    void validate() const;
};

// A dividend as it was actually applied, with the index of the aligned time in
// PdeResult::aligned_times. This is the evidence that alignment happened.
struct PdeDividendEvent {
    double ex_time;
    double amount;
    std::size_t aligned_time_index;
};

// A returned result always carries `discrete_system_converged`: the implicit
// tridiagonal or PSOR system reached its declared residual contract. This says
// nothing about continuum discretization accuracy, which is reported
// separately.
enum class PdeSolverStatus {
    discrete_system_converged,
    psor_iteration_limit_exceeded,
};

enum class PdeDiscretizationAccuracy {
    not_assessed,
};

struct PdeResult {
    double price;
    PdeSolverStatus solver_status;
    PdeDiscretizationAccuracy discretization_accuracy;

    // Grid and time settings actually used.
    std::size_t spot_intervals;
    double spot_maximum;
    double spot_step;
    std::size_t strike_node_index;
    std::size_t time_steps;
    std::size_t rannacher_steps;
    std::size_t damped_half_steps;
    std::size_t crank_nicolson_steps;
    // Interior rows where central differencing would have produced a negative
    // off-diagonal and one-sided upwinding was used instead. Nonzero means the
    // convection term dominates diffusion near S = 0 on this grid.
    std::size_t upwinded_rows;

    // Solver diagnostics.
    std::size_t linear_solves;
    std::size_t psor_solves;
    std::size_t psor_total_iterations;
    std::size_t psor_maximum_iterations_used;
    double psor_tolerance;
    double psor_relaxation;
    // max_i |min(x_i - obstacle_i, (A x - b)_i)| over every solve, absolute and
    // normalised by max(1, ||b||_inf). The normalised value is the one compared
    // against psor_tolerance. For a European step the obstacle is absent and
    // this degenerates to the linear residual ||A x - b||_inf of the direct
    // tridiagonal solve.
    double maximum_lcp_residual;
    double maximum_relative_lcp_residual;

    // Times every dividend and every interior curve knot was aligned to,
    // including both interval endpoints. Small by construction.
    std::vector<double> aligned_times;
    std::vector<PdeDividendEvent> dividend_events;
};

// Everything one backward induction reports about itself, with no reference to
// a requested spot. Every field has exactly the meaning documented on the
// identically named `PdeResult` field above; `PdeResult` keeps its flat layout
// so the scalar API is unchanged, and the scalar wrapper copies this across.
struct PdeSolveDiagnostics {
    PdeSolverStatus solver_status;
    PdeDiscretizationAccuracy discretization_accuracy;

    std::size_t spot_intervals;
    double spot_maximum;
    double spot_step;
    std::size_t strike_node_index;
    std::size_t time_steps;
    std::size_t rannacher_steps;
    std::size_t damped_half_steps;
    std::size_t crank_nicolson_steps;
    std::size_t upwinded_rows;

    std::size_t linear_solves;
    std::size_t psor_solves;
    std::size_t psor_total_iterations;
    std::size_t psor_maximum_iterations_used;
    double psor_tolerance;
    double psor_relaxation;
    double maximum_lcp_residual;
    double maximum_relative_lcp_residual;

    std::vector<double> aligned_times;
    std::vector<PdeDividendEvent> dividend_events;
};

// Where a node sits against the exercise obstacle at the valuation time.
//
// The classification is read off the final implicit system, never off a
// post-hoc decimal tolerance. Let psi be the obstacle, x the solved slice,
// s = x - psi the obstacle slack, and rho = (A x - b) the linear residual of
// that final system. The solver's own convergence contract bounds the natural
// LCP residual by `psor_tolerance * max(1, ||b||_inf)`; call that bound the
// classification scale. A node is classified only when one of the two
// complementary slacks is certified nonzero at that scale:
//
// - `continuation`             when s > scale;
// - `exercise`                 when s <= scale and rho > scale;
// - `numerically_indifferent`  when both slacks sit inside the scale, so
//                              strict complementarity cannot be certified.
//
// `no_obstacle` is reported for every node of a European contract, where no
// complementarity problem is posed at all. It is not a quiet `continuation`.
enum class PdeExerciseState {
    continuation,
    exercise,
    numerically_indifferent,
    no_obstacle,
};

// Why a node or a query is not admissible as a Greek label. `eligible` is a
// member so the reason is always populated and never has to be inferred from a
// bare boolean. The reasons are tested in this declared precedence order.
enum class PdeGreekEligibilityReason {
    eligible,
    // Index 0 or the last index: a centered stencil does not exist and a
    // one-sided domain-boundary difference is deliberately not offered.
    centered_stencil_unavailable,
    inside_domain_boundary_buffer,
    non_finite_stencil_value,
    // The node's own exercise classification is `numerically_indifferent`: the
    // solver cannot certify at its declared residual scale whether this node
    // sits on the obstacle or strictly above it. A difference quotient there is
    // a well-defined number, but which quantity it estimates - the derivative of
    // an obstacle-clamped payoff or of a free PDE solution - is not resolved, so
    // it is refused as a Greek label.
    unresolved_exercise_state,
    // Some node of the five-node regime stencil carries a different exercise
    // classification, so the stencil crosses the free boundary or a
    // numerically indifferent band.
    regime_stencil_not_uniform,
};

[[nodiscard]] std::string_view pde_exercise_state_name(PdeExerciseState state);
[[nodiscard]] std::string_view pde_greek_eligibility_reason_name(PdeGreekEligibilityReason reason);

// One node of the valuation-time slice.
//
// `delta` and `gamma` are engaged exactly when a centered stencil exists and
// every value entering it is finite. They are *not* gated on eligibility: an
// ineligible node with an existing stencil still reports what the stencil says,
// and `greek_eligible` is the field that decides whether it may be used as a
// label. A node with no stencil reports no number at all rather than a
// plausible-looking one-sided value.
struct PdeSurfaceNode {
    std::size_t index;
    double spot;
    double value;
    // The exercise payoff at this node. For a European contract it is reported
    // for reference and no complementarity is posed against it.
    double obstacle;
    double obstacle_slack;
    // (A x - b)_i of the final implicit system, the second complementary slack.
    double lcp_linear_residual;
    PdeExerciseState exercise_state;
    std::optional<double> delta;
    std::optional<double> gamma;
    bool greek_eligible;
    PdeGreekEligibilityReason greek_eligibility_reason;
};

// The valuation-time slice of one backward induction, and nothing else.
//
// Only the final time level is retained, so working memory stays O(N_S): no
// space-time array is ever formed and no time history is exposed.
struct PdeValuationSurface {
    PdeSolveDiagnostics diagnostics;

    // Number of backward inductions behind this object. It is always 1, and it
    // is reported so a caller can prove that N queries cost one solve.
    std::size_t backward_inductions;

    double valuation_time;
    // psor_tolerance * max(1, ||b||_inf) of the final implicit system: the
    // absolute price-scale bound the solver's declared convergence contract
    // permits on the natural LCP residual, and the only threshold the exercise
    // classification uses.
    double exercise_classification_scale;
    std::size_t boundary_exclusion_nodes;
    std::size_t regime_stencil_radius;

    // Ascending in spot, one entry per grid node, index i at spot i * spot_step.
    std::vector<PdeSurfaceNode> nodes;

    // Throws std::logic_error if the node vector is not the length the reported
    // grid implies or is not in ascending index and spot order.
    void validate() const;
};

// One requested spot evaluated against an already-solved surface.
//
// `value` uses the identical interpolation the scalar API uses: monotone cubic
// Hermite.
//
// `delta` and `gamma` are **interpolated nodewise discrete Greeks**, not
// derivatives of that price interpolant. They are the linear blend of the two
// bracketing nodes' centered difference quotients, engaged only when both of
// those nodes are Greek-eligible; two eligible neighbours necessarily share an
// exercise regime, so an engaged query Greek never straddles the free boundary.
//
// The distinction is not cosmetic. Differentiating the Hermite price
// interpolant gives a different number: on the reference fixtures the two
// disagree by up to about 1e-4 in delta at mid-cell positions, which is larger
// than the nodewise delta's own agreement with Black-Scholes. A query placed
// exactly on a grid node is the consistent case and is exact: its value, delta
// and gamma are then bitwise the node's own.
struct PdeSurfaceQuery {
    // Position of this query in the requested vector. Order is preserved and
    // duplicates are neither removed nor reordered.
    std::size_t query_index;
    // Index of the first query with a bitwise-equal spot; equal to
    // `query_index` when this spot is requested only once.
    std::size_t first_occurrence_index;
    double spot;
    double value;
    std::size_t left_node_index;
    std::size_t right_node_index;
    // Engaged only when both bracketing nodes agree, so a query sitting across
    // the free boundary reports no regime rather than a guessed one.
    std::optional<PdeExerciseState> exercise_state;
    std::optional<double> delta;
    std::optional<double> gamma;
    bool greek_eligible;
    PdeGreekEligibilityReason greek_eligibility_reason;
};

struct PdeSurfaceEvaluation {
    PdeValuationSurface surface;
    std::vector<PdeSurfaceQuery> queries;
};

class PdePsorFailure : public std::runtime_error {
public:
    PdePsorFailure(const std::string& message, std::size_t aligned_segment,
                   std::size_t iterations, double residual, double tolerance);

    [[nodiscard]] PdeSolverStatus solver_status() const noexcept {
        return PdeSolverStatus::psor_iteration_limit_exceeded;
    }
    [[nodiscard]] std::size_t aligned_segment() const noexcept {
        return aligned_segment_;
    }
    [[nodiscard]] std::size_t iterations() const noexcept {
        return iterations_;
    }
    [[nodiscard]] double residual() const noexcept {
        return residual_;
    }
    [[nodiscard]] double tolerance() const noexcept {
        return tolerance_;
    }

private:
    std::size_t aligned_segment_;
    std::size_t iterations_;
    double residual_;
    double tolerance_;
};

// The spot immediately after a cash dividend: max(spot - amount, 0). Exposed
// because it is the whole content of the dividend jump in spot space and is
// worth testing on its own, including the spot < amount branch.
[[nodiscard]] double post_dividend_spot(double spot, double amount);

// Deterministic one-dimensional finite-difference price of a European or
// American vanilla option with explicit discrete cash dividends.
//
// Crank-Nicolson time stepping with Rannacher damping after the terminal payoff
// and after each dividend jump; the American obstacle is a linear
// complementarity problem solved by projected successive over-relaxation.
// Working memory is O(spot_intervals) plus the small aligned-time and
// dividend-event vectors: no space-time array is ever held.
//
// Throws std::invalid_argument for an invalid contract or grid, std::overflow_error
// when a price-domain quantity leaves double precision, and PdePsorFailure when
// PSOR does not reach the declared LCP residual within the iteration limit.
[[nodiscard]] PdeResult finite_difference_price(const PdeContract& contract, const PdeGrid& grid);

// Solve once and return the valuation-time slice with nodewise price, delta,
// gamma, exercise classification and Greek eligibility.
//
// The solved domain depends only on the strike and the grid settings, never on
// a requested spot, so one surface serves every spot in the domain. Working
// memory is the same O(N_S) as the scalar solve plus the returned node vector.
//
// Throws the same exceptions as `finite_difference_price`, and additionally
// std::invalid_argument when `spot_maximum` does not exceed the strike or the
// boundary exclusion buffer leaves no eligible interior node.
[[nodiscard]] PdeValuationSurface finite_difference_valuation_surface(
    const PdeSurfaceContract& contract, const PdeGrid& grid, const PdeSurfaceSettings& settings);

// Evaluate requested spots against an already-solved surface. No solve happens
// here. Input order is preserved, duplicates are kept and each spot must lie
// strictly inside the solved domain: extrapolation is refused with
// std::invalid_argument naming the offending query index, never silently
// clamped.
[[nodiscard]] std::vector<PdeSurfaceQuery> evaluate_valuation_surface(
    const PdeValuationSurface& surface, const std::vector<double>& spots);

// One solve followed by many queries. Equivalent to
// `evaluate_valuation_surface(finite_difference_valuation_surface(...), spots)`
// and provided so the common case cannot accidentally solve per spot.
[[nodiscard]] PdeSurfaceEvaluation finite_difference_valuation_surface_at(
    const PdeSurfaceContract& contract, const PdeGrid& grid, const PdeSurfaceSettings& settings,
    const std::vector<double>& spots);

}  // namespace dp
