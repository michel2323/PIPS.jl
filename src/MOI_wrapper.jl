import MathOptInterface
const MOI = MathOptInterface

mutable struct VariableInfo
    lower_bound::Float64  # May be -Inf even if has_lower_bound == true
    has_lower_bound::Bool # Implies lower_bound == Inf
    lower_bound_dual_start::Union{Nothing, Float64}
    upper_bound::Float64  # May be Inf even if has_upper_bound == true
    has_upper_bound::Bool # Implies upper_bound == Inf
    upper_bound_dual_start::Union{Nothing, Float64}
    is_fixed::Bool        # Implies lower_bound == upper_bound and !has_lower_bound and !has_upper_bound.
    start::Union{Nothing, Float64}
end

VariableInfo() = VariableInfo(-Inf, false, nothing, Inf, false, nothing, false, nothing)

mutable struct ConstraintInfo{F, S}
    func::F
    set::S
    dual_start::Union{Nothing, Float64}
end

ConstraintInfo(func, set) = ConstraintInfo(func, set, nothing)

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{PipsNlpProblemStruct,Nothing}
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{MOI.SingleVariable,MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64},Nothing}
    linear_le_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}
    linear_ge_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}
    linear_eq_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}
    quadratic_le_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}}}
    quadratic_ge_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}, MOI.GreaterThan{Float64}}}
    quadratic_eq_constraints::Vector{ConstraintInfo{MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64}}}
    nlp_dual_start::Union{Nothing, Vector{Float64}}
end

struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

function Optimizer()
    return Optimizer(nothing, [], empty_nlp_data(), MOI.FEASIBILITY_SENSE,
                     nothing, [], [], [], [], [], [],
                     nothing)
end
