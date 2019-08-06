using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

MOIU.@model(IpoptModelData,
            (),
            (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan),
            (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives),
            (),
            (MOI.SingleVariable,),
            (MOI.ScalarAffineFunction, MOI.ScalarQuadraticFunction),
            (MOI.VectorOfVariables,),
            (MOI.VectorAffineFunction,))

# Without fixed_variable_treatment set, duals are not computed for variables
# that have lower_bound == upper_bound.
const ipopt_optimizer = Ipopt.Optimizer(print_level=0, fixed_variable_treatment="make_constraint")
const ipopt_config = MOIT.TestConfig(atol=1e-4, rtol=1e-4,
                               optimal_status=MOI.LOCALLY_SOLVED)

@testset "SolverName" begin
    @test MOI.get(ipopt_optimizer, MOI.SolverName()) == "Ipopt"
end

@testset "MOI Linear tests" begin
    exclude = ["linear8a", # Behavior in infeasible case doesn't match test.
               "linear12", # Same as above.
               "linear8b", # Behavior in unbounded case doesn't match test.
               "linear8c", # Same as above.
               "linear7",  # VectorAffineFunction not supported.
               "linear15", # VectorAffineFunction not supported.
               ]
    model_for_ipopt = MOIU.UniversalFallback(IpoptModelData{Float64}())
    linear_optimizer = MOI.Bridges.SplitInterval{Float64}(
                         MOIU.CachingOptimizer(model_for_ipopt, ipopt_optimizer))
    MOIT.contlineartest(linear_optimizer, ipopt_config, exclude)
end

MOI.empty!(ipopt_optimizer)

@testset "MOI QP/QCQP tests" begin
    qp_optimizer = MOIU.CachingOptimizer(IpoptModelData{Float64}(), ipopt_optimizer)
    MOIT.qptest(qp_optimizer, ipopt_config)
    exclude = ["qcp1", # VectorAffineFunction not supported.
              ]
    MOIT.qcptest(qp_optimizer, ipopt_config, exclude)
end

MOI.empty!(ipopt_optimizer)

@testset "MOI NLP tests" begin
    MOIT.nlptest(ipopt_optimizer, ipopt_config)
end
