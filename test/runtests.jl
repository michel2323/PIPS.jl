push!(LOAD_PATH,"/home/michel/git/PIPS.jl/src")
using Revise
using PIPS
using Ipopt
using Test
using MPI

MPI.Init()

@testset "C API PIPS" begin
    # First of all, test that hs071 example works
    include("hs071_test.jl")
end

@testset "C API Ipopt" begin
    # First of all, test that hs071 example works
    include("hs071_test_ipopt.jl")
end

@testset "MathOptInterface PIPS" begin
    include("MOI_wrapper.jl")
end

# @testset "MathOptInterface Ipopt" begin
#     include("MOI_Ipopt_wrapper.jl")
# end
MPI.Finalize()
