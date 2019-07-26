module PIPS
using Libdl
using MPI

PIPSLibfile = ENV["PIPS_NLP_PAR_SHARED_LIB"]
if !isfile(PIPSLibfile)
    error(string("The specified shared library ([", PIPSLibfile, "]) does not exist. Make sure the ENV variable 'PIPS_NLP_PAR_SHARED_LIB' points to its location, usually in the PIPS repo at PIPS/build_pips/PIPS-NLP/libparpipsnlp.so"))
end  

export createProblem, addOption
export openOutputFile, setProblemScaling, setIntermediateCallback
export solveProblem
export IpoptProblem

function __init__()
    try
        global libparpipsnlp=Libdl.dlopen(get(ENV,"PIPS_NLP_PAR_SHARED_LIB",""))
    catch 
        @warn("Could not load PIPS-NLP shared library. Make sure the ENV variable 'PIPS_NLP_PAR_SHARED_LIB' points to its location, usually in the PIPS repo at PIPS/build_pips/PIPS-NLP/libparpipsnlp.so")
        rethrow()
    end
end

mutable struct PipsNlpProblemStruct
    ref::Ptr{Nothing}
    # model::ModelInterface
    comm::MPI.Comm
    prof::Bool

    n_iter::Int
    t_jl_init_x0::Float64
    t_jl_str_prob_info::Float64
    t_jl_eval_f::Float64
    t_jl_eval_g::Float64
    t_jl_eval_grad_f::Float64

    t_jl_eval_jac_g::Float64
    t_jl_str_eval_jac_g::Float64
    t_jl_eval_h::Float64
    t_jl_str_eval_h::Float64
    t_jl_write_solution::Float64

    t_jl_str_total::Float64
    t_jl_eval_total::Float64
    
    function PipsNlpProblemStruct(comm, model, prof)
        prob = new(C_NULL, comm, prof,-3
        # prob = new(C_NULL, model, comm, prof,-3
            ,0.0,0.0,0.0,0.0,0.0
            ,0.0,0.0,0.0,0.0,0.0
            ,0.0,0.0
            )
        finalizer(freeProblemStruct, prob)
        
        return prob
    end
end



include("MOI_wrapper.jl")

end # module
