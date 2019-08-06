module PIPS
using Libdl
using MPI

export PipsNlpProblemStruct, CallBackData
export createProblemStruct, solveProblemStruct, freeProblemStruct
export getPipsLib

const PIPSRetCode = Dict{Int, Symbol}(
    0=>:SUCCESSFUL_TERMINATION,
    1=>:NOT_FINISHED,
    2=>:MAX_ITS_EXCEEDED,
    3=>:INFEASIBLE,
    4=>:NEED_FEASIBILITY_RESTORATION,
    5=>:UNKNOWN
    )

PIPSLibfile = ENV["PIPS_NLP_PAR_SHARED_LIB"]
if !isfile(PIPSLibfile)
    error(string("The specified shared library ([", PIPSLibfile, "]) does not exist. Make sure the ENV variable 'PIPS_NLP_PAR_SHARED_LIB' points to its location, usually in the PIPS repo at PIPS/build_pips/PIPS-NLP/libparpipsnlp.so"))
end

function __init__()
    try
        global libparpipsnlp=Libdl.dlopen(get(ENV,"PIPS_NLP_PAR_SHARED_LIB",""))
    catch
        @warn("Could not load PIPS-NLP shared library. Make sure the ENV variable 'PIPS_NLP_PAR_SHARED_LIB' points to its location, usually in the PIPS repo at PIPS/build_pips/PIPS-NLP/libparpipsnlp.so")
        rethrow()
    end
end

function getPipsLib()
    libparpipsnlp
end

mutable struct PipsNlpProblemStruct
    ref::Ptr{Nothing}
    model # TODO: unclear, has to be removed
    comm::MPI.Comm
    prof::Bool
    x::Vector{Float64}  # Final solution
    g::Vector{Float64}  # Final constraint values
    obj_val::Float64  # Final objective
    n::Int
    x_L::Vector{Float64}
    x_U::Vector{Float64}
    m::Int
    g_L::Vector{Float64}
    g_U::Vector{Float64}
    nele_jac::Int
    nele_hess::Int

    n_iter::Int

    function PipsNlpProblemStruct(comm, model, prof, n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int)
        # prob = new(C_NULL, comm, prof,-3
        prob = new(C_NULL, model, comm, prof,
                   zeros(Float64, n), zeros(Float64, m), 0.0,
                   n, x_L, x_U, m, g_L, g_U,
                   nele_jac, nele_hess, -3
            )

        finalizer(freeProblemStruct, prob)

        return prob
    end
end

# Callback datastructure coming from PIPS
mutable struct CallBackData
	prob::Ptr{Nothing}
	row_node_id::Cint
    col_node_id::Cint
    flag::Cint
end

function init_x0_cb(x0_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    println("init_x0_cb")
    return Int32(1)
end

# prob info (prob_info)
function prob_info_cb(n_ptr::Ptr{Cint}, col_lb_ptr::Ptr{Float64}, col_ub_ptr::Ptr{Float64}, m_ptr::Ptr{Cint}, row_lb_ptr::Ptr{Float64}, row_ub_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    println("prob_info_cb")
	return Int32(1)
end
# Objective (eval_f)
function eval_f_cb(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, obj_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    println("eval_f_cb")
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_cb(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, eq_g_ptr::Ptr{Float64}, inq_g_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    println("eval_g_cb")
    return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_cb(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, grad_f_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    println("eval_grad_f_cb")
    return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_cb(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64},
	e_nz_ptr::Ptr{Cint}, e_values_ptr::Ptr{Float64}, e_row_ptr::Ptr{Cint}, e_col_ptr::Ptr{Cint},
	i_nz_ptr::Ptr{Cint}, i_values_ptr::Ptr{Float64}, i_row_ptr::Ptr{Cint}, i_col_ptr::Ptr{Cint},
	cbd::Ptr{CallBackData}
	)
    println("eval_jac_g_cb")
    return Int32(1)
end

# Hessian
function eval_h_cb(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, lambda_ptr::Ptr{Float64}, nz_ptr::Ptr{Cint}, values_ptr::Ptr{Float64}, row_ptr::Ptr{Cint}, col_ptr::Ptr{Cint}, cbd::Ptr{CallBackData})
    println("eval_h_cb")

    return Int32(1)
end

#write solution
function write_solution_cb(x_ptr::Ptr{Float64}, y_eq_ptr::Ptr{Float64}, y_ieq_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})

    return Int32(1)
end

###########################################################################
# C function wrappers
###########################################################################

function createProblemStruct(comm::MPI.Comm, model, prof::Bool, numscen::Int, n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int)
	# println(" createProblemStruct  -- julia")
    # TODO: Some callbacks to do not much
	c_init_x0_cb = @cfunction(init_x0_cb, Cint, (Ptr{Float64}, Ptr{CallBackData}) )
    c_prob_info_cb = @cfunction(prob_info_cb, Cint, (Ptr{Cint}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    c_eval_f_cb = @cfunction(eval_f_cb,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    c_eval_g_cb = @cfunction(eval_g_cb,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    c_eval_grad_f_cb = @cfunction(eval_grad_f_cb, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    c_eval_jac_g_cb = @cfunction(eval_jac_g_cb, Cint, (Ptr{Float64}, Ptr{Float64},
    	Ptr{Cint}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint},
    	Ptr{Cint}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint},
    	Ptr{CallBackData}))
    c_eval_h_cb = @cfunction(eval_h_cb, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{CallBackData}))
    c_write_solution_cb = @cfunction(write_solution_cb, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}))

    # println(" callback created ")
	@show libparpipsnlp
    prob = PipsNlpProblemStruct(comm, model, prof, n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess)
    # @show prob
    ret = ccall(Libdl.dlsym(libparpipsnlp,:CreatePipsNlpProblemStruct),Ptr{Nothing},
            (MPI.CComm,
            Cint, Ptr{Nothing}, Ptr{Nothing},
	    Ptr{Nothing}, Ptr{Nothing}, Ptr{Nothing},
	    Ptr{Nothing}, Ptr{Nothing}, Ptr{Nothing},Any
            ),
            MPI.CComm(comm),
            numscen,
            c_init_x0_cb,
            c_prob_info_cb,
            c_eval_f_cb,
            c_eval_g_cb,
            c_eval_grad_f_cb,
            c_eval_jac_g_cb,
            c_eval_h_cb,
            c_write_solution_cb,
            prob
            )
    # println(" ccall CreatePipsNlpProblemStruct done ")
    # @show ret
    @show prob
    if ret == C_NULL
        error("PIPS-NLP: Failed to construct problem.")
    else
        prob.ref = ret
    end
    # @show prob
    # println("end createProblemStruct - julia")
    return prob
end

function solveProblemStruct(prob::PipsNlpProblemStruct)
    # println("solveProblemStruct - julia")
    # @show prob

    ret = ccall(Libdl.dlsym(libparpipsnlp,:PipsNlpSolveStruct), Cint,
            (Ptr{Nothing},),
            prob.ref)
            # Nothing)
    # prob.model.set_status(Int(ret))

    # prob.t_jl_eval_total = report_total_now(prob)
    # @show prob
    return PIPSRetCode[ret]
end

function freeProblemStruct(prob::PipsNlpProblemStruct)
    # @show "freeProblemStruct"
    ret = ccall(Libdl.dlsym(libparpipsnlp,:FreePipsNlpProblemStruct),
            Nothing, (Ptr{Nothing},),
            prob.ref)
    # @show ret
    return ret
end




include("MOI_wrapper.jl")

end # module
