push!(LOAD_PATH,"/home/michel/git/PIPS.jl/src")
using PIPS
using Test
using MPI
using Libdl

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

function init_x0_wrapper(x0_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
end

# prob info (prob_info)
function prob_info_wrapper(n_ptr::Ptr{Cint}, col_lb_ptr::Ptr{Float64}, col_ub_ptr::Ptr{Float64}, m_ptr::Ptr{Cint}, row_lb_ptr::Ptr{Float64}, row_ub_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
end
# Objective (eval_f)
function eval_f_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, obj_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
end

# Constraints (eval_g)
function eval_g_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, eq_g_ptr::Ptr{Float64}, inq_g_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, grad_f_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, 
	e_nz_ptr::Ptr{Cint}, e_values_ptr::Ptr{Float64}, e_row_ptr::Ptr{Cint}, e_col_ptr::Ptr{Cint}, 
	i_nz_ptr::Ptr{Cint}, i_values_ptr::Ptr{Float64}, i_row_ptr::Ptr{Cint}, i_col_ptr::Ptr{Cint},  
	cbd::Ptr{CallBackData}
	)
end

# Hessian
function eval_h_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, lambda_ptr::Ptr{Float64}, nz_ptr::Ptr{Cint}, values_ptr::Ptr{Float64}, row_ptr::Ptr{Cint}, col_ptr::Ptr{Cint}, cbd::Ptr{CallBackData})
end

#write solution
function write_solution_wrapper(x_ptr::Ptr{Float64}, y_eq_ptr::Ptr{Float64}, y_ieq_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
end

function PIPScreateProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int)
    comm = MPI.COMM_WORLD
    prof::Bool = false
    model = Ptr{Cvoid}(C_NULL)
    numscen::Cint = 1
	# println(" createProblemStruct  -- julia")
    # TODO: Some callbacks to do not much
	str_init_x0_cb = @cfunction(init_x0_wrapper, Cint, (Ptr{Float64}, Ptr{CallBackData}) )
    str_prob_info_cb = @cfunction(prob_info_wrapper, Cint, (Ptr{Cint}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    str_eval_f_cb = @cfunction(eval_f_wrapper,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    str_eval_g_cb = @cfunction(eval_g_wrapper,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    str_eval_grad_f_cb = @cfunction(eval_grad_f_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}) )
    str_eval_jac_g_cb = @cfunction(eval_jac_g_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, 
    	Ptr{Cint}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, 
    	Ptr{Cint}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, 
    	Ptr{CallBackData}))
    str_eval_h_cb = @cfunction(eval_h_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{CallBackData}))
    str_write_solution_cb = @cfunction(write_solution_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{CallBackData}))
    
    # println(" callback created ")
    prob = PipsNlpProblemStruct(comm, model, prof)
    # @show prob
    ret = ccall(Libdl.dlsym(getPipsLib(),:CreatePipsNlpProblemStruct),Ptr{Nothing},
            (MPI.CComm, 
            Cint, Ptr{Nothing}, Ptr{Nothing}, 
	    Ptr{Nothing}, Ptr{Nothing}, Ptr{Nothing}, 
	    Ptr{Nothing}, Ptr{Nothing}, Ptr{Nothing},Any
            ),
            MPI.CComm(comm), 
            numscen,
            str_init_x0_cb,
            str_prob_info_cb,
            str_eval_f_cb, 
            str_eval_g_cb,
            str_eval_grad_f_cb, 
            str_eval_jac_g_cb, 
            str_eval_h_cb,
            str_write_solution_cb,
            prob
            )
    # println(" ccall CreatePipsNlpProblemStruct done ")
    # @show ret   
    
    if ret == C_NULL
        error("PIPS-NLP: Failed to construct problem.")
    else
        prob.ref = ret
    end
    # @show prob
    # println("end createProblemStruct - julia")
    return prob
end

n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

prob = PIPScreateProblem(n, x_L, x_U, m, g_L, g_U, 8, 10)

# prob.x = [1.0, 5.0, 5.0, 1.0]
# solvestat = solveProblem(prob)

