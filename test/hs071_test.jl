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
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(Int(data.col_node_id))
    @assert(rowid == colid)
    # n0 = prob.model.get_num_cols(colid)
    x0 = unsafe_wrap(Array, x0_ptr, 4)
    x0[1] = 1.0
    x0[2] = 5.0
    x0[3] = 5.0
    x0[4] = 1.0
    # prob.model.str_init_x0(colid,x0)
            # @assert(id in getLocalBlocksIds(instance.internalModel))
            # mm = getModel(instance.internalModel,id)
            # nvar = getNumVars(instance.internalModel,id)
            # @assert length(x0) == nvar
            # 
            # for i=1:nvar
            #     x0[i] = getvalue(Variable(mm,i))
            #     isnan(x0[i]) ? x0[i]=1.0 : nothing
            # end
    return Int32(1)
end

# prob info (prob_info)
function prob_info_wrapper(n_ptr::Ptr{Cint}, col_lb_ptr::Ptr{Float64}, col_ub_ptr::Ptr{Float64}, m_ptr::Ptr{Cint}, row_lb_ptr::Ptr{Float64}, row_ub_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(data.col_node_id)
    flag = Int(data.flag)
    @assert(rowid == colid)
	
	mode = (col_lb_ptr == C_NULL) ? (:Structure) : (:Values)
	if mode==:Structure
        if flag == 1
            m = 0
            println("Only storing this")
            unsafe_store!(m_ptr,convert(Cint,m)::Cint)
        else
            if colid == 0
                n = prob.n
                m = prob.m
            end 
            # n = getNumVars(instance.internalModel,id)
            # m = getNumCons(instance.internalModel,id)
    		# prob.model.set_num_rows(colid, m)
    		# prob.model.set_num_cols(colid, n)
            println("Storing m and n")
            unsafe_store!(n_ptr,convert(Cint,n)::Cint)
            unsafe_store!(m_ptr,convert(Cint,m)::Cint)
        end
	else
        # This is for linking constraints which is not supported
        if flag == 1
            m = unsafe_load(m_ptr)
            @assert m==0
        else
    		n = unsafe_load(n_ptr)
    		m = unsafe_load(m_ptr)

    		col_lb = unsafe_wrap(Array,col_lb_ptr,n)
    		col_ub = unsafe_wrap(Array,col_ub_ptr,n)
    		row_lb = unsafe_wrap(Array,row_lb_ptr,m)
    		row_ub = unsafe_wrap(Array,row_ub_ptr,m)
    		# prob.model.str_prob_info(colid,flag,mode,col_lb,col_ub,row_lb,row_ub)
            if colid == 0
                col_lb[1] = prob.x_L[1]
                col_lb[2] = prob.x_L[2]
                col_lb[3] = prob.x_L[3]
                col_lb[4] = prob.x_L[4]
                col_ub[1] = prob.x_U[1]
                col_ub[2] = prob.x_U[2]
                col_ub[3] = prob.x_U[3]
                col_ub[4] = prob.x_U[4]
                row_lb[1] = prob.g_L[1]
                row_lb[2] = prob.g_L[2]
                row_ub[1] = prob.g_U[1]
                row_ub[2] = prob.g_U[2]
        		neq = 0
        		nineq = 0
        		for i = 1:length(row_lb)
        			if row_lb[i] == row_ub[i]
        				neq += 1
        			else
        				nineq += 1
        			end
        		end
        		@assert(neq+nineq == length(row_lb) == m)
            end
    		# prob.model.set_num_eq_cons(colid,neq)
    		# prob.model.set_num_ineq_cons(colid,nineq) 
        end
	end
	return Int32(1)
end
# Objective (eval_f)
function eval_f_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, obj_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(data.col_node_id)
    @assert(rowid == colid)
    # n0 = prob.model.get_num_cols(0)
    # n1 = prob.model.get_num_cols(colid)
    # Calculate the new objective
    n0 = prob.n
    n1 = prob.n
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    x1 = unsafe_wrap(Array, x1_ptr, n1)
    obj = x0[1] * x0[4] * (x0[1] + x0[2] + x0[3]) + x0[3]
    
    # new_obj = convert(Float64, prob.model.str_eval_f(colid,x0,x1))::Float64
    unsafe_store!(obj_ptr, obj)
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, eq_g_ptr::Ptr{Float64}, ineq_g_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    # @show " julia - eval_g_wrapper " 
    data = unsafe_load(cbd)
    # @show data
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(data.row_node_id)
    colid = Int(data.col_node_id)
    @assert(rowid == colid)
    # n0 = probd.model.get_num_cols(0)
    # n1 = prob.model.get_num_cols(colid)
    n0 = prob.n
    n1 = prob.n
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    x1 = unsafe_wrap(Array, x1_ptr, n1)
    # Calculate the new constraint values
    # neq = prob.model.get_num_eq_cons(rowid)
    # nineq = prob.model.get_num_ineq_cons(rowid)
    neq = 1
    nineq = 1
    eq_g = unsafe_wrap(Array, eq_g_ptr, neq)
    ineq_g = unsafe_wrap(Array, ineq_g_ptr, nineq)
    eq_g[1] = x0[1]^2 + x0[2]^2 + x0[3]^2 + x0[4]^2
    ineq_g[1] = x0[1]   * x0[2]   * x0[3]   * x0[4]
    
    return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, grad_f_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(data.row_node_id)
    colid = Int(data.col_node_id)
    n0 = prob.n
    n1 = prob.n
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    x1 = unsafe_wrap(Array, x1_ptr, n1)
    # grad_len = prob.model.get_num_cols(colid)
    grad_len = prob.n
    grad_f = unsafe_wrap(Array, grad_f_ptr, grad_len)
    grad_f[1] = x0[1] * x0[4] + x0[4] * (x0[1] + x0[2] + x0[3])
    grad_f[2] = x0[1] * x0[4]
    grad_f[3] = x0[1] * x0[4] + 1.0
    grad_f[4] = x0[1] * (x0[1] + x0[2] + x0[3])

    # TODO: Gotta think about this
    # if prob.model.get_sense() == :Max
    #     new_grad_f *= -1.0
    # end
    return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, 
	e_nz_ptr::Ptr{Cint}, e_values_ptr::Ptr{Float64}, e_row_ptr::Ptr{Cint}, e_col_ptr::Ptr{Cint}, 
	i_nz_ptr::Ptr{Cint}, i_values_ptr::Ptr{Float64}, i_row_ptr::Ptr{Cint}, i_col_ptr::Ptr{Cint},  
	cbd::Ptr{CallBackData}
	)
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(Int(data.col_node_id))
    flag = Int(data.flag)
    # n0 = prob.model.get_num_cols(0)
    # n1 = prob.model.get_num_cols(rowid) #we can do this because of 2-level and no linking constraint
    n0 = prob.n
    n1 = prob.n
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    x1 = unsafe_wrap(Array, x1_ptr, n1)
    # nrow = prob.model.get_num_rows(rowid) 
    # ncol = prob.model.get_num_cols(colid) 
    ncol = prob.n
    mode = (e_col_ptr == C_NULL && i_col_ptr == C_NULL) ? (:Structure) : (:Values)
    if flag != 1
        if mode == :Structure
            if colid == 0 && rowid == 0
                e_values = unsafe_wrap(Array,e_values_ptr,0)
                e_colptr = unsafe_wrap(Array,e_col_ptr,0)
                e_rowidx = unsafe_wrap(Array,e_row_ptr,0)
                i_values = unsafe_wrap(Array,i_values_ptr,0)
                i_colptr = unsafe_wrap(Array,i_col_ptr,0)
                i_rowidx = unsafe_wrap(Array,i_row_ptr,0)
                
                # (e_nz,i_nz) = prob.model.str_eval_jac_g(rowid,colid,flag,x0,x1,mode,e_rowidx,e_colptr,e_values,i_rowidx,i_colptr,i_values)
                e_nz::Cint = prob.n
                i_nz::Cint = prob.n
                # else
                #     e_nz = 0
                #     i_nz = 0
                # end
                unsafe_store!(e_nz_ptr,convert(Cint,e_nz)::Cint)
                unsafe_store!(i_nz_ptr,convert(Cint,i_nz)::Cint)
            end
        else
            if colid == 0 && rowid == 0
                e_nz = unsafe_load(e_nz_ptr)
                i_nz = unsafe_load(i_nz_ptr)
                if e_nz > 0
                    e_rowidx = unsafe_wrap(Array,e_row_ptr,e_nz)
                    e_colptr = unsafe_wrap(Array,e_col_ptr,ncol+1)
                    e_values = unsafe_wrap(Array,e_values_ptr,e_nz)
                    e_rowidx[1] = 0
                    e_rowidx[2] = 0
                    e_rowidx[3] = 0
                    e_rowidx[4] = 0
                    e_colptr[1] = 0
                    e_colptr[2] = 1
                    e_colptr[3] = 2
                    e_colptr[4] = 3
                    e_colptr[5] = 4
                    e_values[1] = 2*x0[1]  # 2,1
                    e_values[2] = 2*x0[2]  # 2,2
                    e_values[3] = 2*x0[3]  # 2,3
                    e_values[4] = 2*x0[4]  # 2,4
                end
                if i_nz > 0
                    # Constraint (row) 2
                    i_rowidx = unsafe_wrap(Array,i_row_ptr,i_nz)
                    i_colptr = unsafe_wrap(Array,i_col_ptr,ncol+1)
                    i_values = unsafe_wrap(Array,i_values_ptr,i_nz)
                    i_rowidx[1] = 0 
                    i_rowidx[2] = 0
                    i_rowidx[3] = 0
                    i_rowidx[4] = 0
                    i_colptr[1] = 0
                    i_colptr[2] = 1
                    i_colptr[3] = 2
                    i_colptr[4] = 3
                    i_colptr[5] = 4
                    # prob.model.str_eval_jac_g(rowid,colid,flag,x0,x1,mode,e_rowidx,e_colptr,e_values,i_rowidx,i_colptr,i_values)
                    # Constraint (row) 2
                    i_values[1] = x0[2]*x0[3]*x0[4]  # 1,1
                    i_values[2] = x0[1]*x0[3]*x0[4]  # 1,2
                    i_values[3] = x0[1]*x0[2]*x0[4]  # 1,3
                    i_values[4] = x0[1]*x0[2]*x0[3]  # 1,4
                end
                return Int32(1)
            end
        end
    else
        @assert flag == 1
        if mode == :Structure
            e_nz = 0
            i_nz = 0
            unsafe_store!(e_nz_ptr,convert(Cint,e_nz)::Cint)
            unsafe_store!(i_nz_ptr,convert(Cint,i_nz)::Cint)
        else
            e_nz = unsafe_load(e_nz_ptr)
            i_nz = unsafe_load(i_nz_ptr)
            @assert e_nz == i_nz == 0
        end
    end
    return Int32(1)
end

# Hessian
function eval_h_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, lambda_ptr::Ptr{Float64}, nz_ptr::Ptr{Cint}, values_ptr::Ptr{Float64}, row_ptr::Ptr{Cint}, col_ptr::Ptr{Cint}, cbd::Ptr{CallBackData})
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(data.row_node_id)
    colid = Int(data.col_node_id)
    flag = Int(data.flag)
    
    high = max(rowid,colid)
    low  = min(rowid,colid)
    n0 = prob.n
    # n0 = prob.model.get_num_cols(0) 
    # n1 = prob.model.get_num_cols(high)
    x0 = unsafe_wrap(Array,x0_ptr,n0)
    ncol = prob.n
    # x1 = unsafe_wrap(Array,x1_ptr,n1)
    # ncol = prob.model.get_num_cols(low)
    # g0 = prob.model.get_num_rows(high) 
    lambda = unsafe_wrap(Array,lambda_ptr,2)
    obj_factor = 1.0
    mode = (col_ptr == C_NULL) ? (:Structure) : (:Values)
    if mode == :Structure
		if (rowid == 0 && colid == 0) 
            nz = prob.nele_hess 
        else
            nz = 0
        end 
		unsafe_store!(nz_ptr,convert(Cint,nz)::Cint)
    else
    	nz = unsafe_load(nz_ptr)
        if nz > 0
            if colid == 0 && rowid == 0 
                values = unsafe_wrap(Array, values_ptr, nz)
                rowidx = unsafe_wrap(Array, row_ptr, nz)
                colptr = unsafe_wrap(Array, col_ptr, ncol+1)
                rowidx[1] = 0
                
                rowidx[2] = 0
                rowidx[3] = 1
                
                rowidx[4] = 0
                rowidx[5] = 1
                rowidx[6] = 2
                
                rowidx[7] = 0
                rowidx[8] = 1
                rowidx[9] = 2
                rowidx[10] = 3 
                
                colptr[1] = 0
                colptr[2] = 1
                colptr[3] = 3
                colptr[4] = 6
                colptr[5] = 10
                # return Int32(1)
                
                values[1] = obj_factor * (2*x0[4])  # 1,1
                values[2] = obj_factor * (  x0[4])  # 2,1
                values[3] = 0                      # 2,2
                values[4] = obj_factor * (  x0[4])  # 3,1
                values[5] = 0                      # 3,2
                values[6] = 0                      # 3,3
                values[7] = obj_factor * (2*x0[1] + x0[2] + x0[3])  # 4,1
                values[8] = obj_factor * (  x0[1])  # 4,2
                values[9] = obj_factor * (  x0[1])  # 4,3
                values[10] = 0                     # 4,4
                
                # First constraint
                values[2] += lambda[2] * (x0[3] * x0[4])  # 2,1
                values[4] += lambda[2] * (x0[2] * x0[4])  # 3,1
                values[5] += lambda[2] * (x0[1] * x0[4])  # 4,1
                values[7] += lambda[2] * (x0[2] * x0[3])  # 3,2
                values[8] += lambda[2] * (x0[1] * x0[3])  # 4,2
                values[9] += lambda[2] * (x0[1] * x0[2])  # 4,3
                
                # Second constraint
                values[1]  += lambda[1] * 2  # 1,1
                values[3]  += lambda[1] * 2  # 2,2
                values[6]  += lambda[1] * 2  # 3,3
                values[10] += lambda[1] * 2  # 4,4
                # prob.model.str_eval_h(rowid,colid,flag,x0,x1,obj_factor,lambda,mode,rowidx,colptr,values)
            end
        end
    end
    
    return Int32(1)
end

#write solution
function write_solution_wrapper(x_ptr::Ptr{Float64}, y_eq_ptr::Ptr{Float64}, y_ieq_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    x = unsafe_wrap(Array,x_ptr,4)
    @test x[1] ≈ 1.0000000000000000 atol=1e-5
    @test x[2] ≈ 4.7429996418092970 atol=1e-5
    @test x[3] ≈ 3.8211499817883077 atol=1e-5
    @test x[4] ≈ 1.3794082897556983 atol=1e-5
    return Int32(1)
end

function PIPScreateProblem(comm::MPI.Comm, model, prof::Bool, numscen::Int, n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int)
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
    prob = PipsNlpProblemStruct(comm, model, prof, n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess)
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
g_L = [40.0, 25.0]
g_U = [40.0, 2.0e19]

comm = MPI.COMM_WORLD
model = nothing
prof = false
numscen = 0

prob = PIPScreateProblem(comm, model, prof, numscen, n, x_L, x_U, m, g_L, g_U, 8, 10)
@test solveProblemStruct(prob) == :SUCCESSFUL_TERMINATION
# @test PIPSRetCode(solveProblemStruct(prob)) == :SUCCESSFUL_TERMINATION
freeProblemStruct(prob)

# prob.x = [1.0, 5.0, 5.0, 1.0]
# solvestat = solveProblem(prob)

