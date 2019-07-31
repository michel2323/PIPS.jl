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
    println("init_x0_wrapper")
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
    println("prob_info_wrapper")
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(data.col_node_id)
    flag = Int(data.flag)
    @assert(rowid == colid)
	
	mode = (col_lb_ptr == C_NULL) ? (:Structure) : (:Values)
	if(mode==:Structure)
        if flag == 1
            if colid == 0
                n = 4
                m = 0
            else
                n = 0
                m = 0
            end 
        else
            if colid == 0
                n = 4
                m = 2
            else
                n = 0
                m = 0
            end 
            # n = getNumVars(instance.internalModel,id)
            # m = getNumCons(instance.internalModel,id)
    		# prob.model.set_num_rows(colid, m)
    		# prob.model.set_num_cols(colid, n)
            unsafe_store!(n_ptr,convert(Cint,n)::Cint)
        end
        unsafe_store!(m_ptr,convert(Cint,m)::Cint)
	else
        # This is for linking constraints which is not supported
        if flag == 1
            n = unsafe_load(n_ptr)
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
                col_lb[1] = 1.0
                col_lb[2] = 1.0
                col_lb[3] = 1.0
                col_lb[4] = 1.0
                col_ub[1] = 5.0
                col_ub[2] = 5.0
                col_ub[3] = 5.0
                col_ub[4] = 5.0
                row_lb[1] = 40.0
                row_lb[2] = 25.0
                row_ub[1] = 40.0
                row_ub[2] = 2.0e19
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
    println("eval_f_wrapper")
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(data.col_node_id)
    @assert(rowid == colid)
    # n0 = prob.model.get_num_cols(0)
    # n1 = prob.model.get_num_cols(colid)
    # Calculate the new objective
    n0 = 4
    n1 = 4
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    x1 = unsafe_wrap(Array, x1_ptr, n1)
    obj = x0[1] * x0[4] * (x0[1] + x0[2] + x0[3]) + x0[3]
    
    # new_obj = convert(Float64, prob.model.str_eval_f(colid,x0,x1))::Float64
    unsafe_store!(obj_ptr, obj)
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(x0_ptr::Ptr{Float64}, x1_ptr::Ptr{Float64}, eq_g_ptr::Ptr{Float64}, ineq_g_ptr::Ptr{Float64}, cbd::Ptr{CallBackData})
    println("eval_g_wrapper")
    # @show " julia - eval_g_wrapper " 
    data = unsafe_load(cbd)
    # @show data
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(data.row_node_id)
    colid = Int(data.col_node_id)
    @assert(rowid == colid)
    # n0 = prob.model.get_num_cols(0)
    # n1 = prob.model.get_num_cols(colid)
    n0 = 4
    n1 = 4
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
    println("eval_grad_f_wrapper")
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(data.row_node_id)
    colid = Int(data.col_node_id)
    n0 = 4
    n1 = 4
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    x1 = unsafe_wrap(Array, x1_ptr, n1)
    # grad_len = prob.model.get_num_cols(colid)
    grad_len = 4
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
    println("eval_jac_g_wrapper")
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(Int(data.row_node_id))
    colid = Int(Int(data.col_node_id))
    flag = Int(data.flag)
    # n0 = prob.model.get_num_cols(0)
    # n1 = prob.model.get_num_cols(rowid) #we can do this because of 2-level and no linking constraint
    n0 = 4
    # n1 = 4
    x0 = unsafe_wrap(Array, x0_ptr, n0)
    # x1 = unsafe_wrap(Array, x1_ptr, n1)
    # nrow = prob.model.get_num_rows(rowid) 
    # ncol = prob.model.get_num_cols(colid) 
    @show x0
    ncol = 4
    mode = (e_col_ptr == C_NULL && i_col_ptr == C_NULL) ? (:Structure) : (:Values)
    if flag != 1
        if mode == :Structure
            if colid == 0 && rowid == 0
                # e_values = unsafe_wrap(Array,e_values_ptr,0)
                # e_colptr = unsafe_wrap(Array,e_col_ptr,0)
                # e_rowidx = unsafe_wrap(Array,e_row_ptr,0)
                # i_values = unsafe_wrap(Array,i_values_ptr,0)
                # i_colptr = unsafe_wrap(Array,i_col_ptr,0)
                # i_rowidx = unsafe_wrap(Array,i_row_ptr,0)
                
                # (e_nz,i_nz) = prob.model.str_eval_jac_g(rowid,colid,flag,x0,x1,mode,e_rowidx,e_colptr,e_values,i_rowidx,i_colptr,i_values)
                e_nz = 4
                i_nz = 4
            else
                e_nz = 0
                i_nz = 0
            end
            unsafe_store!(e_nz_ptr,convert(Cint,e_nz)::Cint)
            unsafe_store!(i_nz_ptr,convert(Cint,i_nz)::Cint)
        else
            if colid == 0 && rowid == 0
                e_nz = unsafe_load(e_nz_ptr)
                @show e_nz
                if e_nz > 0
                    e_rowidx = unsafe_wrap(Array,e_row_ptr,e_nz)
                    e_colptr = unsafe_wrap(Array,e_col_ptr,ncol)
                    e_values = unsafe_wrap(Array,e_values_ptr,e_nz)
                    e_rowidx[1] = 0; #e_colptr[1] = 0
                    e_rowidx[2] = 1; #e_colptr[2] = 1
                    e_rowidx[3] = 2; #e_colptr[3] = 2
                    e_rowidx[4] = 3; #e_colptr[4] = 3
                    e_colptr[1] = 0
                    e_values[1] = 2*x0[1]  # 2,1
                    e_values[2] = 2*x0[2]  # 2,2
                    e_values[3] = 2*x0[3]  # 2,3
                    e_values[4] = 2*x0[4]  # 2,4
                end
                i_nz = unsafe_load(i_nz_ptr)
                @show i_nz
                @show ncol
                if i_nz > 0
                    # Constraint (row) 2
                    i_rowidx = unsafe_wrap(Array,i_row_ptr,i_nz)
                    i_colptr = unsafe_wrap(Array,i_col_ptr,ncol)
                    i_values = unsafe_wrap(Array,i_values_ptr,i_nz)
                    i_rowidx[1] = 0;# i_colptr[1] = 0
                    i_rowidx[2] = 1;# i_colptr[2] = 1
                    i_rowidx[3] = 2;# i_colptr[3] = 2
                    i_rowidx[4] = 3;# i_colptr[4] = 3
                    i_colptr[1] = 0
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
    println("eval_h_wrapper")
    data = unsafe_load(cbd)
    userdata = data.prob
    prob = unsafe_pointer_to_objref(userdata)::PipsNlpProblemStruct
    rowid = Int(data.row_node_id)
    colid = Int(data.col_node_id)
    flag = Int(data.flag)
    
    high = max(rowid,colid)
    low  = min(rowid,colid)
    n0 = 4
    # n0 = prob.model.get_num_cols(0) 
    # n1 = prob.model.get_num_cols(high)
    x0 = unsafe_wrap(Array,x0_ptr,n0)
    ncol = 4
    # x1 = unsafe_wrap(Array,x1_ptr,n1)
    # ncol = prob.model.get_num_cols(low)
    # g0 = prob.model.get_num_rows(high) 
    lambda = unsafe_wrap(Array,lambda_ptr,2)
    obj_factor = 1.0
    mode = (col_ptr == C_NULL) ? (:Structure) : (:Values)
    @show flag
    if mode == :Structure
		if (rowid == 0 && colid == 0) 
            nz = 10 
        else
            nz = 0
        end 
		unsafe_store!(nz_ptr,convert(Cint,nz)::Cint)
    else
    	nz = unsafe_load(nz_ptr)
        @show nz
        if nz > 0
            if colid == 0 && rowid == 0 
                values = unsafe_wrap(Array, values_ptr, nz)
                rowidx = unsafe_wrap(Array, row_ptr, nz)
                colptr = unsafe_wrap(Array, col_ptr, ncol+1)
                rowidx[1] = 0
                rowidx[2] = 1
                rowidx[3] = 2
                rowidx[4] = 3
                
                rowidx[5] = 1
                rowidx[6] = 2
                rowidx[7] = 3
                
                rowidx[8] = 2
                rowidx[9] = 3
                
                rowidx[10] = 3 
                
                colptr[1] = 0
                colptr[2] = 4
                colptr[3] = 7
                colptr[4] = 9
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
                values[5] += lambda[2] * (x0[1] * x0[4])  # 3,2
                values[7] += lambda[2] * (x0[2] * x0[3])  # 4,1
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
    println("write_solution_wrapper")
    return Int32(1)
end

function PIPScreateProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int)
    comm = MPI.COMM_WORLD
    prof::Bool = false
    model = Ptr{Cvoid}(C_NULL)
    numscen::Cint = 0
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
solveProblemStruct(prob)
freeProblemStruct(prob)

# prob.x = [1.0, 5.0, 5.0, 1.0]
# solvestat = solveProblem(prob)

