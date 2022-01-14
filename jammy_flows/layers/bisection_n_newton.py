import torch
import numpy

def close(a, b, rtol=1e-5, atol=1e-4):
    equal = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return equal


def inverse_bisection_n_newton_joint_func_and_grad(func, joint_func, target_arg, *args, min_boundary=-100000.0, max_boundary=100000.0, num_bisection_iter=25, num_newton_iter=30, newton_tolerance=1e-14, verbose=0):
    """
    Performs bisection and Newton iterations simulataneously in each 1-d subdimension in a given batch.
    Due to Newton iterations the returned inverse is differentiable and can be used in automatic differentiation.

    Parameters:
        func (function): The function to find the inverse of.
        grad_func (function): The gradient function of the function.
        target_arg (float Tensor): The argument at which the inverse functon should be evaluated. Tensor of size BXD where B is the batchsize, and D the dimension.
        *args (list): Any extra arguments passed to *func*.
        min_boundary (float): Minimum boundary for Bisection.
        max_boundary (float): Maximum boundary for bisection.
        num_bisection_iter (int): Number of bisection iterations.
        num_newton_iter (int): Number of Newton iterations.
    Returns:
        The (differentiable) inverse of the function *func* in each sub-dimension in each batch item.

    """
    new_upper = torch.tensor(max_boundary).type(torch.double).repeat(*target_arg.shape).to(target_arg.device)
    new_lower = torch.tensor(min_boundary).type(torch.double).repeat(*target_arg.shape).to(target_arg.device)
    
    mid=0
    for i in range(num_bisection_iter):
        mid = (new_upper + new_lower) / 2.
        #print("mid: ", mid)
        inverse_mid = func(mid, *args)

        #print("MID", mid)
        
        right_part = (inverse_mid < target_arg).double()
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, target_arg, rtol=1e-6, atol=0)).double()

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid

        
    prev=mid

    #print("target arg", target_arg.shape)


    above_tolerance_mask=torch.ones( target_arg.shape[0], dtype=torch.bool)

    ## check where we want to broadcast the masking, and wnhere not

    broadcasting_bool_args=[True if (prev.shape[0]>1 and arg.shape[0]>1) else False for arg in args ]

   

    for i in range(num_newton_iter):
       
        fn_result, f_prime_eval = joint_func(prev[above_tolerance_mask,:], *[a[above_tolerance_mask] if(broadcasting_bool_args[arg_index] == True) else a for arg_index, a in enumerate(args)])
        f_eval=fn_result-target_arg[above_tolerance_mask,:]

        update=(f_eval/f_prime_eval)

        newsource=prev[above_tolerance_mask,:]-update

        prev=torch.masked_scatter(input=prev, mask=above_tolerance_mask[:,None], source=newsource)

        non_finite_sum=(torch.isfinite(prev)==False).sum()
        if(non_finite_sum>0):


            print("NONZERO")
            print((torch.isfinite(prev)==False).nonzero())


            print("prev", prev[torch.isfinite(prev)==False])
            print("feval ", f_eval[torch.isfinite(prev)==False])
            print("f grad eval ", f_prime_eval[torch.isfinite(prev)==False])

            raise Exception()

        new_tolerance_mask=torch.abs(update)>=newton_tolerance

        above_tolerance_mask=torch.masked_scatter(input=above_tolerance_mask, mask=above_tolerance_mask, source=new_tolerance_mask)

        above_tol=above_tolerance_mask.sum()

        if(verbose):
            print("-- newton iter %d .. %d / %d dims completed" % (i, target_arg.shape[0]-above_tol, target_arg.shape[0]))
        if(above_tol==0):
            if(verbose):
                print("------ done")
            break


    num_non_converged=(torch.abs(f_eval)>1e-7).sum()

    #print(f_eval[torch.abs(f_eval)>1e-5])
    if( num_non_converged>0):
        print(num_non_converged, " items did not converge in Newton iterations")
        print("feval (diff) ",f_eval[torch.abs(f_eval)>1e-7])
        print("PREV VALUE:", prev[torch.abs(f_eval)>1e-7])
    
    return prev

## differentiable newton iterations
def inverse_bisection_n_newton(func, grad_func, target_arg, *args, min_boundary=-100000.0, max_boundary=100000.0, num_bisection_iter=25, num_newton_iter=30, newton_tolerance=1e-14, verbose=0):
    """
    Performs bisection and Newton iterations simulataneously in each 1-d subdimension in a given batch.
    Due to Newton iterations the returned inverse is differentiable and can be used in automatic differentiation.

    Parameters:
        func (function): The function to find the inverse of.
        grad_func (function): The gradient function of the function.
        target_arg (float Tensor): The argument at which the inverse functon should be evaluated. Tensor of size BXD where B is the batchsize, and D the dimension.
        *args (list): Any extra arguments passed to *func*.
        min_boundary (float): Minimum boundary for Bisection.
        max_boundary (float): Maximum boundary for bisection.
        num_bisection_iter (int): Number of bisection iterations.
        num_newton_iter (int): Number of Newton iterations.
    Returns:
        The (differentiable) inverse of the function *func* in each sub-dimension in each batch item.

    """
    new_upper = torch.tensor(max_boundary).type(torch.double).repeat(*target_arg.shape).to(target_arg.device)
    new_lower = torch.tensor(min_boundary).type(torch.double).repeat(*target_arg.shape).to(target_arg.device)
    
    mid=0
    for i in range(num_bisection_iter):
        mid = (new_upper + new_lower) / 2.
        #print("mid: ", mid)
        inverse_mid = func(mid, *args)

        #print("MID", mid)
        
        right_part = (inverse_mid < target_arg).double()
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, target_arg, rtol=1e-6, atol=0)).double()

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid

        
    prev=mid

    #print("target arg", target_arg.shape)


    above_tolerance_mask=torch.ones( target_arg.shape[0], dtype=torch.bool)

    broadcasting_bool_args=[True if (prev.shape[0]>1 and arg.shape[0]>1) else False for arg in args ]


    for i in range(num_newton_iter):
        #print("tol mask")
        #print(above_tolerance_mask)
        #print(prev.shape)
        #print(prev[above_tolerance_mask,:])
        #print("target_arg[above_tolerance_mask,:]", target_arg[above_tolerance_mask,:])

        masked_args=[a[above_tolerance_mask] if(broadcasting_bool_args[arg_index] == True) else a for arg_index, a in enumerate(args)]

        f_eval = func(prev[above_tolerance_mask,:], *masked_args)-target_arg[above_tolerance_mask,:]
     
        f_prime_eval=grad_func(prev[above_tolerance_mask,:], *masked_args) 

        update=(f_eval/f_prime_eval)

        #print("UPDATE", update)

        newsource=prev[above_tolerance_mask,:]-update

        prev=torch.masked_scatter(input=prev, mask=above_tolerance_mask[:,None], source=newsource)

        non_finite_sum=(torch.isfinite(prev)==False).sum()
        if(non_finite_sum>0):


            print("NONZERO")
            print((torch.isfinite(prev)==False).nonzero())


            print("prev", prev[torch.isfinite(prev)==False])
            print("feval ", f_eval[torch.isfinite(prev)==False])
            print("f grad eval ", f_prime_eval[torch.isfinite(prev)==False])

            raise Exception()

        new_tolerance_mask=torch.abs(update)>=newton_tolerance

        above_tolerance_mask=torch.masked_scatter(input=above_tolerance_mask, mask=above_tolerance_mask, source=new_tolerance_mask)

        above_tol=above_tolerance_mask.sum()

        if(verbose):
            print("-- newton iter %d .. %d / %d dims completed" % (i, target_arg.shape[0]-above_tol, target_arg.shape[0]))
        if(above_tol==0):
            if(verbose):
                print("------ done")
            break


    num_non_converged=(torch.abs(f_eval)>1e-7).sum()

    #print(f_eval[torch.abs(f_eval)>1e-5])
    if( num_non_converged>0):
        print(num_non_converged, " items did not converge in Newton iterations")
        print("feval (diff) ",f_eval[torch.abs(f_eval)>1e-7])
        print("PREV VALUE:", prev[torch.abs(f_eval)>1e-7])
    
    return prev

def inverse_bisection_n_newton_slow(func, grad_func, target_arg, *args, min_boundary=-100000.0, max_boundary=100000.0, num_bisection_iter=25, num_newton_iter=30):
    """
    Performs bisection and Newton iterations simulataneously in each 1-d subdimension in a given batch.
    Due to Newton iterations the returned inverse is differentiable and can be used in automatic differentiation.

    Parameters:
        func (function): The function to find the inverse of.
        grad_func (function): The gradient function of the function.
        target_arg (float Tensor): The argument at which the inverse functon should be evaluated. Tensor of size BXD where B is the batchsize, and D the dimension.
        *args (list): Any extra arguments passed to *func*.
        min_boundary (float): Minimum boundary for Bisection.
        max_boundary (float): Maximum boundary for bisection.
        num_bisection_iter (int): Number of bisection iterations.
        num_newton_iter (int): Number of Newton iterations.
    Returns:
        The (differentiable) inverse of the function *func* in each sub-dimension in each batch item.

    """
    new_upper = torch.tensor(max_boundary).type(torch.double).repeat(*target_arg.shape).to(target_arg.device)
    new_lower = torch.tensor(min_boundary).type(torch.double).repeat(*target_arg.shape).to(target_arg.device)
    #print("num iterations: ", iteration)
    #print("input z ... ", z)
    #print("INVESRE BISECTION ", "target shape ", target_arg.shape)
    mid=0
    for i in range(num_bisection_iter):
        mid = (new_upper + new_lower) / 2.
        #print("mid: ", mid)
        inverse_mid = func(mid, *args)

        #print("MID", mid)
        
        right_part = (inverse_mid < target_arg).double()
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, target_arg, rtol=1e-6, atol=0)).double()

        #print("inverse result ",inverse_mid)
        #print("CLOSE",correct_part )
      

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid


    prev=mid


    for i in range(num_newton_iter):
       
        inf_mask_pos=(prev>float("1e200")).double()
        inf_mask_neg=(prev<float("-1e200")).double()
        inf_mask_good=((inf_mask_pos==0) & (inf_mask_neg == 0)).double()
        
        prev=inf_mask_good*prev+inf_mask_pos*1e200+inf_mask_neg*(-1e200)
        
        f_eval = func(prev, *args)-target_arg

        f_prime_eval=grad_func(prev, *args)

        #print("prime eval ", f_prime_eval)
        non_finite_sum=(torch.isfinite(prev)==False).sum()

        prev=prev-(f_eval/f_prime_eval)

        if(non_finite_sum>0):


            print("NONZERO")
            print((torch.isfinite(prev)==False).nonzero())


            print("prev", prev[torch.isfinite(prev)==False])
            print("feval ", f_eval[torch.isfinite(prev)==False])
            print("f grad eval ", f_prime_eval[torch.isfinite(prev)==False])

            raise Exception()

        if(i==(num_newton_iter-1)):
            num_non_converged=(torch.abs(f_eval)>1e-7).sum()

            #print(f_eval[torch.abs(f_eval)>1e-5])
            if( num_non_converged>0):
                print(num_non_converged, " items did not converge in Newton iterations")
                print("feval (diff) ",f_eval[torch.abs(f_eval)>1e-7])
                print("PREV VALUE:", prev[torch.abs(f_eval)>1e-7])
    
    return prev





def inverse_bisection_n_newton_sphere(combined_func, find_orthogonal_vecs_func, basic_exponential_map_func, target_arg, *args, num_newton_iter=50):
    """
    Performs Newton iterations on the sphere over 1-dimensional potential functions via Exponential maps to find the inverse of a given exponential map on the sphere.
    In initial tests it was found that a very precise application requires at least 40-50 ierations, even though one is already pretty close after 10 iterations.
    Distributing the points randomly on the sphere dosnt really help, so every point is initialized just at (0,0,1).
    In order for this to work properly, the function has to be globally dipheomorphic.
    We follow https://arxiv.org/abs/0906.0874 ("A Jacobian inequality for gradient maps on the sphere and its application to directional statistics"),
    in particular the implementation suggested in https://arxiv.org/abs/2002.02428 ("Normalizing Flows on Tori and Spheres") to get this requirement satisfied for "exponential_map_flows".
    """

    ## start with very simple first guess ... could be done better probably

    first_guesses=[]
    pot_funcs=[]
    prev=torch.zeros_like(target_arg)
    prev[:,2]=-1.0
    
    """
    for dim in range(3):

        ## up
        prev[:,dim]=1.0

        first_guesses.append(prev.clone().unsqueeze(2))
        phi_res, _, jac_phi=combined_func(prev, *args)
        pot_func=-(phi_res*target_arg).sum(axis=1)
        
        pot_funcs.append(pot_func.unsqueeze(1))

        ## down
        prev[:,dim]=-1.0

        first_guesses.append(prev.clone().unsqueeze(2))
        phi_res, _, jac_phi=combined_func(prev, *args)
        pot_func=-(phi_res*target_arg).sum(axis=1)
        pot_funcs.append(pot_func.unsqueeze(1))

        prev[:,dim]=0.0

    #######################

    num_first_guesses=len(first_guesses)

    first_guesses=torch.cat(first_guesses, dim=2)
    first_guess_results=torch.cat(pot_funcs, dim=1)

    min_values,min_pot_indices=first_guess_results.min(dim=1)

    prev=first_guesses[torch.arange(first_guesses.shape[0]), :, min_pot_indices]
    
    """
    #test_arg=target_arg[0:1]#torch.Tensor([[1.0,0.0,0.0]]).to(target_arg)

    res_vec_return=0
    for i in range(num_newton_iter):
        
   
        phi_res, _, jac_phi=combined_func(prev, *args)


        
        ## minimize 
        #basic_pot_func=-(phi_res*target_arg).sum(axis=1)+1.0
        # OR
        #pot_func=0.5*(-(phi_res*target_arg).sum(axis=1)+1.0)**2
        #if(pot_func.max()<-1.000+1e-5):
        #    print("Potential minimum reached .. Breaking at iter ", i)
        #    break


        ## projection of 
        res_vec=-torch.bmm(jac_phi.permute(0,2,1), target_arg.unsqueeze(2))#*basic_pot_func.unsqueeze(1).unsqueeze(2)
        
        
        grad_lens=(res_vec**2).sum(axis=1, keepdims=True).sqrt()
      

        ## find the tangent vector along direction of negative gradient (newton iter)
       
        vs=find_orthogonal_vecs_func(prev, -res_vec/grad_lens, in_newton=True)
      
        projection=-(vs*res_vec).sum(axis=1)

      
        # We use the projection again in an exponential map.. this should not be larger than pi to not wrap arround the sphere more than maximally once.
        assert(projection.max()<numpy.pi)  

        ## mirror because both are negative by construction
        vs=-vs.squeeze(-1)
        projection=-projection

        ## A "Newton step" along the exponential map, where "vs" is the tangent vector, "projection" is its length or the step size, and prev is the previous point on the sphere.
        ## The next Point after this exponential map should be closer to "target_arg"

        #prev=prev*torch.cos(projection)+vs*torch.sin(projection)
        prev=basic_exponential_map_func(prev, vs, projection)
        
      
        #prev=new
    
    return prev
