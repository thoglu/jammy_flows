import torch
import numpy
import pylab
import time

def close(a, b, rtol=1e-5, atol=1e-4):
    equal = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return equal


def inverse_bisection_n_newton_joint_func_and_grad(func, 
                                                   joint_func, 
                                                   target_arg, 
                                                   *args, 
                                                   min_boundary=-100000.0, 
                                                   max_boundary=100000.0, 
                                                   num_bisection_iter=25, 
                                                   num_newton_iter=30, 
                                                   newton_tolerance=1e-14, 
                                                   verbose=0):
    """
    Performs bisection and Newton iterations simulataneously in each 1-d subdimension in a given batch.

    Parameters:

        func (function): The function to find the inverse of.
        joint_func (function): A function that calculates the function value and its derivative simultaneously.
        target_arg (float Tensor): The argument at which the inverse functon should be evaluated. Tensor of size (B,D) where B is the batchsize, and D the dimension.
        *args (list): Any extra arguments passed to *func*.
        min_boundary (float): Minimum boundary for Bisection.
        max_boundary (float): Maximum boundary for bisection.
        num_bisection_iter (int): Number of bisection iterations.
        num_newton_iter (int): Number of Newton iterations.

    Returns:

        Tensor
            The inverse of the function *func* in each sub-dimension in each batch item.

    """
    new_upper = torch.tensor(max_boundary).type(target_arg.dtype).repeat(*target_arg.shape).to(target_arg.device)
    new_lower = torch.tensor(min_boundary).type(target_arg.dtype).repeat(*target_arg.shape).to(target_arg.device)
 
    mid=0
    for i in range(num_bisection_iter):
        mid = (new_upper + new_lower) / 2.
        #print("mid: ", mid)
        inverse_mid = func(mid, *args)

        #print("MID", mid)
        
        right_part = (inverse_mid < target_arg).type(target_arg.dtype)
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, target_arg, rtol=1e-6, atol=0)).type(target_arg.dtype)

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid
      
        
    prev=mid

    #print("target arg", target_arg.shape)


    above_tolerance_mask=torch.ones( target_arg.shape[0], dtype=torch.bool, device=target_arg.device)

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

        new_tolerance_mask=(torch.abs(update).sum(axis=1))>=newton_tolerance
    
        above_tolerance_mask=torch.masked_scatter(input=above_tolerance_mask, mask=above_tolerance_mask, source=new_tolerance_mask)

        above_tol=above_tolerance_mask.sum()

        if(verbose):
            print("-- newton iter %d .. %d / %d dims completed" % (i, target_arg.shape[0]-above_tol, target_arg.shape[0]))
        if(above_tol==0):
            if(verbose):
                print("------ done")
            break

    if(target_arg.dtype==torch.float64):

        target_prec=1e-7
    else:

        target_prec=1e-4

    num_non_converged=(torch.abs(f_eval)>target_prec).sum()
    
    if( num_non_converged>0):
        print(num_non_converged, " items did not converge in Newton iterations")
        print("feval (diff) ",f_eval[torch.abs(f_eval)>target_prec])
    
    return prev

def inverse_bisection_n_newton(func, 
                               grad_func, 
                               target_arg, 
                               *args, 
                               min_boundary=-100000.0, 
                               max_boundary=100000.0, 
                               num_bisection_iter=25, 
                               num_newton_iter=30, 
                               newton_tolerance=1e-14, 
                               verbose=0):
    """
    Performs bisection and Newton iterations simulataneously in each 1-d subdimension in a given batch.

    Parameters:
    
        func (function): The function to find the inverse of.
        grad_func (function): The gradient function of the function.
        target_arg (float Tensor): The argument at which the inverse functon should be evaluated. Tensor of size (B,D) where B is the batchsize, and D the dimension.
        *args (list): Any extra arguments passed to *func*.
        min_boundary (float): Minimum boundary for Bisection.
        max_boundary (float): Maximum boundary for bisection.
        num_bisection_iter (int): Number of bisection iterations.
        num_newton_iter (int): Number of Newton iterations.

    Returns:

        Tensor
            The inverse of the function *func* in each sub-dimension in each batch item.

    """
    new_upper = torch.tensor(max_boundary).type(target_arg.dtype).repeat(*target_arg.shape).to(target_arg.device)
    new_lower = torch.tensor(min_boundary).type(target_arg.dtype).repeat(*target_arg.shape).to(target_arg.device)
    
    mid=0
    for i in range(num_bisection_iter):
        mid = (new_upper + new_lower) / 2.
      
        inverse_mid = func(mid, *args)

        right_part = (inverse_mid < target_arg).type(target_arg.dtype)
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, target_arg, rtol=1e-6, atol=0)).type(target_arg.dtype)

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid

        
    prev=mid

    above_tolerance_mask=torch.ones( target_arg.shape[0], dtype=torch.bool, device=target_arg.device)

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

        new_tolerance_mask=(torch.abs(update).sum(axis=1))>=newton_tolerance

        above_tolerance_mask=torch.masked_scatter(input=above_tolerance_mask, mask=above_tolerance_mask, source=new_tolerance_mask)
      
        above_tol=above_tolerance_mask.sum()

        if(verbose):
            print("-- newton iter %d .. %d / %d dims completed" % (i, target_arg.shape[0]-above_tol, target_arg.shape[0]))
        if(above_tol==0):
            if(verbose):
                print("------ done")
            break


    num_non_converged=(torch.abs(f_eval)>1e-7).sum()

    if(target_arg.dtype==torch.float64):

        target_prec=1e-7
    else:

        target_prec=1e-4

    num_non_converged=(torch.abs(f_eval)>target_prec).sum()
    
    if( num_non_converged>0):
        print(num_non_converged, " items did not converge in Newton iterations")
        print("feval (diff) ",f_eval[torch.abs(f_eval)>target_prec])
    
    return prev

def inverse_bisection_n_newton_slow(func, grad_func, target_arg, *args, min_boundary=-100000.0, max_boundary=100000.0, num_bisection_iter=25, num_newton_iter=30):
   
    new_upper = torch.tensor(max_boundary).type(target_arg.dtype).repeat(*target_arg.shape).to(target_arg.device)
    new_lower = torch.tensor(min_boundary).type(target_arg.dtype).repeat(*target_arg.shape).to(target_arg.device)
    #print("num iterations: ", iteration)
    #print("input z ... ", z)
    #print("INVESRE BISECTION ", "target shape ", target_arg.shape)
    mid=0
    for i in range(num_bisection_iter):
        mid = (new_upper + new_lower) / 2.
        #print("mid: ", mid)
        inverse_mid = func(mid, *args)

        #print("MID", mid)
        
        right_part = (inverse_mid < target_arg).type(target_arg.dtype)
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, target_arg, rtol=1e-6, atol=0)).type(target_arg.dtype)

        #print("inverse result ",inverse_mid)
        #print("CLOSE",correct_part )
      

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid


    prev=mid


    for i in range(num_newton_iter):
       
        inf_mask_pos=(prev>float("1e200")).type(target_arg.dtype)
        inf_mask_neg=(prev<float("-1e200")).type(target_arg.dtype)
        inf_mask_good=((inf_mask_pos==0) & (inf_mask_neg == 0)).type(target_arg.dtype)
        
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

def inverse_bisection_n_newton_sphere(combined_func, 
                                      find_tangent_func, 
                                      basic_exponential_map_func, 
                                      target_arg, 
                                      *args, 
                                      num_newton_iter=25):
    """
    Performs Newton iterations on the sphere over 1-dimensional potential functions via Exponential maps to find the inverse of a given exponential map on the sphere.
    In initial tests it was found that a very precise application requires at least 40-50 ierations, even though one is already pretty close after 10 iterations.
    Distributing the points randomly on the sphere doesn't really help, so every point is initialized just at (0,0,1).
    In order for this to work properly, the function *has* to be globally diffeomorphic, so the exponential map has to follow the conditions outlined in 
    https://arxiv.org/abs/0906.0874 (Sei 2009).

    Parameters:
    
        combined_func (function): A function that returns the (x,y,z) unit vector and its jacobian.
        find_tangent_func (function): A funtion to calculate the tangent at a given point along a certain direction.
        target_arg (float Tensor): The argument at which the inverse functon should be evaluated. Tensor of size (B,D) where B is the batchsize, and D the dimension.
        *args (list): Any extra arguments passed to *func*.
        num_newton_iter (int): Number of Newton iterations.

    Returns:

        Tensor
            The inverse of the exponential map.

    """
    
    prev=torch.zeros_like(target_arg)
    prev[:,2]=-1.0
  
    for i in range(num_newton_iter):
        
        phi_res, _, jac_phi,_=combined_func(prev, *args)


        fn_eval=-(phi_res*target_arg).sum(axis=-1, keepdims=True)+1.0

        res_vec=-torch.bmm(jac_phi.permute(0,2,1), target_arg.unsqueeze(2)).squeeze(-1)#*basic_pot_func.unsqueeze(1).unsqueeze(2)


        grad_norm=(res_vec**2).sum(axis=1, keepdims=True).sqrt()

        
        new_vs,_=find_tangent_func(prev, -(res_vec/grad_norm).squeeze(-1))

        
        gpnew=(new_vs*res_vec).sum(axis=1,keepdims=True)
        
        projection_2=fn_eval/gpnew

        projection_2=-projection_2

        prev=basic_exponential_map_func(prev, new_vs, projection_2)
       
    return prev
