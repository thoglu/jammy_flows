import torch
import numpy

def close(a, b, rtol=1e-5, atol=1e-4):
    equal = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return equal


## differentiable newton iterations
def inverse_bisection_n_newton(func, grad_func, target_arg, *args, min_boundary=-100000.0, max_boundary=100000.0, num_bisection_iter=25, num_newton_iter=20):
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

        new_lower = (1. - correct_part) * (right_part * mid + left_part * new_lower) + correct_part * mid
        new_upper = (1. - correct_part) * (right_part * new_upper + left_part * mid) + correct_part * mid

        
    prev=mid


    for i in range(num_newton_iter):
       
        inf_mask_pos=(prev>float("1e200")).double()
        inf_mask_neg=(prev<float("-1e200")).double()
        inf_mask_good=((inf_mask_pos==0) & (inf_mask_neg == 0)).double()
        
        prev=inf_mask_good*prev+inf_mask_pos*1e200+inf_mask_neg*(-1e200)
        
        

        
        f_eval = func(prev, *args)-target_arg

        
      
        #print(f_eval[(torch.abs(f_eval)>1e-7)])
        f_prime_eval=grad_func(prev, *args)
        non_finite_sum=(torch.isfinite(prev)==False).sum()

        """
        print("non finite -- %d " % non_finite_sum)
        
        print("----------------")
        print(i)
        if(prev.shape[0]>48000):
            for kk in [33457]:
                print("FUNC EVAL", func(prev, *args)[kk])
                print("target arg", target_arg[kk])
                print("index ", kk)
                print(prev[kk])
                print(f_eval[kk])
                print(f_prime_eval[kk])
        print("-------------")
        """
        #print("----")
    
        prev=prev-(f_eval/f_prime_eval)

        #print("f_eval")
        #print(f_eval[torch.abs(f_eval)>1e-7])
        # print( (torch.abs(f_eval)>1e-7).nonzero())
        """
        for kk in [1876,9554]:
            print("FUNC EVAL", func(prev, *args)[kk])
            print("target arg", target_arg[kk])
            print("index ", kk)
            print(prev[kk])
            print(f_eval[kk])
            print(f_prime_eval[kk])
        """
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

