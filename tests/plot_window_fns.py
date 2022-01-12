import pylab

import numpy
import scipy
import scipy.special
import torch

def generated_functions_bounded_in_logspace(min_val=1, max_val=10, center_around_zero=False):
    
    ## min and max values are in normal space -> must be positive
    assert(min_val > 0)

    ln_max=numpy.log(max_val)
    ln_min=numpy.log(min_val)

    ## this shift makes the function equivalent to a normal exponential for small values
    center_val=ln_max

    ## can also center around zero (it will be centered in exp space, not in log space)
    if(center_around_zero):
        center_val=0.0

    def f(x):

        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-x+center_val).unsqueeze(-1)], dim=-1)

        first_term=ln_max-torch.logsumexp(res, dim=-1, keepdim=True)

        return torch.logsumexp( torch.cat([first_term, torch.ones_like(first_term)*ln_min], dim=-1), dim=-1)

    def exp_f(x):

        return numpy.exp(f(x))


    return f, exp_f

def generated_function_inverted_softmax_first(min_val=1, max_val=10):
    
    ln_max=numpy.log(max_val)
    ln_min=numpy.log(min_val)

    def f(x):

        res=torch.cat([torch.zeros_like(x).unsqueeze(-1), (-x+ln_max).unsqueeze(-1)], dim=-1)

        first_term=ln_max-torch.logsumexp(res, dim=-1, keepdim=False)

        return first_term

    def exp_f(x):

        return numpy.exp(f(x))


    return f, exp_f

def plot_funcs(f, exp_f, filename):


    fig=pylab.figure()
    

    xvals=numpy.linspace(-20,20)
    xvals_torch=torch.from_numpy(xvals)

    yvals1_numpy=f(xvals_torch).numpy()
    yvals2_numpy=f2(xvals_torch).numpy()

    ax1=fig.add_subplot(2,1,1)
    ax1.plot(xvals, yvals1_numpy)

    ax2=fig.add_subplot(2,1,2)
    ax2.plot(xvals, yvals2_numpy)

    fig.tight_layout()
   
    pylab.savefig(filename)


f1,f2=generated_functions_bounded_in_logspace(min_val=1e-5, max_val=20,center_around_zero=False)

plot_funcs(f1,f2,"default_small_minval.png")


f1,f2=generated_functions_bounded_in_logspace(min_val=1, max_val=20,center_around_zero=False)

plot_funcs(f1,f2,"default_small.png")

f1,f2=generated_functions_bounded_in_logspace(min_val=1, max_val=20,center_around_zero=True)

plot_funcs(f1,f2,"default_centered.png")

f1,f2=generated_function_inverted_softmax_first(min_val=2, max_val=10)

plot_funcs(f1,f2,"softmax_first.png")


