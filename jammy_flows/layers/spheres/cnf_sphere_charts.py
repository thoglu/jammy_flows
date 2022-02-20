import torch
from torch import nn
import numpy

from torchdiffeq import odeint_adjoint as odeint

from . import sphere_base
from . import moebius_1d

from ...amortizable_mlp import AmortizableMLP
from ...extra_functions import  list_from_str
from .cnf_specific.cnf_sphere_manifold import Sphere
import sys
import os
import copy

import torch.autograd


"""
Code implementation of "Neural manifold ordinary differential equations" (https://arxiv.org/abs/2006.10254), 
mostly adapted from https://github.com/CUAI/Neural-Manifold-Ordinary-Differential-Equations. This implemenation
of manifold flows involves multiple charts that stick together and together form a global diffeomophism by repeated projection and flow operations.

NOTE: NOT YET WORKING PROPERLY
"""

sphere=Sphere()

def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


## wrapper required for torchdiffeq
class TimeNetwork(nn.Module):
    def __init__(self, func):
        super(TimeNetwork, self).__init__()
        self.func = func

    def forward(self, t, x, extra_inputs=None):
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        t_p = t.expand(x.shape[:-1] + (1,))
        return self.func(torch.cat((x, t_p), -1), extra_inputs=extra_inputs)


class ODEfunc(nn.Module):

    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()

        self.diffeq = diffeq
        
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(y)
        batchsize = y.shape[0]
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t, y)
            
            divergence = divergence_bf(dy, y).unsqueeze(-1)


        return tuple([dy, -divergence])

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

####################

class AmbientProjNN(nn.Module):
    def __init__(self, func):
        super(AmbientProjNN, self).__init__()
        self.func = func
        self.man = sphere

    def forward(self, t, x, extra_inputs=None):
        x = self.man.proju(x, self.func(t, x, extra_inputs=extra_inputs))
        return x

####################

class SphereProj(nn.Module):
    def __init__(self, func, loc, extra_inputs=None):
        super(SphereProj, self).__init__()
        self.base_func = func
        self.man = sphere
        self.loc = loc.detach()
        self.extra_inputs=extra_inputs

    def forward(self, t, x):
        """
        x is assumed to be an input on the tangent space of self.loc
        """
        y = self.man.exp(self.loc, x)
        val = self.man.jacoblog(self.loc, y) @ self.base_func(t, y, extra_inputs=self.extra_inputs).unsqueeze(-1)
        val = val.squeeze()

        #print("dydx", (val**2).sum(dim=1))
        return val

######################

class cnf_sphere_charts(sphere_base.sphere_base):

    def __init__(self, dimension, euclidean_to_sphere_as_first=False, use_extra_householder=False, use_permanent_parameters=False, natural_direction=0, cnf_network_hidden_dims="64", cnf_network_rank=0, cnf_network_highway_mode=0, num_charts=6, solver="rk4", higher_order_cylinder_parametrization=False):

        super().__init__(dimension=dimension, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_extra_householder=use_extra_householder, use_permanent_parameters=use_permanent_parameters, higher_order_cylinder_parametrization=higher_order_cylinder_parametrization)
        
        if(dimension!=2):
            raise Exception("The sphere CNF should be used for dimension 2!")
        
        self.cnf_network_hidden_dims=cnf_network_hidden_dims
        self.natural_direction=natural_direction
    
        ## 4 input parameters (x,y,z - the embedding coordinates of S-2, aswell as a time coordinate to indicate where in the ODE we are)
        ## 3 outputs, as the output is 3-d vector indicating the ODE vector field
        self.cnf_network=AmortizableMLP(4, cnf_network_hidden_dims, 3, use_permanent_parameters=use_permanent_parameters, low_rank_approximations=cnf_network_rank, highway_mode=cnf_network_highway_mode)
        self.num_nn_pars=self.cnf_network.num_amortization_params
        ## param num = potential_pars*num_potentials 
        self.total_param_num+=self.num_nn_pars
        
        ## ODE specific options - default from original implementation
        self.solver = solver
        self.atol = 1e-7
        self.rtol = 1e-7
        self.solver_options = {'step_size': 1/16}
        self.man = sphere
        self.num_charts=num_charts
            
        ## a function
        self.func = AmbientProjNN(TimeNetwork(self.cnf_network))
    """
    def forward(self, z, extra_inputs=None):
        return self._forward(z, extra_inputs=extra_inputs)

    def inverse(self, z, extra_inputs=None):
        return self._forward(z, reverse=True,extra_inputs=extra_inputs)
    """
    def _forward(self, z, reverse=False, extra_inputs=None):

        #print("extra input ")
        #print(extra_inputs)
        #print("Z %.20f" % z[0][0],z.shape)
        integration_times = torch.tensor(
            [[i/self.num_charts, (i + 1)/self.num_charts] for i in range(self.num_charts)]
        )
        if reverse:
            #flip each time steps [s_t, e_t]
            integration_times = _flip(integration_times, -1)
            #reorder time steps from 0 -> n to give n -> 0
            integration_times = _flip(integration_times, 0)

        #print("integration times ", integration_times)
        # initial values

        print(z)
        loc = z.detach()
        tangval = self.man.log(loc, z)
        print("tangval")
        print(tangval)
        logpz_t = 0
        
        scale = -1 if reverse else 1

        for time in integration_times:
            chartproj = SphereProj(self.func, loc, extra_inputs=extra_inputs)
            chartfunc = ODEfunc(chartproj)

            logpz_t -= scale * self.man.logdetexp(loc, tangval)

            # integrate as a tangent space operation
            state_t = odeint(
                    chartfunc,
                    (tangval, torch.zeros(tangval.shape[0], 1).to(tangval)),
                    time.to(z),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.solver,
                    options=self.solver_options
                )
            
            # extract information
            state_t = tuple(s[1] for s in state_t)

            
            y_t, logpy_t = state_t[:2]
            y_t = self.man.proju(loc, y_t)

            print(y_t)
            
           
            #print("final dy.dt for exp : ", (y_t**2).sum(axis=1).sqrt())

            # log p updates
            logpz_t -= logpy_t.squeeze()
            logpz_t += scale * self.man.logdetexp(loc, y_t)

            # set up next iteration values
            z_n = self.man.exp(loc, y_t)
            loc = z_n
            tangval = self.man.log(loc, z_n)

            #print("num evals .. ", chartfunc.num_evals())

        #print("Zn %.20f" % z_n[0][0])


        return z_n, logpz_t
    """    
    def get_regularization_states(self):
        return None

    def num_evals(self):
        return self.odefunc._num_evals.item()
    """
    def _inv_flow_mapping(self, inputs, extra_inputs=None):

        #if(self.higher_order_cylinder_parametrization):
        #    print("extra params MLP", extra_inputs[0,self.num_mlp_params-self.total_euclidean_pars:self.num_mlp_params])
        ## input structure: 0-num_amortization_params -> MLP  , num_amortizpation_params-end: -> moebius trafo
        [x,log_det]=inputs

        if(self.always_parametrize_in_embedding_space==False):
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)
     
        if(self.natural_direction):
            print("bef ", x)
            res, log_det_fac=self._forward(x, reverse=True, extra_inputs=extra_inputs)
            print("after", res)
            log_det=log_det+log_det_fac
        else:
            res, log_det_fac=self._forward(x, extra_inputs=extra_inputs)
            log_det=log_det+log_det_fac

        if(self.always_parametrize_in_embedding_space==False):
            res, log_det=self.eucl_to_spherical_embedding(res, log_det)

        return res, log_det, None

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        
        [x,log_det]=inputs

        if(self.always_parametrize_in_embedding_space==False):
            x, log_det=self.spherical_to_eucl_embedding(x, log_det)

        
        if(self.natural_direction):
            res, log_det_fac=self._forward(x, extra_inputs=extra_inputs)
            log_det=log_det+log_det_fac
        else:
            res, log_det_fac=self._forward(x, reverse=True, extra_inputs=extra_inputs)
            log_det=log_det+log_det_fac

        if(self.always_parametrize_in_embedding_space==False):
            res, log_det=self.eucl_to_spherical_embedding(res, log_det)

        return res, log_det

    def _init_params(self, params):

        assert(len(params)== (self.num_nn_pars))

        self.cnf_network.initialize_uvbs()
        print("multiplying init_params")
        self.cnf_network.u_v_b_pars.data*=1000.0

    def _get_desired_init_parameters(self):

        print("get desired")
        gaussian_init=torch.randn((self.num_nn_pars))/1.0

        return gaussian_init