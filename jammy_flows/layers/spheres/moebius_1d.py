import torch
from torch import nn
import numpy

from . import sphere_base
from ..bisection_n_newton import inverse_bisection_n_newton

class moebius(sphere_base.sphere_base):
    def __init__(self, dimension=1, euclidean_to_sphere_as_first=True, use_extra_householder=True, natural_direction=0, use_permanent_parameters=False, use_moebius_xyz_parametrization=True, num_moebius=5):

        super().__init__(dimension=1, euclidean_to_sphere_as_first=euclidean_to_sphere_as_first, use_extra_householder=use_extra_householder, use_permanent_parameters=use_permanent_parameters)
        
        if(dimension!=1):
            raise Exception("The moebius flow is defined for dimension 1, but dimension %d is handed over" % (dimension))
           
    
        self.use_moebius_xyz_parametrization=use_moebius_xyz_parametrization
        self.num_moebius=num_moebius
        self.num_omega_pars=4

        if(self.use_moebius_xyz_parametrization==0):
            self.num_omega_pars=3

        ## add parameters from this layer (on top of householder params defined in the class parent)
        self.total_param_num+=self.num_moebius*self.num_omega_pars

        if(use_permanent_parameters):
            self.moebius_pars=nn.Parameter(torch.randn(self.num_moebius,self.num_omega_pars).type(torch.double).unsqueeze(0))
            
        else:
            self.moebius_pars=torch.zeros(self.num_moebius,self.num_omega_pars).type(torch.double).unsqueeze(0)

        ## natural direction means no bisection in the forward pass, but in the backward pass
        self.natural_direction=natural_direction

    def _inv_flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):

        [x,logdet]=inputs

        moebius_pars=self.moebius_pars.to(x)
        #this_num_params=self.use_moebius*num_per_omega
        if(extra_inputs is not None):
            
            moebius_pars=moebius_pars+torch.reshape(extra_inputs, [x.shape[0], self.num_moebius, self.num_omega_pars])
        
       
        ## switch from 0-2pi to -pi/pi
        xmask=x>numpy.pi
        x=(xmask)*(x-2*numpy.pi)+(~xmask)*x

        if(self.natural_direction):
            ## do inverse in -pi/pi
            x=inverse_bisection_n_newton(self.simple_moebius_trafo, self.simple_moebius_trafo_deriv, x, moebius_pars, min_boundary=-numpy.pi, max_boundary=numpy.pi, num_bisection_iter=20, num_newton_iter=20)
            
            log_deriv=-torch.log(self.simple_moebius_trafo_deriv(x,moebius_pars)).sum(axis=-1)
            
        else:
            log_deriv=torch.log(self.simple_moebius_trafo_deriv(x,moebius_pars)).sum(axis=-1)
            x=self.simple_moebius_trafo(x, moebius_pars) 
        ## switch back to 0-2pi
        smaller_mask=x<0
        x=(smaller_mask)*(2*numpy.pi+x)+(~smaller_mask)*x

        
        ### moebius is actually the inverse mapping
        
        
        logdet+=log_deriv

    
       
        return x, logdet

    def _flow_mapping(self, inputs, extra_inputs=None, sf_extra=None):
        
       
        [x,logdet]=inputs

        moebius_pars=self.moebius_pars.to(x)
        #this_num_params=self.num_moebius*self.num_omega_pars
        if(extra_inputs is not None):
         
            moebius_pars=moebius_pars+torch.reshape(extra_inputs, [x.shape[0], self.num_moebius, self.num_omega_pars])
        
        
        ## switch to -pi/pi
        xmask=x>numpy.pi
        x=(xmask)*(x-2*numpy.pi)+(~xmask)*x

        if(self.natural_direction):
            log_deriv=torch.log(self.simple_moebius_trafo_deriv(x,moebius_pars)).sum(axis=-1)
            x=self.simple_moebius_trafo(x, moebius_pars)  
        else:
            x=inverse_bisection_n_newton(self.simple_moebius_trafo, self.simple_moebius_trafo_deriv, x, moebius_pars, min_boundary=-numpy.pi, max_boundary=numpy.pi, num_bisection_iter=20, num_newton_iter=20)
            
            log_deriv=-torch.log(self.simple_moebius_trafo_deriv(x,moebius_pars)).sum(axis=-1)
        ## switch back to 0/2pi
        smaller_mask=x<0
        x=(smaller_mask)*(2*numpy.pi+x)+(~smaller_mask)*x

        logdet+=log_deriv

        return x, logdet

    def simple_moebius_trafo(self, x, omega_pars):

        ## 
        
        cos_x=torch.cos(x)[:,None,:]
        sin_x=torch.sin(x)[:,None,:]
        
        #sin_x=torch.cos(x)[:,None,:]
        #cos_x=torch.sin(x)[:,None,:]

        cos_minus_pi=numpy.cos(-numpy.pi)
        sin_minus_pi=numpy.sin(-numpy.pi)

        MIN_OMEGA_RADIUS=0.001
        MAX_OMEGA_RADIUS=0.999
        ## omega_pars = x,y,radius (overaparametrized), normalization

        log_length_par=omega_pars[:,:,-2:-1]

        #print("shape log_length", log_length_par.shape)

        lse_cat=torch.cat( (torch.zeros_like(log_length_par), -log_length_par),dim=2)

        #print(lse_cat.shape)

        denom=torch.logsumexp(lse_cat, dim=2, keepdims=True)
        
        ## sigmoid between min and max omega radius
        omega_length=MIN_OMEGA_RADIUS+torch.exp(numpy.log(MAX_OMEGA_RADIUS-MIN_OMEGA_RADIUS)-denom)


        """
        omega_vec_normed=omega_pars[:,:,:2]/((omega_pars[:,:,:2]**2).sum(axis=2,keepdims=True)).sqrt()
        omega_vec=omega_vec_normed*omega_length

        """

        if(self.use_moebius_xyz_parametrization):
            omega_vec_normed=omega_pars[:,:,:2]/((omega_pars[:,:,:2]**2).sum(axis=2,keepdims=True)).sqrt()
            omega_vec=omega_vec_normed*omega_length
        else:
            omega_vec=torch.cat( (torch.cos(omega_pars[:,:,0:1])*omega_length, torch.sin(omega_pars[:,:,0:1])*omega_length)  , dim=2)

          
        #single_omegas=torch.sqrt(squared_omegas)

        o_m_o_sq=1.0-omega_length**2
        o_p_o_twice=1.0+omega_length**2-2*(cos_x*omega_vec[:,:,0:1]+sin_x*omega_vec[:,:,1:2])

        o_p_o_twice_m_pi=1.0+omega_length**2-2*(cos_minus_pi*omega_vec[:,:,0:1]+sin_minus_pi*omega_vec[:,:,1:2])

        y_val_m_pi=o_m_o_sq*(sin_minus_pi-omega_vec[:,:,1:2])-omega_vec[:,:,1:2]*o_p_o_twice_m_pi
        x_val_m_pi=o_m_o_sq*(cos_minus_pi-omega_vec[:,:,0:1])-omega_vec[:,:,0:1]*o_p_o_twice_m_pi

        phi_m_pi=torch.atan2(y_val_m_pi,x_val_m_pi)
        rotation_angle=-numpy.pi-phi_m_pi

        y_val=o_m_o_sq*(sin_x-omega_vec[:,:,1:2])-omega_vec[:,:,1:2]*o_p_o_twice
        x_val=o_m_o_sq*(cos_x-omega_vec[:,:,0:1])-omega_vec[:,:,0:1]*o_p_o_twice

        ### now rotate x xval by rotation_angle

        x_prime=torch.cos(rotation_angle)*x_val-torch.sin(rotation_angle)*y_val
        y_prime=torch.sin(rotation_angle)*x_val+torch.cos(rotation_angle)*y_val

        arc_tans=torch.atan2(y_prime,x_prime)[:,:,-1:]+numpy.pi

       
        log_norms=omega_pars[:,:,-1:]
        weighted_arctan=arc_tans*torch.exp(log_norms-torch.logsumexp(log_norms, dim=1,keepdim=True))
        #weighted_arctan=weighted_arctan.squeeze(-1)
      
        ## between -pi to pi
        res=torch.sum(weighted_arctan, dim=1)-numpy.pi
        
        #smaller_mask=(res<0).double()
        #return_val=smaller_mask*(2*numpy.pi+res)+(1-smaller_mask)*res
        
        return res ## transform back to -pi/pi


    def simple_moebius_trafo_deriv(self, x, omega_pars):
        
        cos_x=torch.cos(x)[:,None,:]
        sin_x=torch.sin(x)[:,None,:]
        
        #sin_x=torch.cos(x)[:,None,:]
        #cos_x=torch.sin(x)[:,None,:]

        MIN_OMEGA_RADIUS=0.001
        MAX_OMEGA_RADIUS=0.999
        ## omega_pars = x,y,radius (overaparametrized), normalization

        log_length_par=omega_pars[:,:,-2:-1]

        #print("shape log_length", log_length_par.shape)

        lse_cat=torch.cat( (torch.zeros_like(log_length_par), -log_length_par),dim=2)

        #print(lse_cat.shape)

        denom=torch.logsumexp(lse_cat, dim=2, keepdims=True)
        
        ## sigmoid between min and max omega radius
        omega_length=MIN_OMEGA_RADIUS+torch.exp(numpy.log(MAX_OMEGA_RADIUS-MIN_OMEGA_RADIUS)-denom)

        if(self.use_moebius_xyz_parametrization):
            
            omega_vec_normed=omega_pars[:,:,:2]/((omega_pars[:,:,:2]**2).sum(axis=2,keepdims=True)).sqrt()
            omega_vec=omega_vec_normed*omega_length
        else:
            omega_vec=torch.cat( (torch.cos(omega_pars[:,:,0:1])*omega_length, torch.sin(omega_pars[:,:,0:1])*omega_length)  , dim=2)

        o_m_o_sq=1.0-omega_length**2
        o_p_o_twice=1.0+omega_length**2-2*(cos_x*omega_vec[:,:,0:1]+sin_x*omega_vec[:,:,1:2])

        log_norms=omega_pars[:,:,-1:]
        weighted_deriv=(torch.log(o_m_o_sq/o_p_o_twice)+log_norms)-torch.logsumexp(log_norms, dim=1,keepdim=True)

        res=torch.logsumexp(weighted_deriv, dim=1)

        return torch.exp(res)

    def _init_params(self, params):

        self.moebius_pars.data=params.reshape(1, self.num_moebius, self.num_omega_pars)

    def _get_desired_init_parameters(self):

        ## gaussian init data
        
        return torch.randn((self.num_moebius*self.num_omega_pars))

