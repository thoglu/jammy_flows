import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f
import jammy_flows.helper_fns as helper_fns

def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

class Test(unittest.TestCase):
   

    
    def test_sphere_singular_points(self):


        print("Testing singular points")

        extra_flow_defs=dict()
        extra_flow_defs["m"]=dict()
        extra_flow_defs["m"]["use_extra_householder"]=0


        this_flow=f.pdf("s1", "m", options_overwrite=extra_flow_defs)
        input=torch.from_numpy(numpy.array([0.0,numpy.pi*2.0])[:,None])

        ev,_,_=this_flow(input)

        self.assertTrue( (numpy.isfinite((ev).detach().numpy())==0).sum()==0)

        extra_flow_defs=dict()
        extra_flow_defs["n"]=dict()
       
        extra_flow_defs["n"]["use_extra_householder"]=0
        extra_flow_defs["n"]["higher_order_cylinder_parametrization"]=0

        this_flow=f.pdf("s2", "n", options_overwrite=extra_flow_defs)
        test_input=torch.from_numpy(numpy.array([[0.0,0.0],[numpy.pi,0.0]]))
        
        ev,_,_=this_flow(test_input)
        
        self.assertTrue( (numpy.isfinite((ev).detach().numpy())==0).sum()==0)

        ###
     
        this_flow=f.pdf("s2", "vvv")
        ev,_,_=this_flow(test_input)

        self.assertTrue( (numpy.isfinite((ev).detach().numpy())==0).sum()==0)

        this_flow=f.pdf("s2", "c")
        ev,_,_=this_flow(test_input)

        self.assertTrue( (numpy.isfinite((ev).detach().numpy())==0).sum()==0)
    
      

    def test_2d_sphere_evals(self):


        print("-> Testing normalization of spheres <-")

        seed_everything(1)

        def check_flow(fl):
            """ 
            Check that PDF volume approximates 1.0 for more and more bins
            """
            eval_nums=[10,50,100, 200]

            

            print("checking flow ...", fl.pdf_defs_list, fl.flow_defs_list)

            for epsilon in [0.0,1e-3]:
                print("Theta epsilon: ", epsilon)
                pdf_sums=[]

                for num_per_dim in eval_nums:

                    theta=torch.linspace(0+epsilon, numpy.pi-epsilon,num_per_dim)[1:-1]
                    phi=torch.linspace(0, 2*numpy.pi, len(theta)+1)[:-1]

                    phi_bin_width=phi[1]-phi[0]
                    theta_bin_width=theta[1]-theta[0]

                    mesh_theta, mesh_phi=torch.meshgrid(theta,phi)

                    mesh_theta=mesh_theta.flatten().unsqueeze(1)
                    mesh_phi=mesh_phi.flatten().unsqueeze(1)

                    area=theta_bin_width*phi_bin_width

                    combined_coords=torch.cat([mesh_theta,mesh_phi],dim=1).type(torch.float64)
                    
                    log_evals, _,_=this_flow(combined_coords)

                    ## need that extra sin(theta) factor for correct summation in theta/phi coordinates (not required in equal-area projections of the sphere)
                    pdf_sum=((log_evals[:,None].exp()*area)).sum().detach().numpy()

                    pdf_sums.append(pdf_sum)

                
                print("PDF SUMS ", pdf_sums)

                ## smaller than 1 % off from 1.00 for the last one with the highest number of bins
                assert( numpy.fabs(pdf_sums[-1]-1.0) < 1e-2)


        ## Autoregressive flow With various options

        for zenith_layer_type in ["g", "p", "x", "z", "r"]:
            for cyl_para in [0]:

                extra_flow_defs=dict()
                extra_flow_defs["n"]=dict()
                
                extra_flow_defs["n"]["use_extra_householder"]=1
                extra_flow_defs["n"]["higher_order_cylinder_parametrization"]=cyl_para
                extra_flow_defs["n"]["zenith_type_layers"]=zenith_layer_type
             
                this_flow=f.pdf("s2", "n", options_overwrite=extra_flow_defs)

                check_flow(this_flow)

      
        ## Exponential map flow

        extra_flow_defs=dict()
        extra_flow_defs["v"]=dict()
        
        extra_flow_defs["v"]["natural_direction"]=0

        this_flow=f.pdf("s2", "vv", options_overwrite=extra_flow_defs)
        check_flow(this_flow)

        ##

        extra_flow_defs=dict()
        extra_flow_defs["v"]=dict()
    
        extra_flow_defs["v"]["natural_direction"]=1

        this_flow=f.pdf("s2", "vv", options_overwrite=extra_flow_defs)
        with torch.no_grad():
            check_flow(this_flow)

        ##

        extra_flow_defs=dict()
        extra_flow_defs["c"]=dict()
       
        extra_flow_defs["c"]["cnf_network_hidden_dims"]=""
        extra_flow_defs["c"]["num_charts"]=4

        this_flow=f.pdf("s2", "c", options_overwrite=extra_flow_defs)
        with torch.no_grad():
            check_flow(this_flow)




    

if __name__ == '__main__':
    unittest.main()