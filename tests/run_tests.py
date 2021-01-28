import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.flows as f
#from pytorch_lightning import seed_everything
import jammy_flows.helper_fns as helper_fns

def check_gf_trafo(flow, crude_trafo, new_trafo, new_deriv, new_log_deriv,  plot=False, name="test.png"):

    points=torch.linspace(-2,2,1000).type(torch.float64).reshape(1000,1)
    #points=torch.linspace(-0.64669,-0.64664,5000).type(torch.float64).reshape(5000,1)
    y_crude=crude_trafo(points).detach().numpy()

    y_trafo_new=new_trafo(points).detach().numpy()
    y_deriv_new_torch=torch.autograd.functional.jacobian(new_trafo, points).sum(axis=2).squeeze().detach().numpy()
    y_deriv_new_defined=new_deriv(points).detach().numpy().squeeze()

    y_log_deriv_new_defined=new_log_deriv(points).detach().numpy().squeeze()

    log_deriv_diff=numpy.fabs((numpy.log(y_deriv_new_defined)-y_log_deriv_new_defined)).sum()
    torch_deriv_diff=numpy.fabs(y_deriv_new_torch-y_deriv_new_defined).sum()

    #y_deriv_of_deriv_torch=torch.autograd.functional.jacobian(new_deriv, points).sum(axis=2).squeeze().detach().numpy()
    y_deriv_of_deriv_torch=torch.zeros_like(points).squeeze()

    #y_deriv_new_defined=numpy.where(numpy.isfinite(y_deriv_new_defined), y_deriv_new_defined, 0)

    if(plot):
        fig=pylab.figure()

        ax0=fig.add_subplot(3,1,1)

        log_pdf,_,_=flow(points)

        pts_numpy=points.squeeze(0).detach()
        y_pts=log_pdf.exp().detach().numpy()

        ax0.plot(pts_numpy, y_pts)
        ax0.semilogy()

        ax0.set_title("log_diff %.3e / automatic/manual diff %.3e" % (log_deriv_diff, torch_deriv_diff))
        ax1=fig.add_subplot(3,1,2)

        ax1.plot(pts_numpy, y_crude, color="black")
        
        ax1.plot(pts_numpy, y_trafo_new, color="gray")
        ax1.plot(pts_numpy, y_deriv_new_torch, color="blue")
        ax1.plot(pts_numpy, y_deriv_new_defined, color="magenta")

        ax2=fig.add_subplot(3,1,3)

        ax2.plot(pts_numpy, y_deriv_new_torch-y_deriv_new_defined, color="green", label="diff automatic/defined deriv new")
        #ax2.plot(pts_numpy, y_crude-y_trafo_new, color="black", label="crude-this diff")

        ax2.legend()

        fig.tight_layout()

        pylab.savefig(name)

  
    return log_deriv_diff, torch_deriv_diff, y_deriv_of_deriv_torch.sum()

def compare_two_arrays(arr1, arr2, name1, name2):

    num_non_finite_1=(numpy.isfinite((arr1))==0).sum()

    if(num_non_finite_1>0):
        print(name1, " contains %d non-finite elements" % num_non_finite_1)
        print(arr1[numpy.isfinite((arr1))==0])

        raise Exception()

    num_non_finite_2=(numpy.isfinite((arr1))==0).sum()
    if(num_non_finite_2>0):
        print(name2, " contains %d non-finite elements" % num_non_finite_2)
        print(arr2[numpy.isfinite((arr2))==0])

        raise Exception()


    diff_value=1e-7
    diff_too_large_mask=numpy.fabs(arr1-arr2)>diff_value

    if(diff_too_large_mask.sum()>0):
        print("%s and %s contain elements (%d/%d) that differ by value greater than %.5e" % (name1, name2,diff_too_large_mask.sum(),len(diff_too_large_mask),diff_value))
        print("selected array 1 values ...")
        print(arr1[diff_too_large_mask])
        print("selected array 2 values ...")
        print(arr2[diff_too_large_mask])
        print(".. diffs")
        print(numpy.fabs(arr1-arr2)[diff_too_large_mask])
        raise Exception()

    print("largest diff between ", name1, " and ", name2, " (%d items): " % len(arr1),numpy.fabs(arr1-arr2).max() )

class Test(unittest.TestCase):
    def setUp(self):

        self.flow_inits=[]
        
        pdf_def="e1+e1+e1+s1"
        flow_def="pp+pp+gg+m"

  
        #self.flow_inits.append( [ [pdf_def, flow_def], {"flow_defs_detail":{"g":{"kwargs":{"inverse_function_type":"inormal_partly_precise"}}}}] )
        #self.flow_inits.append( [ [pdf_def, flow_def], {"flow_defs_detail":{"g":{"kwargs":{"inverse_function_type":"inormal_partly_crude"}}}}] )
        #self.flow_inits.append( [ [pdf_def, flow_def], {"flow_defs_detail":{"g":{"kwargs":{"inverse_function_type":"inormal_full_pade"}}}}] )
        #self.flow_inits.append( [ [pdf_def, flow_def], {"flow_defs_detail":{"g":{"kwargs":{"inverse_function_type":"isigmoid"}}}}] )

        
        encoders=["passthrough"]

        mlp_hidden_dims=["64-30"]
        use_low_rank_option=[False, True]
        low_rank_ranks=["2-10-1000"]


        gf_settings=[{"g":{"kwargs":{"inverse_function_type":"inormal_partly_crude"}}},{"g":{"kwargs":{"inverse_function_type":"inormal_partly_precise"}}},{"g":{"kwargs":{"inverse_function_type":"inormal_full_pade"}}}]

        for enc in encoders:
            for mlp_hidden in mlp_hidden_dims:
                for use_rank in use_low_rank_option:
                    for lr_approx in low_rank_ranks:
                        for gf_setting in gf_settings:
                            d=dict()
                            
                            d["conditional_input_dim"]=2
                            d["data_summary_dim"]=2
                            d["input_encoder"]=enc
                            d["hidden_mlp_dims_sub_pdfs"]=mlp_hidden
                            d["rank_of_mlp_mappings_sub_pdfs"]=lr_approx
                            d["use_custom_low_rank_mlps"]=use_rank
                            d["flow_defs_detail"]=gf_setting

                            self.flow_inits.append([[pdf_def, flow_def], d])
        

        

        extra_flow_defs=dict()
        extra_flow_defs["flow_defs_detail"]=dict()
        extra_flow_defs["flow_defs_detail"]["n"]=dict()
        extra_flow_defs["flow_defs_detail"]["n"]["kwargs"]=dict()
        extra_flow_defs["flow_defs_detail"]["n"]["kwargs"]["use_extra_householder"]=1
        extra_flow_defs["flow_defs_detail"]["n"]["kwargs"]["higher_order_cylinder_parametrization"]=True

        pdf_def="s2+e2+s1"
        flow_def="n+gg+m"

        self.flow_inits.append([[pdf_def, flow_def], extra_flow_defs])
    
    
    def test_selfconsistency(self):

        print("-> Testing self consistency of sampling <-")

        samplesize=10000
        
        for ind, init in enumerate(self.flow_inits):
            
            print("checking ...", init)
            #seed_everything(0)
            this_flow=f.pdf(*init[0], **init[1])
            this_flow.double()


            cinput=None
            if("conditional_input_dim" in init[1].keys()):
                rvec=numpy.random.normal(size=(samplesize,2))*100.0
                cinput=torch.from_numpy(rvec)

            samples, base_samples, evals, base_evals=this_flow.sample(samplesize=samplesize,conditional_input=cinput)
            
         
            ## evaluate the samples and see if the reverse direction is compatible
            evals2, base_evals2, base_samples2=this_flow(samples, conditional_input=cinput)

            #this_flow.count_parameters()

            compare_two_arrays(evals.detach().numpy(), evals2.detach().numpy(), "evals", "evals2")
            compare_two_arrays(base_samples.detach().numpy(), base_samples2.detach().numpy(), "base_samples", "base_samples2")
            compare_two_arrays(base_evals.detach().numpy(), base_evals2.detach().numpy(), "base_evals", "base_evals2")
    
    
    
    def test_sphere_singular_points(self):


        print("Testing singular points")
        extra_flow_defs=dict()
        extra_flow_defs["m"]=dict()
        extra_flow_defs["m"]["kwargs"]=dict()
        extra_flow_defs["m"]["kwargs"]["use_extra_householder"]=0


        this_flow=f.pdf("s1", "m", flow_defs_detail=extra_flow_defs)
        input=torch.from_numpy(numpy.array([0.0,numpy.pi*2.0])[:,None])

        ev,_,_=this_flow(input)

        self.assertTrue( (numpy.isfinite((ev).detach().numpy())==0).sum()==0)



        extra_flow_defs=dict()
        extra_flow_defs["n"]=dict()
        extra_flow_defs["n"]["kwargs"]=dict()
        extra_flow_defs["n"]["kwargs"]["use_extra_householder"]=0
        extra_flow_defs["n"]["kwargs"]["higher_order_cylinder_parametrization"]=1

        this_flow=f.pdf("s2", "n", flow_defs_detail=extra_flow_defs)
        input=torch.from_numpy(numpy.array([[0.00,2.0],[numpy.pi,2.0]]))

        ev,_,_=this_flow(input)

        self.assertTrue( (numpy.isfinite((ev).detach().numpy())==0).sum()==0)
    

    def test_gpu(self):

        print("-> Testing gpu-support <-")
        dev=torch.device("cuda")
        try:
            a=torch.Tensor([1.0,2.0]).to(dev)
        except:
            print("GPU not supported - skipping GPU test")
        else:
            print("GPU supported ..")

            check_list=[ ["e1", "g"], ["e1", "p"], ["s1", "m"], ["s2","n"] ]

            for args in check_list:
                pdf=f.pdf(*args)

                gpu_sample,_,_,_ = pdf.sample(device=dev)

                res,_,_=pdf(gpu_sample)

                print(res)


    def test_2d_sphere_evals(self):


        print("-> Testing singular points on S2 <-")
        extra_flow_defs=dict()
        extra_flow_defs["n"]=dict()
        extra_flow_defs["n"]["kwargs"]=dict()
        extra_flow_defs["n"]["kwargs"]["use_extra_householder"]=1
        extra_flow_defs["n"]["kwargs"]["higher_order_cylinder_parametrization"]=True
        #seed_everything(1)
        this_flow=f.pdf("s2", "n", flow_defs_detail=extra_flow_defs)

        theta=torch.linspace(0.0001, numpy.pi-0.0001,700)
        phi=torch.linspace(0.0001, 2*numpy.pi-0.0001,700)
        area=(theta[1]-theta[0])*(phi[1]-phi[0])

        mesh_theta, mesh_phi=torch.meshgrid(theta,phi)

        mesh_theta=mesh_theta.flatten().unsqueeze(1)
        mesh_phi=mesh_phi.flatten().unsqueeze(1)

        combined_coords=torch.cat([mesh_theta,mesh_phi],dim=1).type(torch.float64)
      
        fig=pylab.figure()

        helper_fns.visualize_pdf(this_flow, fig, nsamples=100000, s2_norm="standard", total_pdf_eval_pts=10000)  

        pylab.savefig("sphere_standard.png")  
        
        fig=pylab.figure()

        helper_fns.visualize_pdf(this_flow, fig, nsamples=100000, s2_norm="lambert", total_pdf_eval_pts=10000)  

        pylab.savefig("sphere_lambert.png")  
    
    def test_derivative(self):
        print("Testing derivatives of Gaussianization flows")
        icdf_approximations=["inormal_full_pade", "inormal_partly_crude", "inormal_partly_precise", "isigmoid"]
        #seed_everything(0)

        extra_flow_defs=dict()
        extra_flow_defs["g"]=dict()
        extra_flow_defs["g"]["kwargs"]=dict()

        extra_flow_defs["g"]["kwargs"]["inverse_function_type"]="inormal_partly_crude"

        crude_flow=f.pdf("e1", "g", flow_defs_detail=extra_flow_defs)

        def fn_crude(x):
            b=crude_flow.layer_list[0][0]
            return b.sigmoid_inv_error_pass(x, b.datapoints, b.log_hs, b.log_kde_weights)

        for icdf_approx in icdf_approximations:
            #seed_everything(0)

            extra_flow_defs=dict()
            extra_flow_defs["g"]=dict()
            extra_flow_defs["g"]["kwargs"]=dict()

            extra_flow_defs["g"]["kwargs"]["inverse_function_type"]=icdf_approx

            this_flow=f.pdf("e1", "g", flow_defs_detail=extra_flow_defs)

            for gf_layer in this_flow.layer_list[0]:

                def fn_new(x):
                    return gf_layer.sigmoid_inv_error_pass(x, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights)

                def fn_new_deriv(x):
                    return gf_layer.sigmoid_inv_error_pass_derivative(x, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights)

                def fn_new_deriv_log(x):
                    return gf_layer.sigmoid_inv_error_pass_log_derivative(x, gf_layer.datapoints, gf_layer.log_hs, gf_layer.log_kde_weights)


                log_deriv_check, residual_sum_of_derivative_difference, deriv_of_deriv_sums=check_gf_trafo(this_flow, fn_crude, fn_new, fn_new_deriv,fn_new_deriv_log, plot=True, name="test_gf_pade_mode_%s.png"%(icdf_approx))
                
                if(residual_sum_of_derivative_difference>1e-4 or numpy.isfinite(residual_sum_of_derivative_difference)==False):
                    raise Exception("Sum of automatic-manual derivatives too large", icdf_approx, residual_sum_of_derivative_difference)
                if(log_deriv_check>1e-6 or numpy.isfinite(log_deriv_check)==False):
                    raise Exception("Log Derivative and log(derivative) do not match ", icdf_approx, log_deriv_check)
                if(deriv_of_deriv_sums>1e-6 or numpy.isfinite(deriv_of_deriv_sums)==False):
                    raise Exception("Derivative of derivative is not finite", icdf_approx, deriv_of_deriv_sums)
                
    
    
    

if __name__ == '__main__':
    unittest.main()