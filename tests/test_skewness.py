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

import jammy_flows.layers.extra_functions as extra_functions

def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)

def check_conditional_pdf(num_kde, plot=False, index=0, dimensionality=3):

    extra_flow_defs=dict()
    extra_flow_defs["flow_defs_detail"]=dict()
    extra_flow_defs["flow_defs_detail"]["g"]=dict()
    extra_flow_defs["flow_defs_detail"]["g"]["kwargs"]=dict()
    extra_flow_defs["flow_defs_detail"]["g"]["kwargs"]["add_skewness"]=1
    extra_flow_defs["flow_defs_detail"]["g"]["kwargs"]["num_kde"]=num_kde
    extra_flow_defs["conditional_input_dim"]=2

    pdf=f.pdf("e1", "g", **extra_flow_defs)

    x1=torch.linspace(-3,3, dimensionality, dtype=torch.double)
    x2=torch.linspace(-3,3, dimensionality, dtype=torch.double)
    x=torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)

    print("X ", x)
    #bw=(x[1]-x[0]).numpy()

    exponents=torch.rand( (dimensionality, num_kde, 2) , dtype=torch.double)*2-2

    print(" NUM KDE", num_kde, "INDEX ", index)
    log_widths=torch.rand( (dimensionality, num_kde, 2) , dtype=torch.double)*2+0.1

    log_norms=torch.rand( (dimensionality, num_kde, 2) , dtype=torch.double)*2+0.1
    means=torch.rand( (dimensionality, num_kde, 2) , dtype=torch.double)*5-5

    skew_signs=pdf.layer_list[0][0].kde_skew_signs

    total_cdf, total_sf, total_pdf=pdf.layer_list[0][0].logistic_kernel_log_pdf_quantities(x, means, log_widths, log_norms, exponents, skew_signs)
  

def check_skewness(num_kde, plot=False, index=0):

    extra_flow_defs=dict()
    extra_flow_defs["flow_defs_detail"]=dict()
    extra_flow_defs["flow_defs_detail"]["g"]=dict()
    extra_flow_defs["flow_defs_detail"]["g"]["kwargs"]=dict()
    extra_flow_defs["flow_defs_detail"]["g"]["kwargs"]["add_skewness"]=1
    extra_flow_defs["flow_defs_detail"]["g"]["kwargs"]["num_kde"]=num_kde

    pdf=f.pdf("e1", "g", **extra_flow_defs)

    x=torch.linspace(-500,500, 100000, dtype=torch.double)

    bw=(x[1]-x[0]).numpy()

    log_exponents=torch.rand( (1, num_kde, 1) , dtype=torch.double)*2-1

    print(" NUM KDE", num_kde, "INDEX ", index)
    log_widths=torch.rand( (1, num_kde, 1) , dtype=torch.double)*2+0.1

    log_norms=torch.rand( (1, num_kde, 1) , dtype=torch.double)*2+0.1
    means=torch.rand( (1, num_kde, 1) , dtype=torch.double)*5-5

    skew_signs=pdf.layer_list[0][0].kde_skew_signs


    total_cdf, total_sf, _=pdf.layer_list[0][0].logistic_kernel_log_pdf_quantities(x.reshape(-1,1), means, log_widths, log_norms, log_exponents, skew_signs)
    _, _, total_pdf=pdf.layer_list[0][0].logistic_kernel_log_pdf_quantities((x-0.5*bw).reshape(-1,1), means, log_widths, log_norms, log_exponents, skew_signs)
    total_sf=total_sf.squeeze(-1).exp().numpy()
    total_cdf=total_cdf.squeeze(-1).exp().numpy()
    total_pdf=total_pdf.squeeze(-1).exp().numpy()

    _, _, total_pdf_right=pdf.layer_list[0][0].logistic_kernel_log_pdf_quantities((x+0.5*bw).reshape(-1,1), means, log_widths, log_norms, log_exponents, skew_signs)
    total_pdf_right=total_pdf_right.squeeze(-1).exp().numpy()

    numerical_cdf=numpy.cumsum(total_pdf)*bw
    numerical_sf=(numpy.cumsum(total_pdf_right[::-1])*bw)[::-1]

    assert( numpy.fabs(total_cdf-numerical_cdf).max() <1e-5) , ("MAX DIFF CDF num/exact: ", numpy.fabs(total_cdf-numerical_cdf).max())
    assert( numpy.fabs(total_sf-numerical_sf).max() <1e-5), ("MAX DIFF SF num/exact: ", numpy.fabs(total_sf-numerical_sf).max()) 

    #print("MAX DIFF CDF", numpy.fabs(total_cdf-numerical_cdf).max())
    #print("MAX DIFF SF", numpy.fabs(total_sf-numerical_sf).max())
    if(plot):   
        fig=pylab.figure()

        ## cdf in upper plot
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(x, total_cdf, color="black", label="cdf")
        ax1.plot(x, numerical_cdf, color="green", ls="--", label="numerical cdf")

        ax1.plot(x, total_sf, color="black", label="sf")
        ax1.plot(x, numerical_sf, color="red", ls="--", label="numerical sf")

        ax1.legend(loc="upper left")

        ax2=fig.add_subplot(2,1,2)

        numerical_total_pdf=0.0
        ## pdfs in lower plot

        total_norm_sum=log_norms.exp().sum().numpy()
        for cur_kde in range(num_kde):

            _,_,cur_pdf=pdf.layer_list[0][0].logistic_kernel_log_pdf_quantities((x-0.5*bw).reshape(-1,1), means[:,cur_kde:cur_kde+1,:], log_widths[:,cur_kde:cur_kde+1,:], log_norms[:,cur_kde:cur_kde+1,:], log_exponents[:,cur_kde:cur_kde+1,:],skew_signs[:,cur_kde:cur_kde+1,:])
            cur_pdf=cur_pdf.squeeze(-1).exp().numpy()

            numerical_total_pdf+=cur_pdf*log_norms[0,cur_kde,0].exp().numpy()
            cur_sign=skew_signs[0,cur_kde, 0].numpy()
            cur_exponent=numpy.exp(log_exponents[0,cur_kde,0].numpy())
            ax2.plot(x, cur_pdf, label="weighted pdf %.2d (%d/%.2f)" % (cur_kde, cur_sign,cur_exponent))

        numerical_total_pdf/=total_norm_sum
        ax2.plot(x, total_pdf, color="black", label="tot pdf")
        ax2.plot(x, numerical_total_pdf, color="gray", ls="--", label="num. tot pdf")
        ax2.set_xlim(-50,50)
        ax2.legend(loc="upper left")

        if(not os.path.exists("./skewness_test")):
            os.makedirs("./skewness_test")

        pylab.savefig("./skewness_test/test_skewness_index_%.2d_%d.png" % (index, num_kde))

    

class Test(unittest.TestCase):
    
    def setUp(self, plot=None):
        self.plot=True

    def test_skewness(self):

        ## low exponent
        ## float precision

        seed_everything(0)

        for index in range(10):
                
            ## check if conditional pdf can be evaluated
            check_conditional_pdf(8, dimensionality=index+3)

            check_skewness(8, plot=self.plot, index=index)
        
            


if __name__ == '__main__':
    unittest.main()