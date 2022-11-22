import unittest
import sys
import os
import torch
import numpy
import pylab
import torch.autograd.functional
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jammy_flows.main.default as f

import jammy_flows.helper_fns as helper_fns


def seed_everything(seed_no):
    random.seed(seed_no)
    numpy.random.seed(seed_no)
    torch.manual_seed(seed_no)




def compare_two_arrays(arr1, arr2, name1, name2, diff_value=1e-7):

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


 
    diff_too_large_mask=numpy.fabs(arr1-arr2)>diff_value

    if(diff_too_large_mask.sum()>0):
        print("%s and %s contain elements (%d/%d) that differ by value greater than %.5e" % (name1, name2,diff_too_large_mask.sum(),len(diff_too_large_mask),diff_value))
        print("selected array 1 values ...")
        print(arr1[diff_too_large_mask])
        print("selected array 2 values ...")
        print(arr2[diff_too_large_mask])
        print(".. diffs")
        diffs=numpy.fabs(arr1-arr2)[diff_too_large_mask]
        print(diffs)
        print("min / maxn diff: ", diffs.min(), "/ ", diffs.max())
        raise Exception()

    print("largest diff between ", name1, " and ", name2, " (%d items): " % len(arr1),numpy.fabs(arr1-arr2).max() )

class Test(unittest.TestCase):
    def setUp(self):

        self.flow_inits=[]

        for extra_def in [{"amortization_mlp_dims":"64-64"}, {"conditional_input_dim":2, "amortization_mlp_dims":"64-64"}]:

            self.flow_inits.append([ ["s2", "y"], extra_def])
            
            self.flow_inits.append([ ["s2", "n"], extra_def])
            
            self.flow_inits.append([ ["e1", "gg"], extra_def])

            self.flow_inits.append([ ["e1+e1", "gg+gg"], extra_def])

            self.flow_inits.append([ ["e2", "ggg"], extra_def])
            
            self.flow_inits.append([ ["e1+e1+e1", "gg+gg+gg"], extra_def])

        self.flow_inits_numerical_comp=[]

        ## minimalistic options
        modified_options=dict()
        modified_options["g"]=dict()
        modified_options["g"]["num_kde"]=2
        modified_options["g"]["fit_normalization"]=0

        for extra_def in [{"options_overwrite": modified_options, "amortization_mlp_dims":"64-64"}, {"conditional_input_dim":2, "options_overwrite": modified_options,"amortization_mlp_dims":"64-64"}]:

            self.flow_inits_numerical_comp.append([ ["e1", "g"], extra_def])
            
            self.flow_inits_numerical_comp.append([ ["e1+e1", "g+g"], extra_def])

    
    def test_anumerical_entropy_and_grad(self):

        produce_plots=True
        print("comparing numerical and analytical entropy calculations")

        conditional_inputs=torch.DoubleTensor([[40.0,40.0],[-40.0,-40.0]])
        
        for init_ind, init in enumerate(self.flow_inits_numerical_comp):

            print("entropy test for ", init)
            seed_everything(1)
            tolerance=1e-7
            print("testtorch0 ", torch.randn(2))
            #seed_everything(0)
            print("init", init[1])
            this_flow=f.pdf(*init[0], **init[1])
            print("num params ",this_flow.count_parameters())
        
            this_flow.double()

            ## init with small damping factor
            this_flow.init_params(damping_factor=1.0)

            # 1-d
            if(not "e1+e1" in init[0][0]):

                eval_pts=torch.linspace(-5,6,10000, dtype=torch.double).unsqueeze(-1)
                bw=eval_pts[1][0]-eval_pts[0][0]

                this_conditional_input=None

                if("conditional_input_dim" in init[1]):

                    entropy_numerical=[]
                    #for cur_input in conditional_inputs:
                    log_prob, _,_=this_flow(eval_pts, conditional_input=conditional_inputs[:1,:].repeat(int(eval_pts.shape[0]), 1))
                    entropy_numerical.append(-(log_prob.exp()*log_prob*bw).sum().unsqueeze(-1))
                    
                    log_prob, _,_=this_flow(eval_pts, conditional_input=conditional_inputs[1:,:].repeat(int(eval_pts.shape[0]), 1))
                    entropy_numerical.append(-(log_prob.exp()*log_prob*bw).sum().unsqueeze(-1))
                        
                    entropy_numerical=torch.cat(entropy_numerical)
                    
                    entropy_analytic=this_flow.entropy(samplesize=10000, conditional_input=conditional_inputs)


                else:
                    log_prob, _,_=this_flow(eval_pts)
                    print("eval pts", eval_pts)
                    print("LOG PROB", log_prob)
                    entropy_numerical=-(log_prob.exp()*log_prob*bw).sum().unsqueeze(-1)

                    entropy_analytic=this_flow.entropy(samplesize=20000)

                for name, p in this_flow.named_parameters():
                    print("par NAME", name)

                    ## in exact entropy calculation, the entropy gradient will not at all depend on the offset.. therefore skip it
                    if("offsets" in name or "vs" in name):
                        continue

                    print(entropy_numerical)
                    
                    for grad_index in range(len(entropy_numerical)):
                        print("GRAD INDEX", grad_index)
                        grad_res=torch.autograd.grad(entropy_numerical[grad_index], p, allow_unused=False, retain_graph=True)
                        print(grad_res)

                        grad_res_analytic=torch.autograd.grad(entropy_analytic["total"][grad_index], p, allow_unused=False, retain_graph=True)
                        print(grad_res_analytic)

                        compare_two_arrays(grad_res[0], grad_res_analytic[0], "numerical_grad","analytic_grad", diff_value=1e-2)
                ###

                compare_two_arrays(entropy_numerical.detach(), entropy_analytic["total"].detach(), "numerical", "analytic", diff_value=1e-2)
             
            else:

                num_eval_pts_x=500
                num_eval_pts_y=600

                eval_pts_x=torch.linspace(-5.5,6.6,num_eval_pts_x, dtype=torch.double)#.unsqueeze(-1)
                eval_pts_y=torch.linspace(-5,6,num_eval_pts_y, dtype=torch.double)#.unsqueeze(-1)

                eval_grid_x, eval_grid_y=torch.meshgrid(eval_pts_x, eval_pts_y, indexing="xy")

                eval_xy=torch.cat([eval_grid_x.unsqueeze(-1), eval_grid_y.unsqueeze(-1)], dim=-1).reshape(num_eval_pts_x*num_eval_pts_y, -1)
                

                bw_x=eval_pts_x[1]-eval_pts_x[0]
                bw_y=eval_pts_y[1]-eval_pts_y[0]

                for cur_marginal_option in [0,1]:

                    this_conditional_input=None
                    print("cur marginal ..", cur_marginal_option)
                    if("conditional_input_dim" in init[1]):
                        
                        if(cur_marginal_option=="total"):
                            entropy_numerical=[]
                            for cur_input in conditional_inputs:
                                print("cur_INPUT", cur_input)
                                log_prob, _,_=this_flow(eval_xy, conditional_input=cur_input.unsqueeze(0).repeat(int(eval_xy.shape[0]), 1))
                                print("after first logprob")
                                #reshaped_log_probs=log_prob.reshape(num_eval_pts_x, num_eval_pts_y)

                                entropy_numerical.append(-(log_prob.exp()*log_prob*bw_x*bw_y).sum().unsqueeze(-1))
                                print("entropy numerical after fist", entropy_numerical)
                                #log_prob, _,_=this_flow(eval_xy, conditional_input=conditional_inputs[1:,:].repeat(int(eval_xy.shape[0]), 1))
                                #print("after second logprob")

                                #entropy_numerical.append(-(log_prob.exp()*log_prob*bw_x*bw_y).sum().unsqueeze(-1))
                            
                            entropy_numerical={"total":torch.cat(entropy_numerical)}
                           
                            entropy_analytic=this_flow.entropy(samplesize=10000, conditional_input=conditional_inputs)
                        else:
                            continue

                    else:

                        log_prob, _,_=this_flow(eval_xy)
                        print("evalxy", eval_xy)
                        if(cur_marginal_option=="total"):
                          
                            entropy_numerical=-(log_prob.exp()*log_prob*bw_x*bw_y).sum().unsqueeze(-1)
                            entropy_numerical={"total":entropy_numerical}

                            entropy_analytic=this_flow.entropy(samplesize=100000, sub_manifolds=[-1])
                        else:

                            ## resape logprobs .. first axis is y, second is x
                            print("logprobs")
                            print(log_prob)
                            reshaped_logprobs=log_prob.reshape(num_eval_pts_y, num_eval_pts_x)

                            
                            if(cur_marginal_option==0):
                                ## want to get marginal of x quantity, need to marginalize axis 0 (y)
                               
                                marginal_logprobs=torch.logsumexp(reshaped_logprobs+torch.log(bw_y), dim=0)
                                    
                                entropy_numerical=-(marginal_logprobs.exp()*marginal_logprobs*bw_x).sum().unsqueeze(-1)
                                entropy_numerical={0:entropy_numerical}


                                ## integrate out y
                                entropy_analytic=this_flow.entropy(samplesize=30000, sub_manifolds=[0])
                            else:


                                marginal_logprobs=torch.logsumexp(reshaped_logprobs+torch.log(bw_x), dim=1)
                                    
                                entropy_numerical=-(marginal_logprobs.exp()*marginal_logprobs*bw_y).sum().unsqueeze(-1)
                                entropy_numerical={1:entropy_numerical}

                            
                                ## integrate out y
                                entropy_analytic=this_flow.entropy(samplesize=500, sub_manifolds=[1])


                    diff_value_to_use=1e-2

                    ### crank up the tolerance a little bit to handle marginal=1 case with less samples (500)
                    if(cur_marginal_option == 1):
                        diff_value_to_use=4e-2
                    for name, p in this_flow.named_parameters():
                        print("par NAME", name)

                        ## in exact entropy calculation, the entropy gradient will not at all depend on the offset.. therefore skip it
                        if("offsets" in name or "vs" in name):
                            continue
                        #print("marginal option ", cur_marginal_option)
                        #print(entropy_numerical[cur_marginal_option])
                        #print("LEN OF ENTROPY VEC", len(entropy_numerical[cur_marginal_option]))
                        for grad_index in range(len(entropy_numerical[cur_marginal_option])):

                            grad_res_analytic=torch.autograd.grad(entropy_analytic[cur_marginal_option][grad_index], p, allow_unused=True, retain_graph=True)
                            if(grad_res_analytic[0] is None):
                                ## skip Nones, which happens sometime when no dependence exists, for example for first marginal when there are further autoregressive MLPs
                                continue

                            #print(grad_res_analytic)

                            #print("GRAD INDEX", grad_index)
                            grad_res=torch.autograd.grad(entropy_numerical[cur_marginal_option][grad_index], p, allow_unused=False, retain_graph=True)
                            #print(grad_res)

                            

                            compare_two_arrays(grad_res[0], grad_res_analytic[0], "numerical_grad","analytic_grad", diff_value=diff_value_to_use)
                    ###

                    compare_two_arrays(entropy_numerical[cur_marginal_option].detach(), entropy_analytic[cur_marginal_option].detach(), "numerical", "analytic", diff_value=diff_value_to_use)
    

    
    def test_entropy_and_marginal_entropy(self):

        print("-> Testing entropy and marginal entropy calculation <-")

        samplesizes=[3,10,100]

        #embedding_opts=[[0,1], [1,0], [0,0]]
        embedding_opts=[[1,0]]

        for init_ind, init in enumerate(self.flow_inits):
            ## seed everything to have consistent tests
            print("entropy test for ", init)
            seed_everything(1)
            tolerance=1e-7

            #seed_everything(0)
            this_flow=f.pdf(*init[0], **init[1])
            this_flow.double()

            cinput=None
            if("conditional_input_dim" in init[1].keys()):
                rvec=numpy.random.normal(size=(2,init[1]["conditional_input_dim"]))*1000.0
                cinput=torch.from_numpy(rvec)

            with torch.no_grad():

                for cur_samplesize in samplesizes:
                    for emb_option in embedding_opts:
                        num_sub_manifolds=len(this_flow.flow_defs_list)

                        print("samplesize: ", cur_samplesize)
                        print("embedding:", emb_option)
                        print("INPUT: ", cinput)
                        seed_everything(1)

                        entropy_dict_total=this_flow.entropy(samplesize=cur_samplesize, 
                                                       sub_manifolds=[-1], 
                                                       force_embedding_coordinates=emb_option[0], 
                                                       force_intrinsic_coordinates=emb_option[1],
                                                       conditional_input=cinput)


                        if(cinput is not None):
                            ## get first and second conditional input
                            seed_everything(1)
                            entropy_dict_1=this_flow.entropy(samplesize=cur_samplesize, 
                                                       sub_manifolds=[-1], 
                                                       force_embedding_coordinates=emb_option[0], 
                                                       force_intrinsic_coordinates=emb_option[1],
                                                       conditional_input=cinput[:1,:])

                            seed_everything(1)
                            entropy_dict_2=this_flow.entropy(samplesize=cur_samplesize, 
                                                       sub_manifolds=[-1], 
                                                       force_embedding_coordinates=emb_option[0], 
                                                       force_intrinsic_coordinates=emb_option[1],
                                                       conditional_input=cinput[1:,:])
                            print(entropy_dict_total)
                            print(entropy_dict_1)
                            print(entropy_dict_2)
                            #for sub_manifold in entropy_dict.keys():
                            assert( torch.abs(entropy_dict_total["total"][0]-entropy_dict_1["total"][0])<1e-12), (torch.abs(entropy_dict_total["total"][0]-entropy_dict_1["total"][0]))
                            assert( torch.abs(entropy_dict_total["total"][1]-entropy_dict_2["total"][0])<1e-12), (torch.abs(entropy_dict_total["total"][1]-entropy_dict_2["total"][0]))


                        print("total ", entropy_dict_total)

                        seed_everything(1)

                        entropy_dict_total_n_subs=this_flow.entropy(samplesize=cur_samplesize, 
                                                       sub_manifolds=[-1]+list(range(num_sub_manifolds)), 
                                                       force_embedding_coordinates=emb_option[0], 
                                                       force_intrinsic_coordinates=emb_option[1],
                                                       conditional_input=cinput)


                        compare_two_arrays(entropy_dict_total_n_subs["total"],entropy_dict_total["total"], "total", "total_from_subs", diff_value=1e-12)
                        
                        sub_manifold_options=[]

                        ## only all individuals
                        sub_manifold_options.append(list(range(num_sub_manifolds)))

                        ## also add single individual submanifolds
                        for mf_index in range(num_sub_manifolds):
                            sub_manifold_options.append([mf_index])

                        for cur_sub_manifold_opt in sub_manifold_options:
                            seed_everything(1)
                            print("CINPUT ", cinput)
                            print("cur sub manifold", cur_sub_manifold_opt)
                            entropy_dict=this_flow.entropy(samplesize=cur_samplesize, 
                                                           sub_manifolds=cur_sub_manifold_opt, 
                                                           force_embedding_coordinates=emb_option[0], 
                                                           force_intrinsic_coordinates=emb_option[1],
                                                           conditional_input=cinput)

                            if(cinput is not None):
                                ## get first and second conditional input
                                seed_everything(1)
                                entropy_dict_1=this_flow.entropy(samplesize=cur_samplesize, 
                                                           sub_manifolds=cur_sub_manifold_opt, 
                                                           force_embedding_coordinates=emb_option[0], 
                                                           force_intrinsic_coordinates=emb_option[1],
                                                           conditional_input=cinput[:1,:])

                                seed_everything(1)
                                entropy_dict_2=this_flow.entropy(samplesize=cur_samplesize, 
                                                           sub_manifolds=cur_sub_manifold_opt, 
                                                           force_embedding_coordinates=emb_option[0], 
                                                           force_intrinsic_coordinates=emb_option[1],
                                                           conditional_input=cinput[1:,:])
                                print(entropy_dict_total)
                                print(entropy_dict_1)
                                print(entropy_dict_2)
                                for other_sub_mfa in cur_sub_manifold_opt:
                                    #for sub_manifold in entropy_dict.keys():
                                    assert( torch.abs(entropy_dict[other_sub_mfa][0]-entropy_dict_1[other_sub_mfa][0])<1e-12), (torch.abs(entropy_dict[other_sub_mfa][0]-entropy_dict_1[other_sub_mfa][0]))
                                    assert( torch.abs(entropy_dict[other_sub_mfa][1]-entropy_dict_2[other_sub_mfa][0])<1e-12), (torch.abs(entropy_dict[other_sub_mfa][1]-entropy_dict_2[other_sub_mfa][0]))


                            
                            for sub_manifold in entropy_dict.keys():
                                compare_two_arrays(entropy_dict[sub_manifold], entropy_dict_total_n_subs[sub_manifold], "just_sub", "sub_w_total", diff_value=1e-12)

                            print("submf opt", cur_sub_manifold_opt)
                            print(entropy_dict)
                        
            #this_flow.count_parameters()
            #compare_two_arrays(base_samples.detach().numpy(), base_samples2.detach().numpy(), "base_samples", "base_samples2", diff_value=tolerance)
            #compare_two_arrays(evals.detach().numpy(), evals2.detach().numpy(), "evals", "evals2", diff_value=tolerance)
            #compare_two_arrays(base_evals.detach().numpy(), base_evals2.detach().numpy(), "base_evals", "base_evals2", diff_value=tolerance)
    
    

if __name__ == '__main__':
    unittest.main()