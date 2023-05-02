from torch import nn
import torch
import numpy
import time
import scipy

from .layers.euclidean import gaussianization_flow, multivariate_normal
from .layers import matrix_fns
from scipy.optimize import minimize



def log_one_plus_exp_x_to_a_minus_1(x, a):

    """
    Calculates the logarithm of ((1+exp(x))**a - 1) / ((1+exp(x))**a)

    Parameters:
        x (Tensor): Size (B,D)
        a (Tensor): Size (B,1)

    Returns:
        Tensor
            Result
    """

    assert(x.dtype==torch.double), "This function requires double precision to work properly"
    assert(x.shape[1:]==a.shape[1:])

    ## softplus result must be positive by definition
    softplus_result=a*nn.functional.softplus(x)

    ## small x values
    x_small_mask=x<=-20

    res=torch.where(x_small_mask, torch.log(a)+x, 0.0)

    ### check the value of softplus
    soft_plus_large=softplus_result > 20

    ## we can neglect the final -1 since the soft plus term is large
    mask_2=(~x_small_mask) & soft_plus_large
    
    res=torch.where(mask_2, softplus_result, res)

    ### check the value of softplus
    soft_plus_small=softplus_result < 1e-8

    mask_3=(~x_small_mask) & soft_plus_small

    res=torch.masked_scatter(input=res, mask=mask_3, source=torch.log(softplus_result[mask_3]))

    mask_4=(~x_small_mask) & (~soft_plus_large) & (~soft_plus_small)
    
    res=torch.masked_scatter(input=res, mask=mask_4, source=torch.log(torch.exp(softplus_result[mask_4])-1.0))

    if( (torch.isfinite(res)==False).sum()>0):
        print("LOGPLUS1 FAULTY: RES", res)
        raise Exception("Non-finite values")
    return res-softplus_result

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}

def list_from_str(spec):
    if(spec==""):
        return []
        
    return list(tuple(map(int, spec.split("-"))))

### initializations

## transformations

def get_loss_fn(target_matrix, num_householder_iter=-1):

    dim=target_matrix.shape[0]

    
    def compute_matching_distance(a):

        gblock=gaussianization_flow.gf_block(dim, num_householder_iter=num_householder_iter)

        hh_pars=torch.from_numpy(numpy.reshape(a, gblock.vs.shape))
        mat=gblock.compute_householder_matrix(hh_pars).squeeze(0).detach().numpy()

        test_vec=numpy.ones(dim)
        test_vec/=numpy.sqrt((test_vec**2).sum())

        v1=numpy.matmul(mat,test_vec)
        v2=numpy.matmul(target_matrix,test_vec)

        return -(v1*v2).sum()

    return compute_matching_distance

def get_loss_fn_mvn(target_matrix, cov_type):

    dim=target_matrix.shape[0]

    

    def compute_matching_distance(a):
       
        test_block=multivariate_normal.mvn_block(dim,cov_type=cov_type)

        pars=test_block._obtain_usable_flow_params(a, cov_type, extra_inputs=torch.from_numpy(a).unsqueeze(0))

        lower_trig, _=matrix_fns.obtain_lower_triangular_matrix_and_logdet(dim, single_log_diagonal_entry=pars[0], log_diagonal_entries=pars[1], lower_triangular_entries=pars[2], cov_type=cov_type)

        lower_trig=lower_trig[0].numpy()

        predicted_matrix=numpy.matmul(lower_trig, lower_trig.T)
        
        ## it seems when the target data has larger variance than the unit gaussian, it is much faster to just take the reverse KL    
        #inverse_predicted=scipy.linalg.pinv(predicted_matrix)
        #forward_kl=0.5*(numpy.trace(numpy.matmul(inverse_predicted, target_matrix))+numpy.linalg.slogdet(predicted_matrix)[1]-numpy.linalg.slogdet(target_matrix)[1]-dim)

        ## inverse

        inverse_target=scipy.linalg.pinv(target_matrix)
    
        reverse_kl=0.5*(numpy.trace(numpy.matmul(inverse_target, predicted_matrix))-numpy.linalg.slogdet(predicted_matrix)[1]+numpy.linalg.slogdet(target_matrix)[1]-dim)

        return reverse_kl
        
    return compute_matching_distance

def get_trafo_matrix_mvn(dimension, params, cov_type):
    """
    Return the transformation matrix to apply to the data, in order to remove moments.
    """

    test_block=multivariate_normal.mvn_block(dimension,cov_type=cov_type)

    pars=test_block._obtain_usable_flow_params(params, cov_type, extra_inputs=torch.from_numpy(params).unsqueeze(0))

    lower_trig, _=matrix_fns.obtain_lower_triangular_matrix_and_logdet(dimension, single_log_diagonal_entry=pars[0], log_diagonal_entries=pars[1], lower_triangular_entries=pars[2], cov_type=cov_type)

    lower_trig=lower_trig[0].numpy()

   
    predicted_matrix=numpy.matmul(lower_trig, lower_trig.T)

    inverse_predicted=scipy.linalg.pinv(predicted_matrix)
    
    l, sigma, r=scipy.linalg.svd(inverse_predicted)

    ## return "sqrt" of covariance matrix
    return numpy.sqrt(sigma)*r

        
def find_init_pars_of_chained_blocks(layer_list, data, mvn_min_max_sv_ratio=1e-4):

    ## given an input *data_inits*, this function tries to initialize the gf block parameters
    ## to best match the data intis
    
    cur_data=data

    dim=None
    if(data is not None):
        dim=data.shape[1]

    all_layers_params=[]

    tot_num_expected_params=0

    with torch.no_grad():
        ## traverse layers in reversed order
        for layer_ind, cur_layer in enumerate(layer_list[::-1]):
            tot_num_expected_params+=cur_layer.total_param_num
            # normal default layer init if not data present
            if(data is None):
                all_layers_params.append(cur_layer.get_desired_init_parameters().type(torch.float64))
                continue

            ## param order .. householder / means / width / normaliaztion
            param_list=[]

            ## mean can exist for all layers
            ## subtract means first if necessary
            if(cur_layer.model_offset):

                means=cur_data.mean(axis=0,keepdim=True)
               
                param_list.append(means.squeeze(0))
                
                cur_data=cur_data-means 
            
            # multivariate normal
            if(type(cur_layer)==multivariate_normal.mvn_block):

                if(cur_layer.cov_type=="identity"):
                    # no more variables in layer.. just continue
                    if(len(param_list)>0):
                        all_layers_params.append(torch.cat(param_list))
                    continue
             
                data_matrix=torch.matmul(cur_data.T, cur_data)/float(data.shape[0])

                # svd to increase small eigenvalues and "fix" the data matrix
                l, sigma, r=scipy.linalg.svd(data_matrix)

                minimum_allowed_singular_val=mvn_min_max_sv_ratio*max(sigma)
                new_sigma=numpy.where(sigma<minimum_allowed_singular_val, minimum_allowed_singular_val, sigma)

                fixed_data_matrix=(l*new_sigma) @ r

                loss_fn=get_loss_fn_mvn(fixed_data_matrix, cur_layer.cov_type)

                num_mat_params=cur_layer.total_param_num
                if(cur_layer.model_offset):
                    num_mat_params-=dim

                start_vec=numpy.random.normal(size=num_mat_params)
                res=minimize(loss_fn, start_vec)

                #### set mvn params
                param_list.append(torch.from_numpy(res["x"]).to(data))

                inverse_trafo_matrix=torch.from_numpy(get_trafo_matrix_mvn(dim, res["x"], cur_layer.cov_type)).unsqueeze(0)

                batch_rotation_matrix=inverse_trafo_matrix.repeat(cur_data.shape[0], 1,1)

                cur_data = torch.bmm(batch_rotation_matrix, cur_data.unsqueeze(-1)).squeeze(-1)

            # gaussianization flows
            elif(type(cur_layer)==gaussianization_flow.gf_block):

                if(cur_layer.rotation_mode=="triangular_combination"):
                    cur_data=cur_data
                    param_list.append(torch.zeros(cur_layer.num_triangle_params))
                elif(cur_layer.rotation_mode=="householder"):
                    if(cur_layer.use_householder):

                        ## find householder params that correspond to orthogonal transformation of svd of X^T*X (PCA data matrix) if low dimensionality
                        this_vs=0

                        ## USE PCA for first layer to get major correlation out of the way
                        if(cur_layer.dimension<30 and layer_ind==0):

                            data_matrix=torch.matmul(cur_data.T, cur_data)

                            evalues, evecs=scipy.linalg.eig(data_matrix.cpu().numpy())

                            l, sigma, r=scipy.linalg.svd(data_matrix.cpu().numpy())
                            
                            loss_fn=get_loss_fn(r, num_householder_iter=cur_layer.householder_iter)

                            start_vec=numpy.random.normal(size=dim*dim)

                            ## fit a matrix via householder parametrization such that it fits the target orthogonal matrix V^* from SVD of X^T*X (PCA data Matrix)
                            res=minimize(loss_fn, start_vec)

                            param_list.append(torch.from_numpy(res["x"]).to(data))
                            this_vs=torch.from_numpy(res["x"]).to(data)
                            
                        else:

                            this_vs=torch.randn(cur_layer.dimension*cur_layer.householder_iter).to(data)
                            param_list.append(this_vs)

                        gblock=gaussianization_flow.gf_block(dim, num_householder_iter=cur_layer.householder_iter)

                        hh_pars=this_vs.reshape(gblock.vs.shape)
                        
                        rotation_matrix=gblock.compute_householder_matrix(hh_pars)
                        rotation_matrix=rotation_matrix.repeat(cur_data.shape[0], 1,1)
                        
                        ## inverted matrix
                        cur_data = torch.bmm(rotation_matrix.permute(0,2,1), cur_data.unsqueeze(-1)).squeeze(-1)
                elif(cur_layer.rotation_mode=="angles"):
                    cur_data=cur_data
                    param_list.append(torch.zeros(cur_layer.num_angle_pars))
                elif(cur_layer.rotation_mode=="cayley"):
                    cur_data=cur_data
                    param_list.append(torch.zeros(cur_layer.num_cayley_pars))

             
                num_kde=cur_layer.num_kde

                assert(num_kde<100)

                if(cur_layer.nonlinear_stretch_type=="classic"):

                    ## TODO: This whole init routine for GFs should probably be reworked at some point.
                    
                    ## based on percentiles
                    percentiles_to_use=numpy.linspace(0,100,num_kde)#[1:-1]
                    percentiles=torch.from_numpy(numpy.percentile(cur_data.cpu().numpy(), percentiles_to_use, axis=0)).to(data)


                    ## add means
                    if(cur_layer.center_mean==0):
                        param_list.append(percentiles.flatten())
                    else:
                        param_list.append(percentiles[:-1].flatten())
                 
                    quarter_diffs=percentiles[1:,:]-percentiles[:-1,:]
                    min_perc_diff=(quarter_diffs.min(axis=0, keepdim=True)[0])

                
                    ## this seems to be optimized settings for num_kde=20
                    bw=torch.log(min_perc_diff*1.5)
                    
                    bw=torch.ones_like(percentiles[None,:,:])*bw
                   
                    flattened_bw=bw.flatten()
                    #############
                    """
                    fig=pylab.figure()

                    for x in cur_data:
                        pylab.gca().axvline(x[0],color="black")
                    log_yvals=cur_layer.logistic_kernel_log_pdf(torch.from_numpy(pts)[:,None], percentiles[None,:,0:1], bw[:,:,0:1], torch.ones_like(percentiles[None,:,0:1]))
                    yvals=log_yvals.exp().detach().numpy()
                    pylab.gca().plot(pts, yvals, color="green")

                    pylab.savefig("test_kde_0.png")


                    fig=pylab.figure()

                    for x in cur_data:
                        pylab.gca().axvline(x[1],color="black")
                    log_yvals=cur_layer.logistic_kernel_log_pdf(torch.from_numpy(pts)[:,None], percentiles[None,:,1:2], bw[:,:,1:2], torch.ones_like(percentiles[None,:,1:2]))
                    yvals=log_yvals.exp().detach().numpy()
                    pylab.gca().plot(pts, yvals, color="green")

                    pylab.savefig("test_kde_1.png")

                    print("CUR PARAMS", param_list)
                    ##########

                    """
                    
                    param_list.append(torch.flatten(bw))


                    ## widths

                    if(cur_layer.fit_normalization):

                        ## norms

                        param_list.append(torch.ones_like(flattened_bw))

                    ## skewness is not used, just as a single multiplicator
                    this_skewness_exponent=torch.DoubleTensor([1.0]).to(data)

                    # signs is not used, used as a single multiplicator
                    this_skewness_signs=torch.DoubleTensor([1.0]).to(data)

                    if(cur_layer.add_skewness):

                        ## store zeros (log_exponents) in params
                        param_list.append(torch.zeros_like(flattened_bw))

                        ## pass exponents as 1.0
                        this_skewness_exponent=cur_layer.exponent_regulator(torch.zeros_like(bw)).exp()

                        this_skewness_signs=cur_layer.kde_skew_signs

                    ## transform params according to CDF_norm^-1(CDF_KDE)

                    #gblock=gf_block(dim, num_householder_iter=cur_layer.householder_iter)
                    cur_data=cur_layer.sigmoid_inv_error_pass_w_params(cur_data, percentiles[None,:,:], bw, torch.ones_like(bw), this_skewness_exponent, this_skewness_signs)
                
                else:
                    raise Exception("Data initilaization only implemented (and probably only makes sense) for classic Gaussianization Flow structure")

            else:

                ## default transformation .. importantly WITHOUT mean shift (with the _ in front), since the mean shift was done earlier
                param_list.append(cur_layer._get_desired_init_parameters().type(torch.float64))
                
            all_layers_params.append(torch.cat(param_list))

    all_layers_params=torch.cat(all_layers_params[::-1])

    assert(len(all_layers_params)==tot_num_expected_params), ("Total number of defined params (%d) does not match expected params based on layer definitions (%d)" % (len(all_layers_params), tot_num_expected_params))
    
    return all_layers_params

def _calculate_coverage(base_evals, dim, expected_coverage_probs):
    """
    Used by main class to calculate coverage for various scenarios.

    Returns: True coverage probs
             Twice logprobs
    """

    gauss_log_eval_at_0=-(dim/2.0)*numpy.log(2*numpy.pi)
    actual_twice_logprob=2*(gauss_log_eval_at_0-base_evals)
  
    expected_twice_logprob=scipy.stats.chi2.ppf(expected_coverage_probs, df=dim)

    actual_coverage_probs=[]
   
    for ind,true_cov in enumerate(expected_coverage_probs):

        actual_coverage_probs.append(float(sum(actual_twice_logprob<expected_twice_logprob[ind]))/float(len(actual_twice_logprob)))

    return numpy.array(actual_coverage_probs), actual_twice_logprob 