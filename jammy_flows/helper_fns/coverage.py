import numpy
import torch
from scipy import stats

def find_closest(s, all_xyz_contours, contor_probs_all_cov):
    """
    Find closest contour, with a given contour coverage probability, of all passed contours to a given point *s*.
    Returns coverage probability of closest contour to s.
    """
    cur_min_index=-1
    overall_min=9999999999999999999999

    np_s=s
    if(type(np_s)==torch.Tensor):
        np_s=s.cpu().detach().numpy()

    for ind in range(len(all_xyz_contours)):

        min_dist=min(numpy.sqrt( numpy.sum((np_s-all_xyz_contours[ind])**2, axis=1)))

        if(min_dist<overall_min):
           
            overall_min=min_dist
            cur_min_index=ind

    return contor_probs_all_cov[cur_min_index]

def get_real_coverage_value(true_pos, xy_contours_for_coverage, actual_expected_coverage):
    """
    Calculate real coverage based on contours.
    """
    
    all_joined_contours=[]
    for ind in range(len(xy_contours_for_coverage)):

        joint=numpy.concatenate(xy_contours_for_coverage[ind], axis=0)
        all_joined_contours.append(joint)

    ## find closest contour to a point s
    cb=find_closest(true_pos, all_joined_contours, actual_expected_coverage)

    ## return contour probability
    return cb

def calculate_approximate_coverage(base_evals, dim, expected_coverage_probs):
    """
    Used by main class to calculate coverage for various scenarios.

    Returns: True coverage probs
             Twice logprobs
             chi2 CDF of true delta llh
    """

    gauss_log_eval_at_0=-(dim/2.0)*numpy.log(2*numpy.pi)
    actual_twice_logprob=2*(gauss_log_eval_at_0-base_evals)
  
    expected_twice_logprob=stats.chi2.ppf(expected_coverage_probs, df=dim)

    actual_coverage_probs=[]
   
    for ind,true_cov in enumerate(expected_coverage_probs):

        actual_coverage_probs.append(float(sum(actual_twice_logprob<expected_twice_logprob[ind]))/float(len(actual_twice_logprob)))

    return numpy.array(actual_coverage_probs), actual_twice_logprob, stats.chi2.cdf(actual_twice_logprob, df=dim) 
