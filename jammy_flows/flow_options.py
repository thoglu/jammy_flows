from .layers.euclidean.gaussianization_flow import gf_block
from .layers.euclidean.gaussianization_flow_old import gf_block_old
from .layers.euclidean.polynomial_stretch_flow import psf_block
from .layers.euclidean.multivariate_normal import mvn_block
from .layers.euclidean.euclidean_do_nothing import euclidean_do_nothing

from .layers.spheres.moebius_1d import moebius
from .layers.spheres.splines_1d import spline_1d as sphere_spline_1d
from .layers.spheres.segmented_sphere_nd import segmented_sphere_nd
from .layers.spheres.exponential_map_s2 import exponential_map_s2
from .layers.spheres.spherical_do_nothing import spherical_do_nothing
from .layers.spheres.cnf_sphere_charts import cnf_sphere_charts

from .layers.intervals.interval_do_nothing import interval_do_nothing
from .layers.intervals.rational_quadratic_spline import rational_quadratic_spline

from .layers.simplex.inner_loop_simplex import inner_loop_simplex
from .layers.simplex.gumbel_softmax import gumbel_softmax


"""
Contains default inititialization options for the various flow layers. Default options can
be overwritten by passing *options_overwrite* to the pdf class.
"""

opts_dict=dict()

"""
Euclidean flows
"""

# Gaussianization flow
opts_dict["g"] = dict()
opts_dict["g"]["module"] = gf_block
opts_dict["g"]["type"] = "e"
opts_dict["g"]["kwargs"] = dict()
opts_dict["g"]["kwargs"]["fit_normalization"] = (1,[0,1])
opts_dict["g"]["kwargs"]["num_householder_iter"] = (-1, lambda x: (x==-1) or (x>0))
opts_dict["g"]["kwargs"]["num_kde"] = (10, lambda x: x>0)
opts_dict["g"]["kwargs"]["inverse_function_type"] = ("isigmoid", ["isigmoid", "inormal_partly_precise", "inormal_full_pade", "inormal_partly_crude"])
opts_dict["g"]["kwargs"]["replace_first_sigmoid_with_icdf"]=(1, [0,1])
opts_dict["g"]["kwargs"]["skip_model_offset"]=(0, [0,1])
opts_dict["g"]["kwargs"]["softplus_for_width"]=(0, [0,1]) # use softplus instead of exp to transform log_width -> width
opts_dict["g"]["kwargs"]["upper_bound_for_widths"]=(100, lambda x: (x==-1) or x>0) # define an upper bound for the value of widths.. -1 = no upper bound
opts_dict["g"]["kwargs"]["lower_bound_for_widths"]=(0.01, lambda x: x>0) # define a lower bound for the value of widths
opts_dict["g"]["kwargs"]["upper_bound_for_norms"]=(10, lambda x: (x==-1) or x>0) # define an upper bound for the value of widths.. -1 = no upper bound
opts_dict["g"]["kwargs"]["lower_bound_for_norms"]=(1, lambda x: x>0) # define a lower bound for the value of widths
opts_dict["g"]["kwargs"]["center_mean"]=(0, [0,1]) # center the mean of each mixture PDF
opts_dict["g"]["kwargs"]["clamp_widths"]=(0, [0,1])
opts_dict["g"]["kwargs"]["width_smooth_saturation"]=(1, [0,1]) # 
opts_dict["g"]["kwargs"]["regulate_normalization"]=(1, [0,1])
opts_dict["g"]["kwargs"]["add_skewness"]=(0, [0,1])

opts_dict["g"]["kwargs"]["rotation_mode"]=("householder", ["householder", "triangular_combination", "angles", "cayley"])
opts_dict["g"]["kwargs"]["nonlinear_stretch_type"]=("classic", ["classic", "rq_splines"])

# Old Gaussianization flow implementation (deprecated)
opts_dict["h"] = dict()
opts_dict["h"]["module"] = gf_block_old
opts_dict["h"]["type"] = "e"
opts_dict["h"]["kwargs"] = dict()
opts_dict["h"]["kwargs"]["fit_normalization"] = (1,[0,1])
opts_dict["h"]["kwargs"]["num_householder_iter"] = (-1, lambda x: (x==-1) or (x>0))
opts_dict["h"]["kwargs"]["num_kde"] = (10, lambda x: x>0)
opts_dict["h"]["kwargs"]["inverse_function_type"] = ("isigmoid", ["isigmoid", "inormal_partly_precise", "inormal_full_pade", "inormal_partly_crude"])
opts_dict["h"]["kwargs"]["replace_first_sigmoid_with_icdf"]=(1, [0,1])
opts_dict["h"]["kwargs"]["skip_model_offset"]=(0, [0,1])
opts_dict["h"]["kwargs"]["softplus_for_width"]=(0, [0,1]) # use softplus instead of exp to transform log_width -> width
opts_dict["h"]["kwargs"]["upper_bound_for_widths"]=(100, lambda x: (x==-1) or x>0) # define an upper bound for the value of widths.. -1 = no upper bound
opts_dict["h"]["kwargs"]["lower_bound_for_widths"]=(0.01, lambda x: x>0) # define a lower bound for the value of widths
opts_dict["h"]["kwargs"]["clamp_widths"]=(0, [0,1])
opts_dict["h"]["kwargs"]["width_smooth_saturation"]=(1, [0,1]) # 
opts_dict["h"]["kwargs"]["regulate_normalization"]=(1, [0,1])
opts_dict["h"]["kwargs"]["add_skewness"]=(0, [0,1])

# polynomial stretch flow
opts_dict["p"] = dict()
opts_dict["p"]["module"] = psf_block
opts_dict["p"]["type"] = "e"
opts_dict["p"]["kwargs"] = dict()
opts_dict["p"]["kwargs"]["num_householder_iter"] = (-1, lambda x: (x==-1) or (x>0))
opts_dict["p"]["kwargs"]["num_transforms"] = (1, lambda x: x>0)
opts_dict["p"]["kwargs"]["exact_mode"] = (True, [True, False])
opts_dict["p"]["kwargs"]["skip_model_offset"]=(0, [0,1])

# Multivariate Normal
opts_dict["t"] = dict()
opts_dict["t"]["module"] = mvn_block
opts_dict["t"]["type"] = "e"
opts_dict["t"]["kwargs"] = dict()
opts_dict["t"]["kwargs"]["skip_model_offset"]=(0, [0,1])
opts_dict["t"]["kwargs"]["softplus_for_width"]=(0, [0,1]) # use softplus instead of exp to transform log_width -> width
opts_dict["t"]["kwargs"]["upper_bound_for_widths"]=(100, lambda x: (x==-1) or x>0) # define an upper bound for the value of widths.. -1 = no upper bound
opts_dict["t"]["kwargs"]["lower_bound_for_widths"]=(0.01, lambda x: x>0) # define a lower bound for the value of widths
opts_dict["t"]["kwargs"]["clamp_widths"]=(0, [0,1])
opts_dict["t"]["kwargs"]["width_smooth_saturation"]=(1, [0,1]) # 
opts_dict["t"]["kwargs"]["cov_type"]=("diagonal", ["identity", "diagonal_symmetric", "diagonal", "full"])



"""
S1 flows
"""

## Moebius flow
opts_dict["m"] = dict()
opts_dict["m"]["module"] = moebius
opts_dict["m"]["type"] = "s"
opts_dict["m"]["kwargs"] = dict()
opts_dict["m"]["kwargs"]["add_rotation"] = (0, [0,1])
opts_dict["m"]["kwargs"]["num_basis_functions"] = (5, lambda x: x>0)
opts_dict["m"]["kwargs"]["natural_direction"] = (0, [0,1])

## Spline-Based 1-d flow
opts_dict["o"] = dict()
opts_dict["o"]["module"] = sphere_spline_1d
opts_dict["o"]["type"] = "s"
opts_dict["o"]["kwargs"] = dict()
opts_dict["o"]["kwargs"]["add_rotation"] = (0, [0,1])
opts_dict["o"]["kwargs"]["num_basis_functions"] = (5, lambda x: x>0)
opts_dict["o"]["kwargs"]["natural_direction"] = (0, [0,1])


"""
S2 flows
"""

## 2-d spherical flow based on conditional implementation
opts_dict["n"] = dict()
opts_dict["n"]["module"] = segmented_sphere_nd
opts_dict["n"]["type"] = "s"
opts_dict["n"]["kwargs"] = dict()
opts_dict["n"]["kwargs"]["add_rotation"] = (1, [0,1])
opts_dict["n"]["kwargs"]["rotation_mode"] = ("householder", ["householder", "angles"])
opts_dict["n"]["kwargs"]["hidden_dims"] = ("64", lambda x: type(x)==str)
opts_dict["n"]["kwargs"]["num_basis_functions"] = (10, lambda x: x>0)
opts_dict["n"]["kwargs"]["higher_order_cylinder_parametrization"] = (False, [True, False])
opts_dict["n"]["kwargs"]["zenith_type_layers"] = ("r", lambda x: sum([i in ["r", "g", "p", "x", "z", "t"] for i in x])==len(x) )
opts_dict["n"]["kwargs"]["max_rank"] = (-1, lambda x: (x==-1) or (x>0))
opts_dict["n"]["kwargs"]["subspace_mapping"] = ("logistic", ["logistic"])
opts_dict["n"]["kwargs"]["highway_mode"] = (0, [0,1,2,3,4])

# exponential map s2 flow
opts_dict["v"] = dict()
opts_dict["v"]["module"] = exponential_map_s2
opts_dict["v"]["type"] = "s"
opts_dict["v"]["kwargs"] = dict()
opts_dict["v"]["kwargs"]["exp_map_type"] = ("exponential", ["linear", "quadratic", "splines", "exponential"]) ## supported linear  / exponential
opts_dict["v"]["kwargs"]["num_components"] = (10, lambda x: x>0) ## number of components in convex superposition
opts_dict["v"]["kwargs"]["natural_direction"] = (0, [0,1]) ## natural direction corresponds to the transformation happing in the forward direction - default: 0 (0 faster pdf eval, 1 fast sampling)
opts_dict["v"]["kwargs"]["add_rotation"] = (0, [0,1]) ## natural direction corresponds to the transformation happing in the forward direction - default: 0 (0 faster pdf eval, 1 fast sampling)


# Manifold Continuous normalizing flow
opts_dict["c"] = dict()
opts_dict["c"]["module"] = cnf_sphere_charts
opts_dict["c"]["type"] = "s"
opts_dict["c"]["kwargs"] = dict()
opts_dict["c"]["kwargs"]["num_charts"] = (4, lambda x: x>0)
opts_dict["c"]["kwargs"]["cnf_network_hidden_dims"] = ("32", lambda x: type(x)==str) # hidden dims of cnf MLP network
opts_dict["c"]["kwargs"]["cnf_network_highway_mode"] = (0, [0,1,2,3,4]) # mlp highway dim - 0-4
opts_dict["c"]["kwargs"]["cnf_network_rank"] = (-1, lambda x:  (x==-1) or x>0) # -1 means full rank
opts_dict["c"]["kwargs"]["solver"] = ("dopri5", ["rk4", "dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint"]) ## 
opts_dict["c"]["kwargs"]["rtol"] = (1e-7, lambda x: (x>0) & (x<1)) ## 
opts_dict["c"]["kwargs"]["atol"] = (1e-7, lambda x: (x>0) & (x<1)) ## 
opts_dict["c"]["kwargs"]["step_size"] = (1.0/32.0, lambda x: (x>0) )  ## 

"""
Interval flows
"""

# Rational quadratic spline
opts_dict["r"] = dict()
opts_dict["r"]["module"] = rational_quadratic_spline
opts_dict["r"]["type"] = "i"
opts_dict["r"]["kwargs"] = dict()
opts_dict["r"]["kwargs"]["num_basis_elements"] = (10, lambda x: x>0)

"""
Simplex flows
"""

## gumbel softmax
opts_dict["u"] = dict()
opts_dict["u"]["module"] = gumbel_softmax
opts_dict["u"]["type"] = "a"
opts_dict["u"]["kwargs"] = dict()

## generic simplex flow
opts_dict["w"] = dict()
opts_dict["w"]["module"] = inner_loop_simplex
opts_dict["w"]["type"] = "a"
opts_dict["w"]["kwargs"] = dict()

"""
Spherical/Euclidean/Interval flows that do nothing
"""

opts_dict["x"] = dict()
opts_dict["x"]["module"] = euclidean_do_nothing
opts_dict["x"]["type"] = "e"
opts_dict["x"]["kwargs"] = dict()

opts_dict["y"] = dict()
opts_dict["y"]["module"] = spherical_do_nothing
opts_dict["y"]["type"] = "s"
opts_dict["y"]["kwargs"] = dict()

opts_dict["z"] = dict()
opts_dict["z"]["module"] = interval_do_nothing
opts_dict["z"]["type"] = "i"
opts_dict["z"]["kwargs"] = dict()

def obtain_default_options(flow_abbrevation):
    """
    Obtains a dictionary with default options for a given flow.
    For a complete list of possible options, have a look in *flow_options.py*.

    Parameters:
        flow_abbreviation (str): The character specifying a particular flow layer.

    Returns:
        dict
            Dictionary containing the default options.
    """
    assert(flow_abbrevation in opts_dict.keys()), "Unknown flow abbreviation for default options: %s" % flow_abbrevation
        
    # Copy default options into dictionary
    return {k: opts_dict[flow_abbrevation]["kwargs"][k][0] for k in opts_dict[flow_abbrevation]["kwargs"].keys()}

def check_flow_option(flow_abbrevation, opt_name, opt_val):
    """
    Makes sure configured option is allowed. Used internally.
    """
    assert(flow_abbrevation in opts_dict.keys()), ("flow abbreviation %s not found in options dict" % flow_abbrevation)
    assert(opt_name in opts_dict[flow_abbrevation]["kwargs"].keys()), ("option name %s not found in defined options for flow %s" % (opt_name, flow_abbrevation))

    if(hasattr(opts_dict[flow_abbrevation]["kwargs"][opt_name][1], "__call__")):
        # lambda function check
        assert(opts_dict[flow_abbrevation]["kwargs"][opt_name][1](opt_val)), ("Lambda function check of configured option", opt_name, " failed with value ", opt_val)

    elif(type(opts_dict[flow_abbrevation]["kwargs"][opt_name][1])==list):

        assert(opt_val in opts_dict[flow_abbrevation]["kwargs"][opt_name][1]), ("Configured option ", opt_name, " with value ", opt_val, " not part of allowed options: ", opts_dict[flow_abbrevation]["kwargs"][opt_name][1])
    else:
        raise Exception("Unknown value check type!", type(opts_dict[flow_abbrevation]["kwargs"][opt_name][1]))

def obtain_overall_flow_info():
    """
    Obtains module and type information of all flows in a dict. Used internally.
    """
    info_object=dict()
    for k in opts_dict.keys():
        info_object[k]=dict()
        info_object[k]["type"]=opts_dict[k]["type"]
        info_object[k]["module"]=opts_dict[k]["module"]

    return info_object
