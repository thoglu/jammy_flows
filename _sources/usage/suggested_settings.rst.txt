**************************
Suggested Settings
**************************

The settings of a normalizing flow can drastically change its behavior. In the following, we summarize recommended standard settings to start with. Settings for a normalizing flow are custumized 
via the **options_overwrite** keyword passed at construction.

--------------------------------
Euclidean PDFs
--------------------------------

For an expressive Euclidean PDF we recommend the Gaussianization flow ("g") followed by an affine Flow ("t").
For an expressive result, one typically wants to use at least as many "g" layers as the dimension of the problem.

Example for 1-d:

..  code-block:: python

    opt_dict=dict()

    opt_dict["g"]=dict()
    opt_dict["g"]["fit_normalization"]=0 # normalization switched off can be numerically more stable
    opt_dict["g"]["upper_bound_for_widths"]=1.0 # bound found empirically to work well
    opt_dict["g"]["lower_bound_for_widths"]=0.01 # bound found empirically to work well

    pdf=jammy_flows.pdf("e1", "gggt", options_overwrite=opt_dict) # also in 1-d multiple g flows can help, especially with tail behavior

Example for 3-d:

..  code-block:: python

    opt_dict=dict()
    opt_dict["t"]=dict()
    opt_dict["t"]["cov_type"]="full" # full covariance matrix (only use if dimension > 1)
    opt_dict["g"]=dict()
    opt_dict["g"]["fit_normalization"]=0 # normalization switched off can be numerically more stable
    opt_dict["g"]["upper_bound_for_widths"]=1.0 # bound found empirically to work well
    opt_dict["g"]["lower_bound_for_widths"]=0.01 # bound found empirically to work well

    pdf=jammy_flows.pdf("e3", "gggggt", options_overwrite=opt_dict)

--------------------------------
Spherical PDF (2-sphere)
--------------------------------

A combination of smooth neural spline flows interwoven with von-Mises-Fisher scalings as used in 
https://arxiv.org/abs/2604.19846 is recommended as a starting point. 

..  code-block:: python

    opt_dict=dict()
    opt_dict["f"]=dict()
    opt_dict["f"]["add_vertical_rq_spline_flow"]=1
    opt_dict["f"]["spline_num_basis_functions"]=-1
    opt_dict["f"]["vertical_smooth"]=1
    opt_dict["f"]["vertical_flow_defs"]="rr"
    opt_dict["f"]["circular_flow_defs"] = "oo"
    opt_dict["f"]["vertical_fix_boundary_derivative"]=1
    opt_dict["f"]["min_kappa"]=1e-10
    opt_dict["f"]["kappa_prediction"]="direct_log_real_bounded"
    opt_dict["f"]["kappa_clamping"]=0
    opt_dict["f"]["vertical_restrict_max_min_width_height_ratio"]=-1.0
    opt_dict["f"]["vertical_fix_first_width_n_height_to_zero"]=1 # fix the first width/height to 0
    opt_dict["f"]["vertical_independent_width_height_parametrization"]=1 # better conditioned 
    opt_dict["f"]["add_circular_rq_spline_flow"]=1 # add circle flow
    opt_dict["f"]["circular_add_rotation"]=0 # no extra rotation on circle flow
    opt_dict["f"]["vertical_also_fix_second_width_to_zero"]=1
    opt_dict["f"]["rotation_mode"]="householder" 

    pdf=jammy_flows.pdf("s2", "fffffffffffffff", options_overwrite=opt_dict)

Take more or less "f" flows as needed, depending on the complexity.

--------------------------------
Spherical PDF (1-sphere -> PDF on the circle)
--------------------------------

Both the Moebius flow ("m") and periodic circular spline flow ("o") should work, although the m flow is probably the better default choice. It should work rather well straight out of the box:

..  code-block:: python

    opt_dict=dict()

    pdf=jammy_flows.pdf("s1", "m", options_overwrite=opt_dict)

