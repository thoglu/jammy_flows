******************
API Documentation
******************

General
=============================

Main class
----------------------------

.. automodule:: jammy_flows.flows
    :members: 
    :special-members: __init__
    :exclude-members: obtain_flow_param_structure

.. automodule:: jammy_flows.flow_options
    :members: obtain_default_options

.. automodule:: jammy_flows.extra_functions
    :members: log_one_plus_exp_x_to_a_minus_1

Amortizable MLP
----------------------------

.. automodule:: jammy_flows.amortizable_mlp
    :members:
    :special-members: __init__

Numerical inversion
----------------------------

.. automodule:: jammy_flows.layers.bisection_n_newton
    :members:


Helper Functions
----------------------------

.. automodule:: jammy_flows.helper_fns
    :members:

Layers
=============================

.. automodule:: jammy_flows.layers.layer_base
    :members:
    :special-members: __init__
    :private-members: _embedding_conditional_return, _embedding_conditional_return_num


Euclidean flow layers
----------------------------

Euclidean Base 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.euclidean.euclidean_base
    :members:
    :exclude-members: flow_mapping, get_desired_init_parameters, init_params, inv_flow_mapping, obtain_layer_param_structure, transform_target_space
    :special-members: __init__

Identity layer ("x")
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.euclidean.euclidean_do_nothing
    :members:
    :special-members: __init__

Multivariate Normal ("t")
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.euclidean.multivariate_normal
    :members:
    :special-members: __init__

Gaussianization flow ("g")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.euclidean.gaussianization_flow
    :members:
    :special-members: __init__
    :exclude-members: sigmoid_inv_error_pass_combined_val_n_log_derivative, sigmoid_inv_error_pass_combined_val_n_normal_derivative

Polynomial stretch flow ("p")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.euclidean.polynomial_stretch_flow
    :members:
    :special-members: __init__

Spherical flow layers
----------------------------

.. automodule:: jammy_flows.layers.spheres.sphere_base
    :members:
    :exclude-members: flow_mapping, get_desired_init_parameters, init_params, inv_flow_mapping, obtain_layer_param_structure, return_safe_angle_within_pi, return_problematic_pars_between_hh_and_intrinsic, transform_target_space
    :special-members: __init__

Identity layer ("y")
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.spheres.spherical_do_nothing
    :members:
    :special-members: __init__

Moebius 1-d circle flow ("m")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: jammy_flows.layers.spheres.moebius_1d
    :members:
    :special-members: __init__

Circular 1-d spline flow ("o")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: jammy_flows.layers.spheres.splines_1d
    :members:
    :special-members: __init__

Segmented 2-d sphere flow ("n")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.spheres.segmented_sphere_nd
    :members:
    :special-members: __init__
    :exclude-members: to_subspace, from_subspace

Exponential map 2-d sphere flow ("v")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.spheres.exponential_map_s2
    :members:
    :special-members: __init__
    :exclude-members: generate_normalization_function, basic_exponential_map, basic_logarithmic_map, get_exp_map_and_jacobian

Chart-based continuous 2-d sphere flow ("c")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.spheres.cnf_sphere_charts
    :members:
    :special-members: __init__
    :exclude-members: to_subspace, from_subspace, SphereProj, ODEfunc, TimeNetwork, AmbientProjNN, num_nn_funs

Interval flow layers
----------------------------

.. automodule:: jammy_flows.layers.intervals.interval_base
    :members:
    :special-members: __init__
    :exclude-members: flow_mapping, get_desired_init_parameters, init_params, inv_flow_mapping, obtain_layer_param_structure, transform_target_space

Identity layer ("z")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: jammy_flows.layers.intervals.interval_do_nothing
    :members:
    :special-members: __init__

Rational-quadratic spline flow ("r")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jammy_flows.layers.intervals.rational_quadratic_spline
    :members:
    :special-members: __init__

Simplex flow layers
----------------------------

.. automodule:: jammy_flows.layers.simplex.simplex_base
    :members:
    :special-members: __init__
    :exclude-members: flow_mapping, get_desired_init_parameters, init_params, inv_flow_mapping, obtain_layer_param_structure, base_simplex_to_non_uniform_box, gauss_to_non_uniform_box, non_uniform_box_to_base_simplex, transform_target_space

Iterative interval simplex flow ("w")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: jammy_flows.layers.simplex.inner_loop_simplex
    :members:
    :special-members: __init__

