import numpy
import torch

def find_bins(trace, percentiles=[5.0,95.0], num_bins=50, use_outlier_binning=False):

    if(use_outlier_binning):
        perc=numpy.percentile(trace, percentiles)

        bins=list(numpy.linspace(perc[0], perc[1], num_bins-2))

        new_edges=[min(trace)-1e-5]+bins+[max(trace)+1e-5]

        new_edges=numpy.array(new_edges)

    else:

        new_edges=numpy.linspace(min(trace)-1e-5, max(trace)+1e-5, num_bins)

    return new_edges


def _update_bounds(old_bounds, allowed_min=None, allowed_max=None):

    new_bounds=[old_bounds[0] if allowed_min is None else max(old_bounds[0], allowed_min), old_bounds[1] if allowed_max is None else min(old_bounds[1], allowed_max)]

    return new_bounds

def obtain_bins_and_visualization_regions(samples, model, percentiles=[3.0,97.0], relative_buffer=0.1, num_bins=50, s2_norm="standard", use_outlier_binning=False):
    
    """
    Uses samples and pdf defs to calculate binning and visualization regions.

    Euclidean:
    Obtain an uneven binning based on Bayesian Blocks and a region that makes sense for visualization based on 1-d percentiles. 
    Adds a relative buffer to both sides. 
    
    Other:
    Non-Euclidean manifolds have fixed bounds for binning and visualization, so this is easy.
    """

    cur_index=0

    visualization_bounds=[]
    density_eval_bounds=[]
    histogram_edges=[]

    for pdf_index, pdf_def in enumerate(model.pdf_defs_list):

        dim=int(pdf_def[1:].split("_")[0])

        for sub_index in range(cur_index, cur_index+dim):

            np_samples=samples[:,sub_index].cpu().numpy()
          
            boundaries=numpy.percentile(np_samples, percentiles)

            relative_extra=relative_buffer*(boundaries[1]-boundaries[0])

            visualization_bounds.append((boundaries[0]-relative_extra, boundaries[1]+relative_extra))
            density_eval_bounds.append(visualization_bounds[-1])
            edges=find_bins(np_samples, percentiles=percentiles, num_bins=num_bins)

            if(pdf_def[0]=="e"):
                cur_allowed_min=None
                cur_allowed_max=None

            elif(pdf_def[0]=="s"):

                
                if(dim==1):
                    cur_allowed_min=0.0
                    cur_allowed_max=2*numpy.pi
                    
                else:
                    if(s2_norm=="standard"):

                        cur_allowed_min=0.0
                        if(sub_index==0):
                            ## zenith
                            cur_allowed_max=numpy.pi
                        else:
                            ## azimuth
                            cur_allowed_max=2*numpy.pi
                    else:
                        cur_allowed_min=-2.0
                        cur_allowed_max=2.0

            elif(pdf_def[0]=="i"):
                cur_allowed_min=model.layer_list[pdf_index][-1].low_boundary
                cur_allowed_max=model.layer_list[pdf_index][-1].high_boundary
            elif(pdf_def[0]=="a"):
                cur_allowed_min=0.0
                cur_allowed_max=1.0
           
            visualization_bounds[-1]=_update_bounds(visualization_bounds[-1], allowed_min=cur_allowed_min, allowed_max=cur_allowed_max)
            density_eval_bounds[-1]=_update_bounds(density_eval_bounds[-1], allowed_min=cur_allowed_min, allowed_max=cur_allowed_max)
            
            histogram_edges.append(numpy.linspace(edges[0] if cur_allowed_min is None else max(edges[0],cur_allowed_min), edges[-1] if cur_allowed_max is None else min(edges[-1],cur_allowed_max), num_bins))

        cur_index+=dim

    return visualization_bounds, density_eval_bounds, histogram_edges

def get_pdf_on_grid(mins_maxs, npts, model, conditional_input=None, s2_norm="standard", s2_rotate_to_true_value=False, true_values=None):

    side_vals = []

    bin_volumes = 1.0#numpy.ones([npts]*len(mins_maxs))
    glob_ind = 0
    #has_high_dim_spheres = False
    cinput = None

    sin_zen_mask=[]

    used_npts=npts
    if(npts<2):
        used_npts=2
  
    for pdf_index,pdf in enumerate(model.pdf_defs_list):
        this_sub_dim = int(pdf[1])
        if (pdf == "s2" and s2_norm=="lambert"):
            #has_high_dim_spheres = True
            side_vals.append(numpy.linspace(-2, 2, used_npts))
            bin_volumes *= (side_vals[-1][1] - side_vals[-1][0])

            side_vals.append(numpy.linspace(-2, 2, used_npts))
            bin_volumes *= (side_vals[-1][1] - side_vals[-1][0])

            sin_zen_mask.extend([0,0])

        elif(pdf=="s2" and s2_norm=="standard"):

          sin_zen_mask.extend([0,0])
          
          zen_vals=numpy.linspace(mins_maxs[glob_ind][0]+1e-4, mins_maxs[glob_ind][1]-1e-4, used_npts)
          side_vals.append(zen_vals)
          bin_volumes*=(side_vals[-1][1] - side_vals[-1][0])

          side_vals.append(numpy.linspace(1e-4, 2*numpy.pi-1e-4, used_npts))
          bin_volumes *= (side_vals[-1][1] - side_vals[-1][0])

        elif(pdf=="s2"):
          raise Exception("s2_norm ", s2_norm, " unknown .")
        elif("i1" in pdf):
            
            side_vals.append(numpy.linspace(model.layer_list[pdf_index][-1].low_boundary+1e-5, model.layer_list[pdf_index][-1].high_boundary-1e-5, used_npts))

            bin_volumes *= (side_vals[-1][1] - side_vals[-1][0])
            sin_zen_mask.append(0)
        else:
            
            for ind, mm in enumerate(mins_maxs[glob_ind:glob_ind +
                                               this_sub_dim]):

                side_vals.append(numpy.linspace(mm[0], mm[1], used_npts))
                bin_volumes *= (side_vals[-1][1] - side_vals[-1][0])

                sin_zen_mask.append(0)

        glob_ind += this_sub_dim
   
    eval_positions = numpy.meshgrid(*side_vals)

    torch_positions = torch.from_numpy(
        numpy.resize(
            numpy.array(eval_positions).T,
            (used_npts**len(mins_maxs), len(mins_maxs))))

    eval_positions = torch_positions.clone()

    mask_inner = torch.ones(len(torch_positions)) == 1

    ## check s2 or simplex visualization
    for ind, pdf_def in enumerate(model.pdf_defs_list):
       
        if (pdf_def == "s2" and s2_norm=="lambert"):

            fix_point=None

            if(s2_rotate_to_true_value and true_values is not None):
              fix_point=true_values[model.target_dim_indices_intrinsic[ind][0]:model.target_dim_indices_intrinsic[ind][1]]
           
            mask_inner = mask_inner & (torch.sqrt(
                (eval_positions[:, model.target_dim_indices_intrinsic[ind][0]:model.
                                target_dim_indices_intrinsic[ind][1]]**2).sum(axis=1)) <
                                       2)
            ## transform s2 subdimensions from equal-area lambert dimension to real spherical dimensiosn the model can use

            eval_positions[:, model.target_dim_indices_intrinsic[ind][0]:model.
                           target_dim_indices_intrinsic[ind]
                           [1]] = cartesian_lambert_to_spherical(
                               eval_positions[:, model.
                                              target_dim_indices_intrinsic[ind][0]:model.
                                              target_dim_indices_intrinsic[ind][1]], fix_point=fix_point)

            # need some extra care, it seems sometimes nans can appear in the trafo step (only happened on GPU?)
            mask_inner=torch.isfinite(eval_positions[:,0]) & mask_inner
            mask_inner=torch.isfinite(eval_positions[:,1]) & mask_inner

        elif("c" in pdf_def):
           
            ## simplex .. mask everything outside allowed region
            mask_inner=mask_inner & (eval_positions[:, model.target_dim_indices_intrinsic[ind][0]:model.target_dim_indices_intrinsic[ind][1] ].sum(axis=1) < 1.0)

    batch_size=1
    if (conditional_input is not None):

        if(type(conditional_input)==list):
            batch_size=conditional_input[0].shape[0]
            mask_inner=mask_inner.repeat_interleave(batch_size, dim=0)
            cinput=[]
            for ci in conditional_input:
                cinput.append(ci.repeat_interleave(used_npts**len(mins_maxs), dim=0)[mask_inner])

            if(cinput[0].is_cuda):
                eval_positions=eval_positions.to(cinput[0]).repeat_interleave(batch_size, dim=0)
        else:
            batch_size=conditional_input.shape[0]
            mask_inner=mask_inner.repeat_interleave(batch_size, dim=0)

            cinput = conditional_input.repeat_interleave(used_npts**len(mins_maxs), dim=0)[mask_inner]
            if(cinput.is_cuda):
                eval_positions=eval_positions.to(cinput).repeat_interleave(batch_size, dim=0)
    
    log_res, _, _ = model(eval_positions[mask_inner], conditional_input=cinput, force_intrinsic_coordinates=True)

    ## update s2+lambert visualizations by adding sin(theta) factors to get proper normalization
    for ind, pdf_def in enumerate(model.pdf_defs_list):
        if (pdf_def == "s2" and s2_norm=="lambert"):
            ## first coordinate is theta currently
           
            upd=torch.log(torch.sin(eval_positions[mask_inner][:,model.target_dim_indices_intrinsic[ind][0]:model.target_dim_indices_intrinsic[ind][0]+1])).sum(axis=-1)
            
            ## angle -> cartesian -> subtract
            log_res-=upd

        
    ## no conditional input and only s2 pdf .. mask bad regions
    flagged_coords=numpy.array([])
    if(conditional_input is None and model.pdf_defs_list[0]=="s2"):
      
      
      problematic_pars=model.layer_list[0][0].return_problematic_pars_between_hh_and_intrinsic(eval_positions[mask_inner], flag_pole_distance=0.02)

      if(problematic_pars.shape[0]>0):
        if(s2_norm=="lambert"):
          fix_point=None
          if(s2_rotate_to_true_value and true_values is not None):
              fix_point=true_values[model.target_dim_indices[ind][0]:model.target_dim_indices[ind][1]]
          problematic_pars=spherical_to_cartesian_lambert(problematic_pars, fix_point=fix_point)
      flagged_coords=problematic_pars.cpu().numpy()

    res = (-600.0)*torch.ones(len(eval_positions)).type_as(log_res).to(log_res)
   
    res[mask_inner] = log_res  #.exp()

    res = res.cpu().numpy()
    
    if((numpy.isfinite(res)==False).sum()>0):

        numpy_positions=eval_positions.cpu().numpy()
        print("Non-finite evaluation during PDF eval for plotting..")
        print((numpy.isfinite(res)==False).sum())
        print(numpy_positions[(numpy.isfinite(res)==False)])

        if(cinput is None):
            r,_,_=model(eval_positions[mask_inner][torch.isfinite(log_res)==False][:])
        else:
            r,_,_=model(eval_positions[mask_inner][torch.isfinite(log_res)==False][:], conditional_input=cinput[torch.isfinite(log_res)==False])
        print(r)
        raise Exception()

    has_bad_regions=len(mask_inner)!=mask_inner.sum()

    res=res.reshape(*([batch_size]+[used_npts] * len(mins_maxs)))

    resized_torch_positions = torch_positions.cpu().numpy()
    resized_torch_positions=resized_torch_positions.reshape(*([batch_size]+[used_npts] * len(mins_maxs) + [len(mins_maxs)]))
    
    return resized_torch_positions, res, bin_volumes, sin_zen_mask, flagged_coords

def rotate_coords_to(theta, phi, target, reverse=False):

  target_theta=target[0].cpu().numpy()
  target_phi=target[1].cpu().numpy()

  phi=phi.cpu().numpy()
  theta=theta.cpu().numpy()

  x=numpy.cos(target_phi)*numpy.sin(target_theta)
  y=numpy.sin(target_phi)*numpy.sin(target_theta)
  z=numpy.cos(target_theta)

  ###########

  axis=-numpy.cross(numpy.array([x,y,z]), numpy.array([0,0,1]))
  axis_len=numpy.sqrt((axis**2).sum())
  axis/=axis_len

  rot_angle=numpy.pi-target_theta
  if(reverse):
    rot_angle=-rot_angle


  axis*=rot_angle.item()

  rot_matrix = R.from_rotvec(axis)
  ###########
  
  x=numpy.cos(phi)*numpy.sin(theta)
  y=numpy.sin(phi)*numpy.sin(theta)
  z=numpy.cos(theta)

  vals=numpy.concatenate([x[:,None], y[:,None],z[:,None]], axis=1)

  res=torch.from_numpy(rot_matrix.apply(vals))
  
  ##########

  theta=numpy.arccos(res[:,2])
  non_finite_mask=numpy.isfinite(theta)==False
  larger=non_finite_mask & (res[:,2] > 0)
  smaller=non_finite_mask & (res[:,2] < 0)

  theta[smaller]=numpy.pi
  theta[larger]=0.0


  phi=numpy.arctan2(res[:,1],res[:,0])

  #phi_smaller_mask=phi<0
  #phi[phi_smaller_mask]=phi[phi_smaller_mask]+2*numpy.pi
 
  return theta.to(target), phi.to(target)


def cartesian_lambert_to_spherical(xl, fix_point=None):

    ## first go to spherical lambert

    r = torch.sqrt((xl**2).sum(axis=1))
    phi = torch.acos(xl[:, 0] / r)
    larger_mask = (xl[:, 1] >= 0)

    phi = larger_mask * phi + (larger_mask == 0) * (2 * numpy.pi - phi)
    theta = 2 * torch.acos(r / 2.0)

    if(fix_point is not None):

      theta, phi = rotate_coords_to(theta, phi, fix_point, reverse=True)

    ## now go to spherical real coordinates

    return torch.cat([theta[:, None], phi[:, None]], dim=1)


def spherical_to_cartesian_lambert(spherical, fix_point=None):

    #####################

    theta = spherical[:, 0]
    phi_lambert = spherical[:, 1]

    ######################
    if(fix_point is not None):
      theta, phi_lambert = rotate_coords_to(theta, phi_lambert, fix_point)

  
    r_lambert = 2 * torch.cos(theta / 2.0)

    x_l = r_lambert * torch.cos(phi_lambert)
    y_l = r_lambert * torch.sin(phi_lambert)

    return torch.cat([x_l[:, None], y_l[:, None]], dim=1)


def get_basic_gridlines():
    """
    Some gridlines to indicate whats going on.
    """
    n_theta=5
    n_phi=10

    gridlines=[]

    for g in numpy.linspace(0.1,numpy.pi-0.1, n_theta):
        azis=torch.linspace(0,2*numpy.pi, 100)
        zens=torch.ones_like(azis)*g
        gl=torch.cat( [zens[:,None], azis[:,None]], dim=1)
        gridlines.append(gl)

    for a in numpy.linspace(0,2*numpy.pi-2*numpy.pi/n_phi, n_phi):
        zens=torch.linspace(0,numpy.pi,100)
        azis=torch.ones_like(zens)*a
        gl=torch.cat( [zens[:,None], azis[:,None]], dim=1)
        gridlines.append(gl)

    return gridlines