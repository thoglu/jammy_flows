import os
import sys
import pylab
import numpy
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import copy
#import astropy.stats

from scipy.spatial.transform import Rotation as R


def _update_attached_visualization_bounds(subgridspec, visualization_bounds):
    """
    Attach visualization bounds to subgridspec or update existing bounds.
    """

    if(hasattr(subgridspec, "visualization_bounds")):
        new_bounds=[]

        for ind_b,cur_b in enumerate(subgridspec.visualization_bounds):
            new_b=(min([cur_b[0], visualization_bounds[ind_b][0]]), max([cur_b[1], visualization_bounds[ind_b][1]]))

            new_bounds.append(new_b)

        setattr(subgridspec, "visualization_bounds", new_bounds)

    else:
        setattr(subgridspec, "visualization_bounds", visualization_bounds)
    

def find_bins(trace, percentiles=[5.0,95.0], num_bins=50, use_outlier_binning=False):

    if(use_outlier_binning):
        perc=numpy.percentile(trace, percentiles)

        bins=list(numpy.linspace(perc[0], perc[1], num_bins-2))

        new_edges=[min(trace)-1e-5]+bins+[max(trace)+1e-5]

        new_edges=numpy.array(new_edges)

    else:

        new_edges=numpy.linspace(min(trace)-1e-5, max(trace)+1e-5, num_bins)

    return new_edges

def obtain_bins_and_visualization_regions(samples, model, percentiles=[5.0,95.0], relative_buffer=0.1, num_bins=50, s2_norm="standard", use_outlier_binning=False):
    
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

        if(pdf_def[0]=="e"):
            dim=int(pdf_def[1:])

            for sub_index in range(cur_index, cur_index+dim):

                np_samples=samples[:,sub_index].cpu().numpy()
                # vis bounds
                boundaries=numpy.percentile(np_samples, percentiles)

                relative_extra=relative_buffer*(boundaries[1]-boundaries[0])

                visualization_bounds.append((boundaries[0]-relative_extra, boundaries[1]+relative_extra))

                # density bounds
                min_val = float(min(samples[:, sub_index]).cpu())
                max_val = float(max(samples[:, sub_index]).cpu())

                relative_extra=0.2*(max_val-min_val)

                density_eval_bounds.append((min_val-relative_extra, max_val+relative_extra))

                # histo bins
                #edges = astropy.stats.bayesian_blocks(np_samples, p0=0.1, gamma=0.9)

                ## find bins based on percentiles
                edges=find_bins(np_samples, percentiles=percentiles, num_bins=num_bins)
                histogram_edges.append(edges)

            cur_index+=dim

        elif(pdf_def[0]=="s"):

            if(int(pdf_def[1])==1):

                visualization_bounds.append((0,2*numpy.pi))
                density_eval_bounds.append((0,2*numpy.pi))
                histogram_edges.append(numpy.linspace(0, 2*numpy.pi, num_bins))
                cur_index+=1

            elif(int(pdf_def[1])==2):

                if(s2_norm=="standard"):

                    visualization_bounds.append((0,numpy.pi))
                    visualization_bounds.append((0,2*numpy.pi))

                    density_eval_bounds.append((0,numpy.pi))
                    density_eval_bounds.append((0,2*numpy.pi))

                    histogram_edges.append(numpy.linspace(0, numpy.pi, num_bins))
                    histogram_edges.append(numpy.linspace(0, 2*numpy.pi, num_bins))

                else:

                    visualization_bounds.append((-2.0,2.0))
                    visualization_bounds.append((-2.0,2.0))

                    density_eval_bounds.append((0,numpy.pi))
                    density_eval_bounds.append((0,2*numpy.pi))

                    histogram_edges.append(numpy.linspace(-2.0,2.0, num_bins))
                    histogram_edges.append(numpy.linspace(-2.0,2.0, num_bins))

                cur_index+=2

        elif(pdf_def[0]=="i"):

            lower=model.layer_list[pdf_index][-1].low_boundary
            higher=model.layer_list[pdf_index][-1].high_boundary

            visualization_bounds.append((lower,higher))
        
            density_eval_bounds.append((lower, higher))
         
            histogram_edges.append(numpy.linspace(lower,higher, num_bins))
           
        elif(pdf_def[0]=="a"):

            lower=0.0
            higher=1.0

            dim=dim=int(pdf_def[1:])

            for dindex in range(dim):
                visualization_bounds.append((lower,higher))
            
                density_eval_bounds.append((lower, higher))
             
                histogram_edges.append(numpy.linspace(lower,higher, num_bins))

    return visualization_bounds, density_eval_bounds, histogram_edges


def calculate_contours(pdf_vals, bin_volumes, probs=[0.68, 0.95]):
    totsum = 0.0
    flattend_pdf = pdf_vals.flatten()
    #flattend_volumes = bin_volumes.flatten()

    sorta = numpy.argsort(flattend_pdf)[::-1]

    contour_values = []

    cur_prob_index = 0

    for ind, pdf_eval in enumerate(flattend_pdf[sorta]):
        totsum += pdf_eval*bin_volumes#flattend_volumes[sorta][ind]

        if (totsum > probs[cur_prob_index]):
            contour_values.append(pdf_eval)
            cur_prob_index += 1

        if (cur_prob_index >= len(probs)):
            break

    return contour_values


def get_bounds_from_contour(cres, boundary=0.1):

    cont_min_x = 9999999.9
    cont_max_x = -9999999.9

    cont_min_y = 999999999.9
    cont_max_y = -9999999999.9

    for i in cres.allsegs[0]:

        for j in i:

            if (j[0] < cont_min_x):
                cont_min_x = j[0]
            if (j[0] > cont_max_x):
                cont_max_x = j[0]

            if (j[1] < cont_min_y):
                cont_min_y = j[1]
            if (j[1] > cont_max_y):
                cont_max_y = j[1]

    return cont_min_x, cont_max_x, cont_min_y, cont_max_y


def get_density_eval_minmax_values(samples):
    """
    Obtain a simple estimate of pdf extent based on min/max values of samples.
    Used to determine a region for density evaluation (not plotting).
    """
    mins_maxs = []

    for ind in range(samples.shape[1]):
        min_val = float(min(samples[:, ind]).cpu())
        max_val = float(max(samples[:, ind]).cpu())

        mins_maxs.append((min_val, max_val))

    return mins_maxs


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

    if (conditional_input is not None):

        if(type(conditional_input)==list):
           
            cinput=[]
            for ci in conditional_input:
                cinput.append(ci.repeat(used_npts**len(mins_maxs), 1)[mask_inner])

            if(cinput[0].is_cuda):
                eval_positions=eval_positions.to(cinput[0])
        else:
            cinput = conditional_input.repeat(used_npts**len(mins_maxs), 1)[mask_inner]
            if(cinput.is_cuda):
                eval_positions=eval_positions.to(cinput)
   
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

    res = (-600.0)*torch.ones(len(torch_positions)).type_as(torch_positions).to(log_res)
    res[mask_inner] = log_res  #.exp()
   
    res = res.cpu().numpy()
    numpy_positions=eval_positions.cpu().numpy()
    if((numpy.isfinite(res)==False).sum()>0):
      print("Non-finite evaluation during PDF eval for plotting..")
      print((numpy.isfinite(res)==False).sum())
      print(numpy_positions[(numpy.isfinite(res)==False)])

      if(cinput is None):
        r,_,_=model(eval_positions[mask_inner][torch.isfinite(log_res)==False][:])
      else:
        r,_,_=model(eval_positions[mask_inner][torch.isfinite(log_res)==False][:], conditional_input=cinput[torch.isfinite(log_res)==False])
      print(r)
      raise Exception()

    #######################


    res.resize([used_npts] * len(mins_maxs))

    resized_torch_positions = torch_positions.cpu().numpy()
    resized_torch_positions.resize([used_npts] * len(mins_maxs) + [len(mins_maxs)])

    ## add in sin(theta) factors into density

    """

    for ind, sz in enumerate(sin_zen_mask):
      if(sz==1):
        slice_mask=(None,)*ind+(slice(None,None),)+(None,)*(len(sin_zen_mask)-1-ind)

        zen_vals=numpy.sin(numpy.linspace(mins_maxs[ind][0]+1e-4, mins_maxs[ind][1]-1e-4, npts))

        ## log val, adding zenith factors where needed
        res+=numpy.log(zen_vals[slice_mask])
    """

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


def show_sample_contours(ax,
                         samples,
                         bins=50,
                         color="white",
                         contour_probs=[0.68, 0.95],
                         sin_zen_mask=[0,0]):

    ## if bins is a list, make sure samples are within bounds by artifical intervention

    new_samples = samples
    if (type(bins) == list):

        bounds_x_mask = (samples[:, 0] < bins[0][-1]) & (samples[:, 0] >
                                                         bins[0][0])
        bounds_y_mask = (samples[:, 1] < bins[1][-1]) & (samples[:, 1] >
                                                         bins[1][0])

        total_included_mask = bounds_x_mask & bounds_y_mask

        if (total_included_mask.sum() < 100):
            print("TOTAL INCLUDED SUM ", total_included_mask.sum())
            print(
                "too few SAMPLES IN PLOTTING RANGE ... fake entries to evaluate fake contour"
            )
            somex = numpy.random.uniform(bins[0][0],
                                         bins[0][-1],
                                         size=len(samples))
            somey = numpy.random.uniform(bins[1][0],
                                         bins[1][-1],
                                         size=len(samples))

            new_samples = numpy.zeros_like(samples)
            new_samples[:, 0] = somex
            new_samples[:, 1] = somey


    bin_fillings, xedges, yedges = numpy.histogram2d(new_samples[:, 0],
                                                     new_samples[:, 1],
                                                     bins=bins,
                                                     density=True)
   
    xvals = 0.5 * (xedges[1:] + xedges[:-1])
    yvals = 0.5 * (yedges[1:] + yedges[:-1])
    bw = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

    bin_volumes=bw
    
    contour_values = calculate_contours(bin_fillings, bin_volumes, probs=contour_probs)
   
    ## reverse
    contour_values = contour_values[::-1]

    bounds=None

    try:
        ret = ax.contour(xvals,
                       yvals,
                       bin_fillings.T,
                       levels=contour_values,
                       colors=color)

        fmt_dict = dict()

        for ind, cprob in enumerate(contour_probs[::-1]):
          if(ind<len(contour_values)):
              fmt_dict[contour_values[ind]] = "%d" % (int(cprob * 100)) + r" %"

        ax.clabel(ret,
                fontsize=9,
                inline=1,
                fmt=fmt_dict,
                levels=contour_values,
                colors=color)


        bounds = get_bounds_from_contour(ret)
    
    except:
      print("Contour failed")
      return [[-1.0,1.0], [-1.0,1.0]]
    


    return [[bounds[0], bounds[1]], [bounds[2], bounds[3]]]


def plot_density_with_contours(ax,
                               log_evals,
                               evalpositions,
                               bin_volumes,
                               pts_per_dim,
                               color="black",
                               contour_probs=[0.68, 0.95]):
  

    xvals=evalpositions[:,0,0]
    yvals=evalpositions[0,:,1]
   
    #xr = (evalpositions[:, :, 0].min(), evalpositions[:, :, 0].max())
    #yr = (evalpositions[:, :, 1].min(), evalpositions[:, :, 1].max())

    xr = (xvals[0], xvals[-1])
    yr = (yvals[0], yvals[-1])

    contour_values = calculate_contours(numpy.exp(log_evals), bin_volumes,
                                        probs=contour_probs)

   
    ## reverse

    contour_values = contour_values[::-1]

    pcol_result = ax.pcolorfast(xr, yr, numpy.exp(log_evals).T)

    #contour_x = numpy.linspace(evalpositions[:, :, 0].min(),
    #                           evalpositions[:, :, 0].max(), pts_per_dim)
    #contour_y = numpy.linspace(evalpositions[:, :, 1].min(),
    #                           evalpositions[:, :, 1].max(), pts_per_dim)

    valid_contour=True

    if(len(contour_values)>=2):
      if(contour_values[0]==contour_values[1]):
        valid_contour=False

    if(len(contour_probs) != len(contour_values)):
      valid_contour=False

    if(valid_contour):
      contour_x=xvals
      contour_y=yvals

      bins = [contour_x, contour_y]

      res = ax.contour(contour_x,
                       contour_y,
                       numpy.exp(log_evals).T,
                       levels=contour_values,
                       colors=color)

      fmt_dict = dict()
  
      for ind, cprob in enumerate(contour_probs[::-1]):
          fmt_dict[contour_values[ind]] = "%d" % (int(cprob * 100)) + r" %"

      ax.clabel(res,
                fontsize=9,
                inline=1,
                fmt=fmt_dict,
                levels=contour_values,
                colors=color)

    pylab.colorbar(pcol_result, ax=ax)

def get_basic_gridlines():

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

def plot_joint_pdf(pdf,
                   fig,
                   gridspec,
                   samples,
                   subgridspec=None,
                   conditional_input=None,
                   bounds=None,
                   multiplot=False,
                   total_pdf_eval_pts=10000,
                   true_values=None,
                   plot_only_contours=False,
                   contour_probs=[0.68, 0.95],
                   contour_color="white",
                   autoscale=True,
                   skip_plotting_density=False,
                   hide_labels=False,
                   s2_norm="standard",
                   colormap=cm.rainbow,
                   s2_rotate_to_true_value=True,
                   s2_show_gridlines=True,
                   skip_plotting_samples=False,
                   var_names=[],
                   relative_buffer=0.1,
                   vis_percentiles=[3.0,97.0],
                   show_relative_std=0,
                   use_outlier_binning=False,
                   **kwargs):

    plot_density = False
    dim = len(samples[0])
    if (pdf.total_target_dim == 1 and dim == pdf.total_target_dim):
        plot_density = True

    if (pdf.total_target_dim == 2 and dim == pdf.total_target_dim):
        plot_density = True

    if (conditional_input is not None):
        if(type(conditional_input)==list):
            if (conditional_input[0].shape[0] > 1):
                plot_density = False
        else:
            if (conditional_input.shape[0] > 1):
                plot_density = False

    if (skip_plotting_density):
        plot_density = False

    ## 


    visualization_bounds, density_eval_bounds, histogram_edges=obtain_bins_and_visualization_regions(samples, pdf, percentiles=vis_percentiles, relative_buffer=relative_buffer, use_outlier_binning=use_outlier_binning, num_bins=50, s2_norm=s2_norm)
    
    if(bounds is not None):
        visualization_bounds=bounds

    ## true positions are typically labels
    plotted_true_values=None
    if(true_values is not None):
      plotted_true_values=copy.deepcopy(true_values)

    totalpts = total_pdf_eval_pts
    pts_per_dim = int(totalpts**(1.0 / float(dim)))

    samples = samples.cpu().clone()

    gridline_dict=None
    if(s2_show_gridlines and "s2" in pdf.pdf_defs_list):
      gridline_dict=dict()
      for ind, pdf_type in enumerate(pdf.pdf_defs_list):
        if(pdf_type=="s2"):
          gridline_dict[(pdf.target_dim_indices[ind][0], pdf.target_dim_indices[ind][1])]=get_basic_gridlines()


    ## transform samples to lambert space if necessary
    for ind, pdf_type in enumerate(pdf.pdf_defs_list):
        if (pdf_type == "s2" and s2_norm=="lambert"):
            ## transform to labmert space

            ## calculate fix point if rotation for visualtizion is desired
            fix_point=None
            

            if(s2_rotate_to_true_value and true_values is not None):
             
              fix_point=true_values[pdf.target_dim_indices[ind][0]:pdf.target_dim_indices[ind][1]]

            ## transform samples to lambert space
            samples[:,
                    pdf.target_dim_indices[ind][0]:pdf.target_dim_indices[ind]
                    [1]] = spherical_to_cartesian_lambert(
                        samples[:, pdf.target_dim_indices[ind][0]:pdf.
                                target_dim_indices[ind][1]], fix_point=fix_point)

            ## transform true value to lambert space
            if(plotted_true_values is not None):
              
                res=spherical_to_cartesian_lambert(true_values[pdf.target_dim_indices[ind][0]:pdf.
                target_dim_indices[ind][1]].unsqueeze(0), fix_point=fix_point)

                plotted_true_values[pdf.target_dim_indices[ind][0]:pdf.
                target_dim_indices[ind][1]]=res.squeeze(0)

          
            ## transform gridlines to lambert space
            if(s2_show_gridlines):
              tup=(pdf.target_dim_indices[ind][0],pdf.target_dim_indices[ind][1])

              new_list=[]

              for l in gridline_dict[tup]:
                new_list.append(spherical_to_cartesian_lambert(l, fix_point=fix_point))

              gridline_dict[tup]=new_list



    samples = samples.cpu().numpy()

    pdf_conditional_input = conditional_input

    if (pdf_conditional_input is not None):

        if(type(pdf_conditional_input)==list):
            pdf_conditional_input=[ci[0:1] for ci in pdf_conditional_input]
        else:
            pdf_conditional_input = pdf_conditional_input[0:1]

    evalpositions, log_evals, bin_volumes, sin_zen_mask, unreliable_spherical_regions= get_pdf_on_grid(
        density_eval_bounds,
        pts_per_dim,
        pdf,
        conditional_input=pdf_conditional_input,
        s2_norm=s2_norm,
        s2_rotate_to_true_value=s2_rotate_to_true_value,
        true_values=true_values)
   
    total_pdf_integral=numpy.exp(log_evals).sum()*bin_volumes
    

    if (dim == 1):

        if (subgridspec is None):
            
            subgridspec = gridspec.subgridspec(1, 1)
            ax = fig.add_subplot(subgridspec[0, 0])
            setattr(subgridspec, "axdict", {"ax": ax})
            
            ## find ax in existing gridspec
            """
            for ax in fig.get_axes():
                ax_geometry=ax.get_subplotspec().get_geometry()

                if(subgridspec!=ax.get_subplotspec().get_gridspec() ):
                    continue

                if(ax_geometry[0]==1 and ax_geometry[1]==1):
                    break
            """
        ax=subgridspec.axdict["ax"]


        ## attach/update "visualization_bounds" to subgridspec
        _update_attached_visualization_bounds(subgridspec, visualization_bounds)
        
        ax.hist(samples[:, 0], bins=histogram_edges[0], density=True)

        if (plot_density):

            ax.plot(evalpositions[:, 0], numpy.exp(log_evals), color="k")

        if (true_values is not None):
            ax.axvline(true_values[0].cpu().numpy(), color="red", lw=2.0)

        if (hide_labels):
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        ax.set_xlim(subgridspec.visualization_bounds[0][0], subgridspec.visualization_bounds[0][1])

    elif (dim == 2 and multiplot == False):
     
        if (subgridspec is None):
            subgridspec = gridspec.subgridspec(1, 1)
            ax = fig.add_subplot(subgridspec[0, 0])
            setattr(subgridspec, "axdict", {"ax": ax})
        
        ax=subgridspec.axdict["ax"]
  

        _update_attached_visualization_bounds(subgridspec, visualization_bounds)
        
        if(plot_density):
            plot_density_with_contours(ax, log_evals, evalpositions,
                                     bin_volumes, pts_per_dim)

        ## plot a histogram density from samples

        if ( (plot_only_contours == False) and (plot_density == False) and (skip_plotting_samples==False)):
           
            ax.hist2d(samples[:, 0],
                      samples[:, 1],
                      bins=histogram_edges,
                      density=True)
        
        if (contour_probs != [] and skip_plotting_samples==False):
            
            _ = show_sample_contours(ax,
                                     samples,
                                     bins=histogram_edges,
                                     color=contour_color,
                                     contour_probs=contour_probs,
                                     sin_zen_mask=sin_zen_mask)


        ## mark poles
        if(len(unreliable_spherical_regions)>0):
          
          ax.plot(unreliable_spherical_regions[:,0], unreliable_spherical_regions[:,1], color="orange", marker="x", lw=0.0)

        ## plot true values
        if (plotted_true_values is not None):
            
            ax.plot([plotted_true_values[0].cpu().numpy()], [plotted_true_values[1].cpu().numpy()],
                    color="red",
                    marker="o",
                    ms=3.0)

        ## plot gridlines if desired
        if(s2_show_gridlines and gridline_dict is not None):
          
          for gl in gridline_dict[(0,2)]:
            np_gl=gl.cpu().numpy()

            ax.plot(np_gl.T[0], np_gl.T[1], color="gray", alpha=0.5)
       
        ax.set_xlim(subgridspec.visualization_bounds[0][0], subgridspec.visualization_bounds[0][1])
        ax.set_ylim(subgridspec.visualization_bounds[1][0], subgridspec.visualization_bounds[1][1]) 
    
        if (hide_labels):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
       
    else:

        ###########
        if (subgridspec is None):
            
            subgridspec = gridspec.subgridspec(dim, dim)
        
            setattr(subgridspec, "axdict", {})

            for ind1 in range(dim):
                for ind2 in range(dim):
                    if(ind2>ind1):
                        continue

                    subgridspec.axdict[(ind1,ind2)]=fig.add_subplot(subgridspec[ind1, ind2])
                    ## make sure background looks similar to histogram empty bins
                    
        ## attach/update "visualization_bounds" to subgridspec
        _update_attached_visualization_bounds(subgridspec, visualization_bounds)
        
        for ind1 in range(dim):
            for ind2 in range(dim):
                      
                if (ind2 < ind1):

                    """
                    found_ax=False
                    for ax in fig.get_axes():

                        ax_geometry=ax.get_subplotspec().get_geometry()
                        print("checking ... ", ax_geometry)
                        num_rows=ax_geometry[0]
                        this_gridspec=ax.get_gridspec()

                        if(subgridspec!=ax.get_subplotspec().get_gridspec() ):
                            continue
                      
                        if(ax_geometry[2]==(ind1*num_rows+ind2+1) and (num_rows==dim)):
                            found_ax=True
                           
                            break


                    if(found_ax==False):
                        print("have not found in off diagonal ", ind1, ind2)
                        ax = fig.add_subplot(subgridspec[ind1, ind2])
                        print("added AX ", ax.get_subplotspec().get_geometry())
                    """

                    ax=subgridspec.axdict[(ind1,ind2)]

                    if (plot_only_contours == False):
                        print("plot histo")
                        ax.hist2d(samples[:, ind2],
                                  samples[:, ind1],
                                  bins=[histogram_edges[ind2], histogram_edges[ind1]],
                                  density=True,
                                  cmap=colormap,
                                  cmin=1e-20)

                    if (plotted_true_values is not None):
                        ax.plot([plotted_true_values[ind2].cpu().numpy()], [plotted_true_values[ind1].cpu().numpy()],
                                color="red",
                                marker="o",
                                ms=3.0)

                    new_samples = numpy.concatenate(
                        [samples[:, ind2:ind2 + 1], samples[:, ind1:ind1 + 1]],
                        axis=1)
                    print("IND 2", ind2, "ind 1 ", ind1)
                    if (contour_probs != []):
                        _ = show_sample_contours(
                            ax,
                            new_samples,
                            bins=[histogram_edges[ind2], histogram_edges[ind1]],
                            color=contour_color,
                            contour_probs=contour_probs)
                    
                    ax.set_xlim(subgridspec.visualization_bounds[ind2][0], subgridspec.visualization_bounds[ind2][1])
                    ax.set_ylim(subgridspec.visualization_bounds[ind1][0], subgridspec.visualization_bounds[ind1][1])

                    ## always hide labels if left or bottom 
                    if(ind2==0 and ind1<(dim-1)):
                        ax.set_xticklabels([])

                        fontsize = ax.get_window_extent().height/5.0
                        #for lab in ax.get_yticklabels():
                        #    lab.set_fontsize(fontsize)

                        if("labels" in kwargs):
                            assert(len(kwargs["labels"])==dim)

                            ax.set_ylabel(kwargs["labels"][ind1])
                       
                    elif((ind1==(dim-1)) and (ind2!=0)):
                        ax.set_yticklabels([])

                        fontsize = ax.get_window_extent().width/5.0
                        for lab in ax.get_xticklabels():
                            #lab.set_fontsize(fontsize)
                            lab.set_rotation(45)

                        if("labels" in kwargs):
                            assert(len(kwargs["labels"])==dim)

                            ax.set_xlabel(kwargs["labels"][ind2])


                    elif(ind1<(dim-1) and ind2>0):
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])

                    
                    else:   
                        ## lower left corner
                        fontsize = ax.get_window_extent().width/5.0
                        for lab in ax.get_xticklabels():
                            #lab.set_fontsize(fontsize)
                            lab.set_rotation(45)

                        fontsize = ax.get_window_extent().height/5.0
                        #for lab in ax.get_yticklabels():
                        #    lab.set_fontsize(fontsize)

                        if("labels" in kwargs):
                            assert(len(kwargs["labels"])==dim)

                            ax.set_xlabel(kwargs["labels"][ind2])
                            ax.set_ylabel(kwargs["labels"][ind1])

                    if (hide_labels):
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])

                elif (ind2 == ind1):

                    ## looking for ax
                    """
                    found_ax=False
                    for ax in fig.get_axes():


                            
                        ax_geometry=ax.get_subplotspec().get_geometry()
                        num_rows=ax_geometry[0]
                        
                        if(subgridspec!=ax.get_subplotspec().get_gridspec() ):
                            continue


                        
                        if(ax_geometry[2]==(ind1*num_rows+ind2+1) and (num_rows==dim)):
                            found_ax=True
                           
                            break


                    if(found_ax==False):
                        print("have not found hte ax ....", ax, ind2, ind1)
                        ax = fig.add_subplot(subgridspec[ind1, ind2])

                        print("added AX ", ax.get_subplotspec().get_geometry())
                    """

                    ##############

                    ax=subgridspec.axdict[(ind1,ind2)]

                    color="black"
                    if("color_1d" in kwargs.keys()):
                        color=kwargs["color_1d"]
                    ax.hist(samples[:, ind1], bins=histogram_edges[ind1], histtype="step", density=True, color=color)

                    if(show_relative_std):
                        std=numpy.std(samples[:, ind1])
                        relative_wrt_axwidth=0.5*(visualization_bounds[ind2][1]-visualization_bounds[ind2][0])/std

                        ax.set_title("%.1f" % (relative_wrt_axwidth),fontsize=ax.get_window_extent().width/4.5)

                    if (plotted_true_values is not None):
                        ax.axvline(plotted_true_values[ind1].cpu().numpy(), color="red", lw=2.0)
                
                   
                    ax.set_xlim(subgridspec.visualization_bounds[ind2][0], subgridspec.visualization_bounds[ind2][1])

                    # 1-d hists do not need y labels
                    ax.set_yticklabels([])
                   
                    if (hide_labels or ind2 < (dim-1)):
                        
                        ax.set_xticklabels([])
                    else:
                        fontsize = ax.get_window_extent().width/5.0
                        for lab in ax.get_xticklabels():
                            #lab.set_fontsize(fontsize)
                            lab.set_rotation(45)

                        if("labels" in kwargs):
                            assert(len(kwargs["labels"])==dim)

                            ax.set_xlabel(kwargs["labels"][ind2])
                          

    
    return subgridspec, total_pdf_integral


def visualize_pdf(pdf,
                  fig,
                  gridspec=None,
                  subgridspec=None,
                  conditional_input=None,
                  nsamples=10000,
                  total_pdf_eval_pts=10000,
                  bounds=None,
                  true_values=None,
                  plot_only_contours=False,
                  contour_probs=[0.68, 0.95],
                  contour_color="white",
                  autoscale=True,
                  seed=None,
                  skip_plotting_density=False,
                  hide_labels=False,
                  s2_norm="standard",
                  colormap=cm.rainbow,
                  s2_rotate_to_true_value=True,
                  s2_show_gridlines=True,
                  skip_plotting_samples=False,
                  var_names=[],
                  num_iterative_steps=-1,
                  relative_vis_buffer=0.1,
                  vis_percentiles=[3.0, 97.0],
                  show_relative_std=0,
                  use_outlier_binning=False,
                  **kwargs
                  ):
   
    with torch.no_grad():
      sample_conditional_input = conditional_input
      if (conditional_input is not None):

        if(type(conditional_input)==list):

            sample_conditional_input=[]

            for inp_ind, ci in enumerate(conditional_input):

                this_ci=ci

                if (len(this_ci.shape) == 1):
                    this_ci=this_ci.unsqueeze(0)

                if (this_ci.shape[0] == 1):
                    this_ci = this_ci.repeat(nsamples, 1)



                sample_conditional_input.append(this_ci)

            ci_batch_size=sample_conditional_input[0].shape[0]

        else:

            if (len(conditional_input.shape) == 1):
                sample_conditional_input = sample_conditional_input.unsqueeze(0)

            if (sample_conditional_input.shape[0] == 1):
                sample_conditional_input = sample_conditional_input.repeat(nsamples, 1)

            ci_batch_size=sample_conditional_input.shape[0]

      if (gridspec is None):
          gridspec = fig.add_gridspec(1, 1)[0, 0]

      if(sample_conditional_input is not None):

        if(num_iterative_steps>0):
            
            assert(ci_batch_size%num_iterative_steps==0), "Number of total sample points must be divisible by iterative steps!"
            
            num_per_step=int(ci_batch_size/num_iterative_steps)
            samples=[]

            for cur_step in range(num_iterative_steps):
                
                if(type(sample_conditional_input)==list):
                    this_sample_input=[si[cur_step*num_per_step:cur_step*num_per_step+num_per_step] for si in sample_conditional_input]

                    cur_samples, _, _, _ = pdf.sample(
                      conditional_input=this_sample_input,
                      seed=seed+cur_step if seed is not None else None,
                      force_intrinsic_coordinates=True)
                else:
                    cur_samples, _, _, _ = pdf.sample(
                      conditional_input=sample_conditional_input[cur_step*num_per_step:cur_step*num_per_step+num_per_step],
                      seed=seed+cur_step if seed is not None else None,
                      force_intrinsic_coordinates=True)

                samples.append(cur_samples)

            samples=torch.cat(samples, dim=0)

        else:

            samples, _, _, _ = pdf.sample(
              conditional_input=sample_conditional_input,
              seed=seed,
              force_intrinsic_coordinates=True)
      else:

        # check if PDF involves only a "x" flow (which does not nothing.. enforce device then)
        used_dtype, used_device=pdf.obtain_current_dtype_n_device()

        if(used_dtype is None):
            if("dtype" in kwargs):
                used_dtype=kwargs["dtype"]

        if(used_device is None):
            if("device" in kwargs):
                used_device=kwargs["device"]

        
        if(num_iterative_steps>0):
            assert(nsamples%num_iterative_steps==0), "Number of total sample points must be divisible by iterative steps!"
            
            num_per_step=int(nsamples/num_iterative_steps)
            samples=[]

            for cur_step in range(num_iterative_steps):

                cur_samples, _, _, _ = pdf.sample(
                  samplesize=num_per_step,
                  seed=seed+cur_step if seed is not None else None, device=used_device, dtype=used_dtype, force_intrinsic_coordinates=True)


                samples.append(cur_samples)

            samples=torch.cat(samples, dim=0)

        else:

          samples, samples_base, evals, evals_base = pdf.sample(
              samplesize=nsamples,
              seed=seed,
              force_intrinsic_coordinates=True,
              device=used_device,
              dtype=used_dtype)


      higher_dim_spheres = False

      new_subgridspec, total_pdf_integral = plot_joint_pdf(
          pdf,
          fig,
          gridspec,
          samples,
          subgridspec=subgridspec,
          conditional_input=conditional_input,
          bounds=bounds,
          multiplot=False,
          total_pdf_eval_pts=total_pdf_eval_pts,
          true_values=true_values,
          plot_only_contours=plot_only_contours,
          contour_probs=contour_probs,
          contour_color=contour_color,
          autoscale=autoscale,
          skip_plotting_density=skip_plotting_density,
          hide_labels=hide_labels,
          s2_norm=s2_norm,
          colormap=colormap,
          s2_rotate_to_true_value=s2_rotate_to_true_value,
          s2_show_gridlines=s2_show_gridlines,
          skip_plotting_samples=skip_plotting_samples,
          var_names=var_names,
          relative_buffer=relative_vis_buffer,
          vis_percentiles=vis_percentiles,
          show_relative_std=show_relative_std,
          use_outlier_binning=use_outlier_binning,
          **kwargs)
        
    
    return samples, new_subgridspec, total_pdf_integral
