import numpy

try:
    import meander
except:
    print("Meander not installed... spherical contours can not be calculated.")
    meander=None

try:
    import healpy
except:
    print("Healpy not installed... spherical contours can not be calculated.")
    healpy=None


def find_closest(s, all_xyz_contours, contor_probs_all_cov):
    
    cur_min_index=-1
    overall_min=9999999999999999999999

    np_s=s.cpu().detach().numpy()

    for ind in range(len(all_xyz_contours)):

        min_dist=min(numpy.sqrt( numpy.sum((np_s-all_xyz_contours[ind])**2, axis=1)))

        if(min_dist<overall_min):
           
            overall_min=min_dist
            cur_min_index=ind

    return contor_probs_all_cov[cur_min_index]

def get_real_coverage_value(true_pos, xy_contours_for_coverage, actual_expected_coverage):

    
    all_joined_contours=[]
    for ind in range(len(xy_contours_for_coverage)):

        joint=numpy.concatenate(xy_contours_for_coverage[ind], axis=0)
        all_joined_contours.append(joint)

    ## find closest contour to a point s
    cb=_find_closest(true_pos, all_joined_contours, actual_expected_coverage)

    ## return contour probability
    return cb

def compute_spherical_contours(proportions, pdf_evals_with_area, pdf_evals):
    ''' Compute spherical contours using the meander package.

        Parameters:
        -----------
        proportions: list
            list of containment level to make contours for.
            E.g [0.68,0.9]
        samples: array
            array of values read in from healpix map
            E.g samples = hp.read_map(file)
        Returns:
        --------
        theta_list: list
            List of arrays containing theta values for desired contours
        phi_list: list
            List of arrays containing phi values for desired contours
    '''

    assert(healpy is not None), "Spherical contour calculation requires healpy!"
    assert(meander is not None), "Spherical contour calculation requires meander!"

    levels = []

    """
    flattend_pdf = pdf_evals.flatten()
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
    
    """
    inv_sorted=numpy.argsort(pdf_evals_with_area)[::-1]
    

    sorted_pdf_with_area = pdf_evals_with_area[inv_sorted] #list(reversed(list(sorted(samples))))
    sorted_pdf=pdf_evals[inv_sorted]

    nside = healpy.pixelfunc.get_nside(sorted_pdf_with_area)
    sample_points = numpy.array(healpy.pix2ang(nside,numpy.arange(len(sorted_pdf_with_area)))).T

    for proportion in proportions:
        level_index = (numpy.cumsum(sorted_pdf_with_area) > proportion).tolist().index(True)

        level = (sorted_pdf[level_index] + (sorted_pdf[level_index+1] if level_index+1 < len(sorted_pdf_with_area) else 0)) / 2.0
        levels.append(level)

    
   
    contours_by_level = meander.spherical_contours(sample_points, pdf_evals, levels)

    
    theta_list = []
    phi_list=[]

    combined_list=[]

    for contours in contours_by_level:
        
        inner_list=[]

        for contour in contours:
           
            theta, phi = contour.T
            phi[phi<0] += 2.0*numpy.pi
            inner_list.append(numpy.concatenate( [theta[:,None], phi[:,None]], axis=1))

        combined_list.append(inner_list)
       
    return combined_list

## find closest contour to point *s*
def find_closest_contour(pdf, s, all_xyz_contours, contor_probs_all_cov):
    
    cur_min_index=-1
    overall_min=9999999999999999999999

    np_s=s[None,:].cpu().numpy()
    
    for ind in range(len(all_xyz_contours)):
        
        min_dist=min(numpy.sqrt( numpy.sum((np_s-all_xyz_contours[ind])**2, axis=1)))
       
        if(min_dist<overall_min):
           
            overall_min=min_dist
            cur_min_index=ind
            
    return contor_probs_all_cov[cur_min_index]

def find_1d_contours(proportions, xvals, pdf_evals_with_area, pdf_evals):
    """
    Find 1D contours for a given level of a 1D function.

    :param func: The 1D function.
    :param level: The level for which to find the contour points.
    :param domain: A tuple (start, end) representing the domain to search.
    :param num_points: Number of points to sample in the domain.
    :return: A list of x values (of size num_valuesX1) where func(x) is approximately equal to the level.
    """



    levels = []

    inv_sorted=numpy.argsort(pdf_evals_with_area)[::-1]
    

    sorted_pdf_with_area = pdf_evals_with_area[inv_sorted] #list(reversed(list(sorted(samples))))
    sorted_pdf=pdf_evals[inv_sorted]

    for proportion in proportions:
     
        larger_mask=numpy.cumsum(sorted_pdf_with_area) > proportion
      
        if(larger_mask.sum()>0):
            level_index = (numpy.cumsum(sorted_pdf_with_area) > proportion).tolist().index(True)

            level = (sorted_pdf[level_index] + (sorted_pdf[level_index+1] if level_index+1 < len(sorted_pdf_with_area) else 0)) / 2.0
            levels.append(level)
        else:
            ## did we alrady attach it? must have decreasing sequence
            
            levels.append(min(sorted_pdf))

    levels=numpy.array(levels)
    equal_last_levels=levels==min(sorted_pdf)

    if(equal_last_levels.sum()>1):
        # we have to make the last entries decreasing while being larger or equal than min
        last_ones=[]
        for ind in range(equal_last_levels.sum()):
            last_ones.append(min(sorted_pdf)*(1+ind*0.01))
        last_ones=last_ones[::-1]

 
        levels[equal_last_levels]=numpy.array(last_ones)

    tweak_offset=1e-10
    # Initialize the transformed array with the first element
    transformed_levels = [levels[0]]
    
    # Transform to a new array that has strictly decreasing elemeents
    for i in range(1, len(levels)):
        # If the current element is the same as the previous one
        if levels[i] == levels[i - 1]:
            # Calculate the number of times this element has appeared consecutively
            count = 1
            while i - count >= 0 and levels[i] == levels[i - count]:
                count += 1
            # Add the element with the offset
            transformed_levels.append(levels[i] - (count - 1) * tweak_offset)
        else:
            # If it's not a repeating element, just add it to the transformed array
            transformed_levels.append(levels[i])

    levels=numpy.array(transformed_levels)

    ##### #################
    

    contour_list = []

    for level in levels:
        contour=[]
        for i in range(len(pdf_evals) - 1):
            if (pdf_evals[i] - level) * (pdf_evals[i + 1] - level) <= 0:
                # Linear interpolation to find a more accurate point of crossing
                x_contour = xvals[i] + (level - pdf_evals[i]) * (xvals[i + 1] - xvals[i]) / (pdf_evals[i + 1] - pdf_evals[i])
                contour.append(x_contour)
        """
        if(len(contour)==1):
            contour.append(contour[-1])
        assert(len(contour)==2), ("Issue with contour ", contour)
        """
        contour_list.append(numpy.array(contour)[...,None])

    return contour_list

def fake_plot_and_calc_eucl_contours(ax, colors, proportions, xvals, yvals, pdf_evals_with_area, pdf_evals, linewidths=1.0, linestyles=["-"]):
    """
    Calculate contours using matplotlib ax object.
    """
    levels = []

    inv_sorted=numpy.argsort(pdf_evals_with_area)[::-1]
    

    sorted_pdf_with_area = pdf_evals_with_area[inv_sorted] #list(reversed(list(sorted(samples))))
    sorted_pdf=pdf_evals[inv_sorted]

    for proportion in proportions:
     
        larger_mask=numpy.cumsum(sorted_pdf_with_area) > proportion
      
        if(larger_mask.sum()>0):
            level_index = (numpy.cumsum(sorted_pdf_with_area) > proportion).tolist().index(True)

            level = (sorted_pdf[level_index] + (sorted_pdf[level_index+1] if level_index+1 < len(sorted_pdf_with_area) else 0)) / 2.0
            levels.append(level)
        else:
            ## did we alrady attach it? must have decreasing sequence
            
            levels.append(min(sorted_pdf))

    levels=numpy.array(levels)
    equal_last_levels=levels==min(sorted_pdf)

    if(equal_last_levels.sum()>1):
        # we have to make the last entries decreasing while being larger or equal than min
        last_ones=[]
        for ind in range(equal_last_levels.sum()):
            last_ones.append(min(sorted_pdf)*(1+ind*0.01))
        last_ones=last_ones[::-1]

 
        levels[equal_last_levels]=numpy.array(last_ones)

    tweak_offset=1e-10
    # Initialize the transformed array with the first element
    transformed_levels = [levels[0]]
    
    # Transform to a new array that has strictly decreasing elemeents
    for i in range(1, len(levels)):
        # If the current element is the same as the previous one
        if levels[i] == levels[i - 1]:
            # Calculate the number of times this element has appeared consecutively
            count = 1
            while i - count >= 0 and levels[i] == levels[i - count]:
                count += 1
            # Add the element with the offset
            transformed_levels.append(levels[i] - (count - 1) * tweak_offset)
        else:
            # If it's not a repeating element, just add it to the transformed array
            transformed_levels.append(levels[i])

    transformed_levels=numpy.array(transformed_levels)
    
    per_dim=int(numpy.sqrt(pdf_evals.shape[0]))
    pdf_evals_resized=numpy.resize(pdf_evals, (per_dim,per_dim))

   
    res = ax.contour(xvals, 
                       yvals,
                       pdf_evals_resized.T,
                       linestyles=linestyles[::-1],
                       linewidths=linewidths,
                       levels=transformed_levels[::-1],
                       colors=colors[::-1])

    return res.allsegs[::-1]