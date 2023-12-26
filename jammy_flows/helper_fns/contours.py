import numpy
import numpy as np
from numpy import ma
import matplotlib
import matplotlib as mpl

try:
    import meander
except:
    print("Meander not installed... contours can not be calculated.")
    meander=None

try:
    import healpy
except:
    print("Healpy not installed... spherical contours can not be calculated.")
    healpy=None


def find_contour_levels(proportions, pdf_evals, areas):

    assert(len(pdf_evals.shape)==1), "pdf evals must be a 1-d array!"

    pdf_evals_with_area=pdf_evals*areas

    levels = []

    inv_sorted=numpy.argsort(pdf_evals)[::-1]
    
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

    return levels

def compute_contours(proportions, pdf_evals, areas, sample_points=None, manifold="euclidean"):
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

    
    levels=find_contour_levels(proportions, pdf_evals, areas)
    
    ##############################################
    combined_list=[]

    if(manifold=="euclidean"):
        assert(sample_points is not None)

        if(sample_points.shape[1]==1):
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

                ## create 1 "joint" 1-d contour here
                combined_list.append(numpy.array(contour)[...,None])
        elif(sample_points.shape[1]==2):
            contours_by_level = meander.euclidean_contours(sample_points, pdf_evals, levels)

    elif(manifold=="sphere"):

        if(sample_points is None):
            nside = healpy.pixelfunc.get_nside(sorted_pdf_with_area)
            sample_points = numpy.array(healpy.pix2ang(nside,numpy.arange(len(sorted_pdf_with_area)))).T

        contours_by_level = meander.spherical_contours(sample_points, pdf_evals, levels)
    else:
        raise Exception("Unknown manifold for coverage! ", manifold)

    
    
    if(sample_points.shape[1]==2):
        theta_list = []
        phi_list=[]

        combined_list=[]

        for contours in contours_by_level:
            
            inner_list=[]

            for contour in contours:
               
                theta, phi = contour.T
                if(manifold=="sphere"):
                    phi[phi<0] += 2.0*numpy.pi
                inner_list.append(numpy.concatenate( [theta[:,None], phi[:,None]], axis=1))

            combined_list.append(inner_list)
       
    return combined_list

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




"""
Custom contour generator for CustomSphereContourSet.
"""
class custom_contour_generator(object):
    
    def __init__(self, x,y, pdf_evals, areas, ax_obj):
        self.joint_xy=np.concatenate([x[:,None],y[:,None]],axis=1)
        self.pdf_evals=pdf_evals
        self.areas=areas
        self.ax_obj=ax_obj
    
    
    def _get_azimuth_split_contours(self, c, is_azimuthal=True):
        # split contour into isolated ones that get split by azimuth split
       
        reduced_contour=c#[:-1]
        new_groups=[]
        
        last_start=0
        num_splits=0
        
        if(is_azimuthal==False):
            diffs=numpy.sqrt(numpy.sum((reduced_contour[1:]-reduced_contour[:-1])**2, axis=1))
            median_diff=numpy.median(diffs)
            
            
        for cur_ind in range(len(reduced_contour)-1):
            ## for azimuthal coordinates .. a jump in delta azimuth > 3 indicates a split

            if(is_azimuthal):
                split_condition=numpy.fabs(reduced_contour[cur_ind+1][1]-reduced_contour[cur_ind][1])>3
            else:
                # world pixel coordinates
                this_diff=diffs[cur_ind]
                split_condition=this_diff>8*median_diff

            if(split_condition):

                #if(is_azimuthal):
                #    print("cur ind ", cur_ind)
                #    print("SPLIT:", numpy.fabs(reduced_contour[cur_ind+1][1]-reduced_contour[cur_ind][1]), reduced_contour[cur_ind+1], reduced_contour[cur_ind])
                #    print(reduced_contour[:10], reduced_contour[-10:])
                #    print("----------")
                if( (cur_ind+1-last_start)>2):
                    new_groups.append(reduced_contour[last_start:cur_ind+1])
                #else:
                #    print("dropping len ..", (cur_ind+1-last_start))
                num_splits+=1
                 
                    
                last_start=cur_ind+1
            
            if(cur_ind==(len(reduced_contour)-2)):
                ## we already have split at least one off, and now reached the end.. create another item
                if(num_splits>=1):
                    #print("final one")
                    #print(reduced_contour[last_start:cur_ind+2])
                    #print("OTHER LAST ONE", c[-3:])
                    ## check if the last one of real contour is also
                    new_groups.append(reduced_contour[last_start:cur_ind+2])
                    num_splits+=1
            
        if(num_splits==0):
           
            return [c]
        
        return new_groups
        
        
    def _make_isolated_kind(self,c):
        new_kind=[1]+(len(c)-2)*[2]
            
        if numpy.fabs(c[0]-c[-1]).sum()==0:
            new_kind.append(79)
        else:
            new_kind.append(2)
            
        return new_kind
        
        
    def create_contour(self, contour_prob):
        
        contours=compute_contours([contour_prob], self.pdf_evals, self.areas, sample_points=self.joint_xy, manifold="sphere")
       
        contours=contours[0]
        
        all_contours=[]
        
        min_contour_len=2

        for c in contours:
            
            sub_contours=self._get_azimuth_split_contours(c)
            
            for s in sub_contours:
                if(len(s)>min_contour_len):
                    all_contours.append(s)
                ## rad to deg, zen->dec etc
        
        transformed_contours=[]
        for c in all_contours:
            new_dec=90.0-c[:,0]*180.0/numpy.pi
            new_ra=c[:,1]*180.0/numpy.pi
            transformed_contours.append(numpy.concatenate([new_ra[:,None],new_dec[:,None]],axis=1))
       
        ## nans can appear for hidden points in current projection.. get rid of those...

        contours=[]

        for c in transformed_contours:
            safe_c=self.ax_obj.wcs.all_world2pix(c,1)[~numpy.isnan(self.ax_obj.wcs.all_world2pix(c,1)[:,0])]
            
            if(len(safe_c)>0):
                #contours.append(safe_c)
                
                sub_contours=self._get_azimuth_split_contours(safe_c, is_azimuthal=False)
                
                for sub_c in sub_contours:
                    if(len(sub_c)>min_contour_len): # only take contours longer than 2
                       
                        contours.append(sub_c)
                    

        #contours=[self.ax_obj.wcs.all_world2pix(c,1)[~numpy.isnan(self.ax_obj.wcs.all_world2pix(c,1))] for c in transformed_contours]
        #print("transformed conts ", contours)
        assert(type(contours)==list)
        
        
        all_kinds=[]
     
        for c in contours:
            
            ## default is a repeating kind
            new_kind=[1]+(len(c)-2)*[2]+[79]
            
            if( (c[0]-c[-1]).sum()==0):
                new_kind[-1]=2
           
            all_kinds.append(numpy.array(new_kind, dtype=numpy.uint8))
            
                
        return contours, all_kinds
    
    

class CustomSphereContourSet(matplotlib.contour.ContourSet):
    """
    A custom contour set that has similar structure to QuadContourSet in matplotlib,
    but is customized to work with variable resolution spherical data.
    """

    def _process_args(self, *args, corner_mask=None, algorithm=None, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], matplotlib.contour.QuadContourSet):
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._corner_mask = args[0]._corner_mask
            contour_generator = args[0]._contour_generator
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
            self._algorithm = args[0]._algorithm
        else:
            import contourpy

            if algorithm is None:
                algorithm = mpl.rcParams['contour.algorithm']
            mpl.rcParams.validate["contour.algorithm"](algorithm)
            self._algorithm = algorithm

            if corner_mask is None:
                if self._algorithm == "mpl2005":
                    # mpl2005 does not support corner_mask=True so if not
                    # specifically requested then disable it.
                    corner_mask = False
                else:
                    corner_mask = mpl.rcParams['contour.corner_mask']
            self._corner_mask = corner_mask
    
            assert(len(args)==4), args
            
            x = args[0]
            y = args[1]
            log_evals = args[2]
            areas = args[3]
            
            self.zmin=min(log_evals)
            self.zmax=max(log_evals)

            contour_generator = custom_contour_generator(x,y,log_evals, areas, self.axes)

            t = self.get_transform()

            # if the transform is not trans data, and some part of it
            # contains transData, transform the xs and ys to data coordinates
            if (t != self.axes.transData and
                    any(t.contains_branch_seperately(self.axes.transData))):
                trans_to_data = t - self.axes.transData
                pts = np.vstack([x.flat, y.flat]).T
                transformed_pts = trans_to_data.transform(pts)
                x = transformed_pts[..., 0]
                y = transformed_pts[..., 1]

            self._mins = [ma.min(x), ma.min(y)]
            self._maxs = [ma.max(x), ma.max(y)]

        self._contour_generator = contour_generator

        return kwargs

    def _contour_args(self, args, kwargs):
        if self.filled:
            fn = 'contourf'
        else:
            fn = 'contour'
        nargs = len(args)
        if nargs <= 2:
            z, *args = args
            z = ma.asarray(z)
            x, y = self._initialize_x_y(z)
        elif nargs <= 4:
            x, y, z_orig, *args = args
            x, y, z = self._check_xyz(x, y, z_orig, kwargs)
        else:
            raise _api.nargs_error(fn, takes="from 1 to 4", given=nargs)
        z = ma.masked_invalid(z, copy=False)
        self.zmax = float(z.max())
        self.zmin = float(z.min())
        if self.logscale and self.zmin <= 0:
            z = ma.masked_where(z <= 0, z)
            _api.warn_external('Log scale: values of z <= 0 have been masked')
            self.zmin = float(z.min())
        self._process_contour_level_args(args, z.dtype)
        return (x, y, z)

    def _check_xyz(self, x, y, z, kwargs):
        """
        Check that the shapes of the input arrays match; if x and y are 1D,
        convert them to 2D using meshgrid.
        """
        x, y = self.axes._process_unit_info([("x", x), ("y", y)], kwargs)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = ma.asarray(z)

        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        if z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        Ny, Nx = z.shape

        if x.ndim != y.ndim:
            raise TypeError(f"Number of dimensions of x ({x.ndim}) and y "
                            f"({y.ndim}) do not match")
        if x.ndim == 1:
            nx, = x.shape
            ny, = y.shape
            if nx != Nx:
                raise TypeError(f"Length of x ({nx}) must match number of "
                                f"columns in z ({Nx})")
            if ny != Ny:
                raise TypeError(f"Length of y ({ny}) must match number of "
                                f"rows in z ({Ny})")
            x, y = np.meshgrid(x, y)
        elif x.ndim == 2:
            if x.shape != z.shape:
                raise TypeError(
                    f"Shapes of x {x.shape} and z {z.shape} do not match")
            if y.shape != z.shape:
                raise TypeError(
                    f"Shapes of y {y.shape} and z {z.shape} do not match")
        else:
            raise TypeError(f"Inputs x and y must be 1D or 2D, not {x.ndim}D")

        return x, y, z

    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i, j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        elif z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        else:
            Ny, Nx = z.shape
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                x0, x1, y0, y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)
        # Match image behavior:
        if self.extent is None:
            x0, x1, y0, y1 = (0, Nx, 0, Ny)
        else:
            x0, x1, y0, y1 = self.extent
        dx = (x1 - x0) / Nx
        dy = (y1 - y0) / Ny
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return np.meshgrid(x, y)