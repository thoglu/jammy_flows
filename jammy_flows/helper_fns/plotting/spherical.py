import matplotlib
from matplotlib.colors import LogNorm
import pylab
import torch
import numpy
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy import units as u

import mhealpy
from mhealpy import HealpixMap
from mhealpy.plot.axes import HealpyAxes

from matplotlib.projections import register_projection
from matplotlib.transforms import Bbox

from ..contours import compute_contours, CustomSphereContourSet

###### plotting functions for sphere (s2), employing a flexible grid to save computing while still having smooth contours

#### monkeypatching add function for more flexible ticklabels
def add(self,
        axis=None,
        world=None,
        pixel=None,
        angle=None,
        text=None,
        axis_displacement=None,
        data=None,
    ):
        """
        Add a label.

        Parameters
        ----------
        axis : str
            Axis to add label to.
        world : Quantity
            Coordinate value along this axis.
        pixel : [float, float]
            Pixel coordinates of the label. Deprecated and no longer used.
        angle : float
            Angle of the label.
        text : str
            Label text.
        axis_displacement : float
            Displacement from axis.
        data : [float, float]
            Data coordinates of the label.
        """
       
        required_args = ["axis", "world", "angle", "text", "axis_displacement", "data"]
        
        if (
            axis is None
            or world is None
            or angle is None
            or text is None
            or axis_displacement is None
            or data is None
        ):
            raise TypeError(
                f"All of the following arguments must be provided: {required_args}"
            )

        
        self.world[axis].append(world)
        self.data[axis].append(data)
        self.angle[axis].append(angle)
       
        num_significant=0
        if("." in text):
            num_significant=len(text.split(".")[-1][:-1])
            txt_lab="%."+("%d" % num_significant)+"f°"
            self.text[axis].append(txt_lab % (90.0-world))
        else:
            self.text[axis].append("%d°" % (90-world))

        self.disp[axis].append(axis_displacement)

        self._stale = True

class HealpyAxesAzimuthOrdering(HealpyAxes):

    def __init__(self, *args, rot = 0, **kwargs):
        """
        We creae a "custom" cylindrial axes here that is always fixed in terms of rotation
        """
        
        ## *args must contain fig and rect
        fig=args[0]
        rect=args[1:]

        if not isinstance(rect, Bbox):
            if(type(rect)==int or type(rect)==tuple):
                ## whole figure.. assert that indices match with that
                if(type(rect)==int):
                    assert(rect==1), rect
                    rect = Bbox.from_bounds(0,0,1,1)
                elif(type(rect)==tuple):
                    if(len(rect)==1):
                        assert(isinstance(rect[0], Bbox))
                        rect = rect[0]
                    else:
                        assert(rect[0]==1 and rect[1]==1 and rect[2]==1), rect
                        rect = Bbox.from_bounds(0,0,1,1)
                
            elif(type(rect)==list):
                rect = Bbox.from_bounds(*rect)
            elif(type(rect)==matplotlib.gridspec.SubplotSpec):
                rect=rect.get_position(fig).extents
                rect=Bbox.from_bounds(*rect)

            else:
                raise Exception("Unknown ax rect ", type(rect))

        ## default setting for 
        fixed_rot=numpy.array([0.0,90.0,0.0]) # rotation z set to 0 ensures azimuth starts at 0 
        
        super().__init__(fig, rect,
                         rot = fixed_rot,
                         flip="geo", # this ordering assures that azimuth spans from left to right
                         **kwargs)

    def graticule(self, 
                  dpar = 45, 
                  dmer = 60, 
                  grid = True,
                  ticks = True, 
                  show_zenith_axis=True,
                  zenith_axislabel_minpad=2.0,
                  show_azimuth_axis=True,
                  tick_format = 'd', frame = True, zen_azi_mode="zen_azi",**kwargs):
        """
        Graticule overwrite.

        Args:
            dpar (float): Interval for the latitude axis (parallels)
            dmer (float): Interval for the longitude axis (meridians)
            grid (bool): Whether to show the grid
            ticks (bool): Whether to shoe the tick labels
            tick_format ('str'): Tick label formater. e.g. 'dd:mm:ss.s', 
                'hh:mm:ss.s', 'd.d'
            frame (bool): Draw plot frame  
        """

        assert(zen_azi_mode=="zen_azi"), "Only supporting zen_azi mode right now"

        self.grid(grid, **kwargs)

        if(ticks):
            self.coords[0].set_ticks(spacing=dmer * u.deg)
            self.coords[1].set_ticks(spacing=dpar * u.deg)

        if(show_azimuth_axis):
            self.coords[0].set_major_formatter(tick_format)
            self.coords[0].set_ticklabel_visible(ticks)
            self.coords[0].set_ticklabel(color='white', size=12)

        if(show_zenith_axis):
            self.coords[1].set_major_formatter(tick_format)
            self.coords[1].set_ticklabel_visible(ticks)
            self.coords[1].set_ticklabel(color='black', size=12)

        if frame:
            self.coords.frame.set_linewidth(kwargs.get('linewidth', 3))
            self.coords.frame.set_color(kwargs.get('color', 'black'))
        else:
            self.coords.frame.set_linewidth(0)            
            
  
        # shift axis label to left
        #self.coords[1].set_label_coords(-0.1, 0.5)

        if(zen_azi_mode=="zen_azi"):
            if(show_azimuth_axis):
                self.coords[0].set_axislabel("azimuth [deg]", color="white") # shift the label a little to the left

            if(show_zenith_axis):
                self.coords[1].set_axislabel("zenith [deg]", minpad=zenith_axislabel_minpad) # shift the label a little to the left

                ## monkeypatching ticklabels for zenith
                self.coords[1].ticklabels.add = add.__get__(self.coords[1].ticklabels)
                
        
class MollviewAzimuth(HealpyAxesAzimuthOrdering):

    name = "mollview_azimuth"
    _wcsproj = "MOL"
    _aspect = 2

    ## multiply by 0.995 for correct labeling (due to rounding errors of WCSaxis)
    _cdelt = 2*numpy.sqrt(2)/numpy.pi*0.995 # Sqrt of pixel size
    _autoscale = True
    _center = [0,0]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         frame_class = kwargs.pop('frame_class', EllipticalFrame),
                         **kwargs)
        
register_projection(MollviewAzimuth)

class OrthviewAzimuth(HealpyAxes):

    name = "orthview_azimuth"
    _wcsproj = "SIN"
    _aspect  = 1
    _center = [0,0]
    # Sqrt of pixel area at point of tangency. Sign matches healpy
    _cdelt = -1/numpy.pi
    
    
    _autoscale = True

    def __init__(self, *args, zoom_center=None, zoom_diameter=None,**kwargs):

        ## *args must contain fig and rect
        fig=args[0]
        rect=args[1:]

        if not isinstance(rect, Bbox):
            if(type(rect)==int or type(rect)==tuple):
                ## whole figure.. assert that indices match with that
                if(type(rect)==int):
                    assert(rect==1), rect
                    rect = Bbox.from_bounds(0,0,1,1)
                elif(type(rect)==tuple):
                    if(len(rect)==1):
                        assert(isinstance(rect[0], Bbox))
                        rect = rect[0]
                    else:
                        assert(rect[0]==1 and rect[1]==1 and rect[2]==1), rect
                        rect = Bbox.from_bounds(0,0,1,1)
                
            elif(type(rect)==list):
                rect = Bbox.from_bounds(*rect)
            elif(type(rect)==matplotlib.gridspec.SubplotSpec):
                rect=rect.get_position(fig).extents
                rect=Bbox.from_bounds(*rect)

            else:
                raise Exception("Unknown ax rect ", type(rect))
        
        self._cdelt = 1/numpy.pi
        
        if(zoom_diameter is not None):
            # zoom_diameter is in radian .. only use if smaller than pi
            if(zoom_diameter<numpy.pi):
                self._cdelt=(zoom_diameter/numpy.pi)*1/numpy.pi

            ## multiply by 0.995 for correct labeling (due to rounding errors of WCSaxis)
            self._cdelt=self._cdelt*0.995
            
        rot=(0.0,0.0,180.0)
        if(zoom_center is not None):
          
            assert(len(zoom_center)==2)
            # zoom_center is in zen/azimuth, but rotation entries are opposite order
            # last entry has to be 180.0 to yield correct view
            rot=(zoom_center[1]*180.0/numpy.pi, 90.0-(zoom_center[0]*180.0/numpy.pi), 180.0)

        super().__init__(fig, rect,
                         flip="geo",
                         rot=rot,
                         frame_class = kwargs.pop('frame_class', EllipticalFrame),
                         **kwargs)

    def graticule(self, 
                  dpar = 45, 
                  dmer = 60, 
                  grid = True,
                  ticks = True, 
                  show_zenith_axis=True,
                  zenith_axislabel_minpad=4.0,
                  show_azimuth_axis=False,
                  tick_format = 'd', frame = True, zen_azi_mode="zen_azi",**kwargs):
        """
        Graticule overwrite.

        Args:
            dpar (float): Interval for the latitude axis (parallels)
            dmer (float): Interval for the longitude axis (meridians)
            grid (bool): Whether to show the grid
            ticks (bool): Whether to shoe the tick labels
            tick_format ('str'): Tick label formater. e.g. 'dd:mm:ss.s', 
                'hh:mm:ss.s', 'd.d'
            frame (bool): Draw plot frame  
        """

        assert(zen_azi_mode=="zen_azi"), "Only supporting zen_azi mode right now"

        self.grid(grid, **kwargs)

        if(ticks):
            self.coords[0].set_ticks(spacing=dmer * u.deg)
            self.coords[1].set_ticks(spacing=dpar * u.deg)

    
        self.coords[0].set_major_formatter(tick_format)
        self.coords[0].set_ticklabel_visible(ticks)
        self.coords[0].set_ticklabel(color='white', size=12)

        if frame:
            self.coords.frame.set_linewidth(kwargs.get('linewidth', 3))
            self.coords.frame.set_color(kwargs.get('color', 'black'))
        else:
            self.coords.frame.set_linewidth(0)    

       
        self.coords[1].set_major_formatter(tick_format)
        self.coords[1].set_ticklabel_visible(ticks)
        self.coords[1].set_ticklabel(color='black', size=12)

        ## monkeypatching ticklabels for zenith
        self.coords[1].ticklabels.add = add.__get__(self.coords[1].ticklabels)

        if(show_azimuth_axis):
            self.coords[0].set_axislabel("azimuth [deg]", color="white") # shift the label a little to the left

        if(show_zenith_axis):
            self.coords[1].set_axislabel("zenith [deg]", minpad=zenith_axislabel_minpad) # shift the label a little to the left


register_projection(OrthviewAzimuth)

def get_meshed_positions_and_areas(samples, 
                                   max_entries_per_pixel=10):
    """
    Obtain the positions and areas of a meshed healpy grid. Meshing is based on samples, such that at max *max_entries_per_pixel*
    fall into a given mesh simplex.
    """
    assert(samples.shape[1]==2)
    used_samples=samples
    
    if(type(samples)==torch.Tensor):
        used_samples=samples.cpu().detach().numpy()
    assert(type(used_samples)==numpy.ndarray), type(samples)
    
    sample_pix = mhealpy.ang2pix(mhealpy.MAX_NSIDE, used_samples[:,0], used_samples[:,1], nest = True)
    
    moc_map=HealpixMap.moc_histogram(mhealpy.MAX_NSIDE, sample_pix, max_entries_per_pixel, nest=True)
    
    pix_ids=numpy.arange(moc_map.npix)
    ang_vals=moc_map.pix2ang(pix_ids)
    ang_vals=numpy.concatenate([ang_vals[0][:,None], ang_vals[1][:,None]], axis=1)
   
    per_pixel_nside, _ = mhealpy.uniq2nest(moc_map.pix2uniq(pix_ids))
    per_pixel_areas= mhealpy.nside2pixarea(per_pixel_nside)
    
    return ang_vals, per_pixel_areas, moc_map

def get_multiresolution_evals(pdf, 
                              sub_pdf_index=0,
                              samplesize=10000, 
                              max_entries_per_pixel=5,
                              use_density_if_possible=True):
    """
    Sample a model and mesh the sky based on the samples, evaluate it, and calculate areas of mesh regions. 

    Returns:
        eval_positions (numpy.ndrray): Positions of mesh simplex centers.
        pdf_evals (numpy.ndarray): Evaluations of PDF at mesh simplex centers.
        eval_areas (numpy.ndarray): Mesh simplex areas.
        moc_map (Healpix map): Multiresolution healpix map
    """

    samples,_,_,_=pdf.sample(samplesize=samplesize)
    eval_positions, eval_areas, moc_map=get_meshed_positions_and_areas(samples,max_entries_per_pixel=max_entries_per_pixel)

    assert(pdf.pdf_defs_list[sub_pdf_index]=="s2"), ("Trying to get multiresolution for s2 subdimension, but subdimension %d is of type %s" % (sub_pdf_index, pdf.pdf_defs_list[sub_pdf_index]))

    if(use_density_if_possible and (sub_pdf_index==0)):
        xyz_positions=pdf.transform_target_into_returnable_params(torch.from_numpy(eval_positions).to(samples))
        log_pdf,_,_=pdf(xyz_positions, force_embedding_coordinates=True)
        pdf_evals=log_pdf.exp().cpu().detach().numpy()
    else:

        
        ipix_vals=moc_map.ang2pix(samples[:,0], samples[:,1])
        sorted_pixel_vals=numpy.sort(ipix_vals)

       
        unique_indices, counts=numpy.unique(sorted_pixel_vals, return_counts=True)
       
        pdf_evals=numpy.zeros(len(eval_areas))
        pdf_evals[unique_indices]=counts/float(sum(counts))#eval_areas[unique_indices]
        pdf_evals=pdf_evals/eval_areas

    return eval_positions, pdf_evals, eval_areas, moc_map


def plot_multiresolution_healpy(pdf,
                                fig=None, 
                                ax_to_plot=None,
                                samplesize=10000, 
                                sub_pdf_index=0,
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=True,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                visualization="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None): # required to transform to dec/ra before plotting 
    
    """
    Visualizes an S2 pdf, or a certain S2 subpart of a PDF using an adaptive healpix grid from mhealpy. Useful if the PDF
    is very small and higher nside becomes computationally expensive. Can also overlay smooth contours on this irregular grid
    using the *meander* package.

    """
    assert("s2" in pdf.pdf_defs_list ), "Requires that at least one s2 sub-manifold exists."

    eval_positions, pdf_evals, eval_areas, moc_map=get_multiresolution_evals(pdf, 
                                                                             sub_pdf_index=sub_pdf_index,
                                                                    samplesize=samplesize,
                                                                    max_entries_per_pixel=max_entries_per_pixel,
                                                                    use_density_if_possible=use_density_if_possible)
    
    
    ax=_plot_multiresolution_healpy(eval_positions,
                                    pdf_evals,
                                    eval_areas,
                                    moc_map=moc_map,
                                    fig=fig, 
                                    ax_to_plot=ax_to_plot,
                                    samplesize=samplesize, 
                                    max_entries_per_pixel=max_entries_per_pixel,
                                    draw_pixels=draw_pixels,
                                    use_density_if_possible=use_density_if_possible,
                                    log_scale=log_scale,
                                    cbar=cbar,
                                    cbar_kwargs=cbar_kwargs,
                                    graticule=graticule,
                                    graticule_kwargs=graticule_kwargs,
                                    draw_contours=draw_contours,
                                    contour_probs=contour_probs,
                                    contour_colors=contour_colors, # None -> pick colors from color scheme
                                    zoom=zoom,
                                    visualization=visualization, # zen_azi or dec_ra
                                    declination_trafo_function=declination_trafo_function) # required to transform to dec/ra before plotting )

    return ax

def _plot_multiresolution_healpy(eval_positions,
                                pdf_evals,
                                eval_areas,
                                moc_map=None,
                                fig=None, 
                                ax_to_plot=None,
                                samplesize=10000, 
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=True,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                visualization="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None): # required to transform to dec/ra before plotting 
    
    """
    Visualizes an S2 pdf, or a certain S2 subpart of a PDF using an adaptive healpix grid from mhealpy. Useful if the PDF
    is very small and higher nside becomes computationally expensive. Can also overlay smooth contours on this irregular grid
    using the *meander* package. This function differs in that it directly takes the PDF values instead of a pdf object.

    """
    
    zoom_diameter=None
    mean_coords=None

    ## recreate moc map if necessary
    if(moc_map is None):
        sample_pix = mhealpy.ang2pix(mhealpy.MAX_NSIDE, eval_positions[:,0], eval_positions[:,1], nest = True)
        moc_map = HealpixMap.moc_histogram(mhealpy.MAX_NSIDE, sample_pix, 1, nest=True)

    if(zoom):
        
        tot_sums=pdf_evals*eval_areas
        sorta=numpy.argsort(tot_sums)[::-1] # large to small
        ## add largest areas
        contained_mask=numpy.cumsum(tot_sums[sorta])<0.97
        contained_positions=eval_positions[sorta][contained_mask]

        # calculate zoom center
        argmax_index=numpy.argmax(pdf_evals)
        mean_coords=eval_positions[argmax_index]

        # calculate zoom diameter
        zen_range=(min(contained_positions[:,0]),max(contained_positions[:,0]) )
        azi_range=(min(contained_positions[:,1]),max(contained_positions[:,1]) )

        # zoom can be on 2pi boundary.. make sure this is taken into account in diameter
        if( (mean_coords[1]>=azi_range[0]) and (mean_coords[1]<=azi_range[1])):
            azi_diff=azi_range[1]-azi_range[0]
        else:
            azi_diff=azi_range[0]+2*numpy.pi-azi_range[1]

        zen_diff=zen_range[1]-zen_range[0]

        # take larger value as approximate diameter
        zoom_diameter=max(zen_diff, azi_diff)


    if(fig is None):
        assert(ax_to_plot is None)

        kw_dict=dict()

        if(zoom):
            kw_dict["projection"]="orthview_azimuth"
            kw_dict["zoom_center"]=mean_coords
            kw_dict["zoom_diameter"]=zoom_diameter
        else:
            kw_dict["projection"]="mollview_azimuth"

        fig=pylab.figure(figsize=(8,6), dpi=200)
        ax=fig.add_subplot(1,1,1,**kw_dict)
    else:
        assert(ax_to_plot is not None)

        if(not isinstance(ax_to_plot, WCSAxes)):
            if(zoom):
                ax=OrthviewAzimuth(fig, ax_to_plot.get_position(), zoom_center=mean_coords, zoom_diameter=zoom_diameter)
            else:
                ax=MollviewAzimuth(fig, ax_to_plot.get_position())

            fig.add_axes(ax)
            ax_to_plot.remove() # remove original ax_to_plot
        else:
            ax=ax_to_plot
        
    if(draw_pixels):
        moc_map.density(density=True)
        moc_map[numpy.arange(len(eval_positions))]=pdf_evals

        if(log_scale):
            img,_=moc_map.plot(ax, cbar=False, norm=LogNorm())
        else:
            img,_=moc_map.plot(ax, cbar=False)

        if(cbar):

            default_cbar_kwargs=dict()
            default_cbar_kwargs["orientation"]="horizontal"
            default_cbar_kwargs["pad"]=0.05
            default_cbar_kwargs["fraction"]=0.1
            default_cbar_kwargs["shrink"]=0.5
            default_cbar_kwargs["aspect"]=25
            default_cbar_kwargs["label"]=r"pdf value"

            for extra_kwarg in cbar_kwargs:
                default_cbar_kwargs[extra_kwarg]=cbar_kwargs[extra_kwarg]
           
            fig.colorbar(img, ax = ax, **default_cbar_kwargs)
        
    contours=None
    if(draw_contours):
        if(contour_probs is not None):

            used_contour_colors=matplotlib.cm.tab10.colors

            if(contour_colors is not None):
                used_contour_colors=contour_colors

            assert(len(used_contour_colors)>=len(contour_probs))

            used_contour_colors=used_contour_colors[:len(contour_probs)]

            ret=CustomSphereContourSet(ax, 
                                       eval_positions[:,0], 
                                       eval_positions[:,1], 
                                       pdf_evals, 
                                       eval_areas,
                                       levels=contour_probs[::-1],
                                       colors=used_contour_colors[::-1],
                                       zorder=1
                                       )
            
            
            fmt_dict = dict()
            for ind, cprob in enumerate(contour_probs):
                fmt_dict[contour_probs[ind]] = "%d" % (int(cprob * 100)) + r" %"
            
            ax.clabel(ret,
                    fontsize=9,
                    inline=1,
                    fmt=fmt_dict,
                    levels=contour_probs[::-1],
                    colors=used_contour_colors[::-1],
                    zorder=1)
          

    if(graticule):
        graticule_default_kwargs=dict()

        if(zoom):
            ## find out how many gridlines to show by default 
            desirable_dist_between_graticules=(180.0/numpy.pi)*zoom_diameter/5.0
            
            target=3
            smaller_than_target=desirable_dist_between_graticules<target
            num_significant_points=0

            while(smaller_than_target):
                target=target/10.0
                num_significant_points+=1
                smaller_than_target=desirable_dist_between_graticules<target

         
            grat_format="d"
            if(num_significant_points>0):
                grat_format=grat_format+"."+"d"*num_significant_points

            graticule_default_kwargs["tick_format"]=grat_format

        for extra_kwarg in graticule_kwargs:
            graticule_default_kwargs[extra_kwarg]=graticule_kwargs[extra_kwarg]

        ax.graticule(**graticule_default_kwargs)

    return ax

