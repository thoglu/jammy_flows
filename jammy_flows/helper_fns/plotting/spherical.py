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

from ..contours import compute_contours, ContourGenerator
from .general import replace_axes_with_gridspec
from ...layers.spheres import sphere_base

import sys

from astropy.time import Time
import astropy.units as u

## functions to transform local to equatorial coordinates

def gmst_from_mjd(mjd):
    """
    Compute Greenwich Mean Sidereal Time (GMST) in radians from MJD.
    (An approximate formula.)
    """
    JD = mjd + 2400000.5
    GMST_hours = 18.697374558 + 24.06570982441908*(JD - 2451545.0)
    GMST_rad = (GMST_hours % 24) * (15 * numpy.pi/180.0)  # convert hours to radians
    return GMST_rad

def precession_matrix(mjd):
    """
    Compute the precession matrix from the apparent (date) equatorial frame to J2000
    using the IAU 1976 precession model.
    
    Assumes the provided MJD is on the TT scale.
    """
    JD = mjd + 2400000.5
    T = (JD - 2451545.0) / 36525.0  # Julian centuries from J2000.0
    # Precession angles in arcseconds (IAU 1976 approximations):
    zeta_arcsec  = (2306.2181 + 1.39656*T - 0.000139*T**2)*T + (0.30188 - 0.000344*T)*T**2 + 0.017998*T**3
    z_arcsec     = (2306.2181 + 1.39656*T - 0.000139*T**2)*T + (1.09468 + 0.000066*T)*T**2 + 0.018203*T**3
    theta_arcsec = (2004.3109 - 0.85330*T - 0.000217*T**2)*T - (0.42665 + 0.000217*T)*T**2 - 0.041833*T**3
    arcsec_to_rad = numpy.pi/(180.0*3600.0)
    zeta  = zeta_arcsec  * arcsec_to_rad
    z     = z_arcsec     * arcsec_to_rad
    theta = theta_arcsec * arcsec_to_rad
    # In the forward transformation we rotate from horizontal to the apparent equatorial frame,
    # then precess to J2000 via R_prec (apparent -> J2000) as:
    #   v_J2000 = v_eq_app @ R_prec.T
    # Here we define R_prec:
    return numpy.array([[ numpy.cos(zeta)*numpy.cos(theta)*numpy.cos(z) - numpy.sin(zeta)*numpy.sin(z),
                       -numpy.cos(zeta)*numpy.cos(theta)*numpy.sin(z) - numpy.sin(zeta)*numpy.cos(z),
                        numpy.cos(zeta)*numpy.sin(theta)],
                     [ numpy.sin(zeta)*numpy.cos(theta)*numpy.cos(z) + numpy.cos(zeta)*numpy.sin(z),
                      -numpy.sin(zeta)*numpy.cos(theta)*numpy.sin(z) + numpy.cos(zeta)*numpy.cos(z),
                        numpy.sin(zeta)*numpy.sin(theta)],
                     [ -numpy.sin(theta)*numpy.cos(z),
                        numpy.sin(theta)*numpy.sin(z),
                        numpy.cos(theta)]])
    # (Note: There are several equivalent ways to formulate precession; this polynomial form is a standard approximation.)

def local_to_apparent_pal(zenith, local_az, mjd, observer_lat, observer_lon):
    """
    Mimic the official PAL conversion from observed (local) to geocentric apparent coordinates.
    
    Parameters:
      zenith      : observed zenith distance in radians (0 = zenith)
      local_az    : local azimuth in radians, defined so that 0 = local East and increasing counterclockwise.
      mjd         : Modified Julian Date (assumed UT1-UTC = 0)
      observer_lat: observer's geodetic latitude (radians)
      observer_lon: observer's geodetic longitude (radians, positive east)
      
    This function mimics the official code:
    
        PalAz = π/2 - local_az - LONGITUDE
        then, for type 'A':
          ce = sin(zenith)
          x = -cos(PalAz)*ce
          y =  sin(PalAz)*ce
          z =  cos(zenith)
          
        Then, with LST computed from mjd and observer_lon, the hour angle (HA) is found 
        from (x,y,z) and RA = LST - HA.
    
    Returns:
      ra, dec : apparent Right Ascension and Declination in radians.
    """
    # Convert local azimuth to PAL's observed azimuth:
    PalAz = (numpy.pi/2 - local_az - observer_lon) % (2*numpy.pi)
    
    # Here, observed zenith distance remains unchanged.
    # According to the PAL routine, compute:
    ce = numpy.sin(zenith)  # ce = sin(zenith)
    x = -numpy.cos(PalAz) * ce
    y =  numpy.sin(PalAz) * ce
    z =  numpy.cos(zenith)
    
    # Now convert this vector to spherical coordinates.
    # For the PAL routines the conversion is done so that:
    #   declination = arcsin(z)
    dec = numpy.arcsin(z)
    
    # Compute the hour angle (HA) from x and y.
    # In PAL, after converting, they compute hma using the arctan2 of the components,
    # with the x component effectively reversed. One can mimic that by:
    HA = numpy.arctan2(y, -x)
    
    # Compute local sidereal time:
    GMST = gmst_from_mjd(mjd)
    LST = (GMST + observer_lon) % (2*numpy.pi)
    
    # Then, apparent RA = LST - HA, normalized to [0, 2π)
    ra = (LST - HA) % (2*numpy.pi)
    
    return ra, dec

def local_to_apparent(zenith, local_az, mjd, observer_lat, observer_lon):
    """
    Mimic the PAL routine for converting observed (local) coordinates
    to geocentric apparent RA/Dec.
    
    Inputs:
      zenith    : zenith distance in radians (0 at zenith)
      local_az  : local azimuth in radians, defined such that 0 = local East and increasing counterclockwise.
      mjd       : Modified Julian Date (assumed UT1–UTC = 0)
      observer_lat : observer's geodetic latitude (radians)
      observer_lon : observer's geodetic longitude (radians, positive east)
      
    Returns:
      ra, dec   : apparent Right Ascension and Declination in radians.
      
    This mimics the official PAL code:
       PalAz = π/2 – local_az – LONGITUDE
       then palOap is called with (PalAz, zenith, mjd, 0, LONGITUDE, LATITUDE, ... )
    """
    # Convert the local azimuth into PAL's expected observed azimuth:
    # (Here, PAL expects azimuth measured from North, with 0 = North, 90° = East.)
    PalAz = (numpy.pi/2 - local_az - observer_lon) % (2*numpy.pi)
    
    # The zenith remains unchanged (PAL uses the observed zenith distance).
    # Now, we need to convert the horizontal coordinates (PalAz, zenith) into equatorial coordinates.
    # First, convert zenith to altitude:
    alt = numpy.pi/2 - zenith  # altitude above horizon
    
    # In a local ENU system (with x = East, y = North, z = Up), 
    # a direction given by altitude and azimuth (where azimuth is measured from North)
    # has the unit vector:
    #   v_local = [ cos(alt)*sin(PalAz), cos(alt)*cos(PalAz), sin(alt) ]
    # (Note: since PAL assumes 0 = North, the roles of sine and cosine are swapped relative to our previous ENU system.)
    # Here we assume that is what PAL does.
    v_local = numpy.array([ numpy.cos(alt)*numpy.sin(PalAz),
                         numpy.cos(alt)*numpy.cos(PalAz),
                         numpy.sin(alt) ])
    
    # Now, to convert from local horizontal to equatorial (apparent) coordinates,
    # we use the observer’s local sidereal time (LST). In the PAL routines, the observer’s
    # longitude is applied internally as well. We compute GMST then LST:
    GMST = gmst_from_mjd(mjd)
    LST = (GMST + observer_lon) % (2*numpy.pi)
    
    # The conversion from horizontal (azimuth, altitude) to equatorial (hour angle, dec)
    # can be done by applying the following rotation (this is one standard formulation):
    #   sin(dec) = sin(alt)*sin(observer_lat) + cos(alt)*cos(observer_lat)*cos(PalAz)
    #   cos(dec)*sin(HA) = - cos(alt)*sin(PalAz)
    #   cos(dec)*cos(HA) = cos(alt)*cos(observer_lat)*cos(PalAz) - sin(alt)*sin(observer_lat)
    #
    # Then, RA = LST - HA.
    #
    # We can compute these step by step.
    
    dec = numpy.arcsin(numpy.sin(alt)*numpy.sin(observer_lat) + 
                    numpy.cos(alt)*numpy.cos(observer_lat)*numpy.cos(PalAz))
    
    # Compute hour angle (HA)
    # Here, we compute HA using the formula:
    #   HA = arctan2( - cos(alt)*sin(PalAz), cos(alt)*cos(observer_lat)*cos(PalAz) - sin(alt)*sin(observer_lat) )
    HA = numpy.arctan2(-numpy.cos(alt)*numpy.sin(PalAz),
                    numpy.cos(alt)*numpy.cos(observer_lat)*numpy.cos(PalAz) - numpy.sin(alt)*numpy.sin(observer_lat))
    
    # Now compute RA:
    ra = (LST - HA) % (2*numpy.pi)
    
    return ra, dec

def local_to_j2000_batch(zenith, local_az, mjd, observer_lat, observer_lon):
    """
    Convert arrays of local coordinates (zenith and local azimuth) to equatorial (J2000) RA/Dec.
    
    Parameters:
      zenith: array-like, zenith distance (radians) with 0 at zenith.
      local_az: array-like, local azimuth (radians) with 0 defined as local East and increasing counterclockwise.
      mjd: scalar, Modified Julian Date (assumed on the TT scale for precession).
      observer_lat: scalar, observer's latitude in radians.
      observer_lon: scalar, observer's longitude in radians (positive eastward).
    
    Returns:
      ra: array of Right Ascension in radians (in [0, 2π)).
      dec: array of Declination in radians (in [-π/2, π/2]).
    """
    # Convert local azimuth to the “PAL” convention:
    #   pal_az = π/2 – local_az – observer_lon

    pal_az = (numpy.pi/2 - local_az - observer_lon) % (2*numpy.pi)
    
    #pal_az = (numpy.pi/2 - local_az) % (2*numpy.pi)

    # Altitude is the complement of the zenith distance.
    alt = numpy.pi/2 - zenith
    cos_alt = numpy.cos(alt)
    sin_alt = numpy.sin(alt)
    # Build the horizontal (local) unit vectors.
    # We assume the pixel is defined in an ENU system (East, North, Up):
    v_local = numpy.stack((cos_alt * numpy.cos(pal_az),
                        cos_alt * numpy.sin(pal_az),
                        sin_alt), axis=-1)

    

    t = Time(mjd, format='mjd', scale='utc')
    lst = t.sidereal_time('apparent', longitude=observer_lon*u.rad)

    # Compute local sidereal time:
    GMST = gmst_from_mjd(mjd)
    LST = (GMST + observer_lon) % (2*numpy.pi)

   
    
    # Build the rotation matrix from horizontal to the (apparent) equatorial frame.
    R_lst = numpy.array([[ numpy.cos(LST), -numpy.sin(LST), 0],
                      [ numpy.sin(LST),  numpy.cos(LST), 0],
                      [ 0, 0, 1]])
    R_lat = numpy.array([[-numpy.sin(observer_lat), 0, numpy.cos(observer_lat)],
                      [0, -1, 0],
                      [numpy.cos(observer_lat), 0, numpy.sin(observer_lat)]])
    R_equ = R_lst @ R_lat
    # Rotate the local unit vectors.
    v_eq_app = v_local @ R_equ.T

    ra_app  = numpy.arctan2(v_eq_app[...,1], v_eq_app[...,0]) % (2*numpy.pi)
    dec_app = numpy.arcsin(v_eq_app[...,2])

   
    # Precess from the apparent equatorial frame to J2000.
    R_prec = precession_matrix(mjd)
    v_j2000 = v_eq_app @ R_prec.T
    
    # Convert the 3D vectors to spherical coordinates.
    ra  = numpy.arctan2(v_j2000[...,1], v_j2000[...,0]) % (2*numpy.pi)
    dec = numpy.arcsin(v_j2000[...,2])

    print(ra,ra_app, ra-ra_app)
    print(dec, dec_app, dec-dec_app)
    return ra, dec


def local_to_j2000_single(zenith, local_az, mjd, observer_lat, observer_lon):
    """
    Convert arrays of local coordinates (zenith and local azimuth) to equatorial (J2000) RA/Dec.
    
    Parameters:
      zenith: array-like, zenith distance (radians) with 0 at zenith.
      local_az: array-like, local azimuth (radians) with 0 defined as local East and increasing counterclockwise.
      mjd: scalar, Modified Julian Date (assumed on the TT scale for precession).
      observer_lat: scalar, observer's latitude in radians.
      observer_lon: scalar, observer's longitude in radians (positive eastward).
    
    Returns:
      ra: array of Right Ascension in radians (in [0, 2π)).
      dec: array of Declination in radians (in [-π/2, π/2]).
    """
    # Convert local azimuth to the “PAL” convention:
    #   pal_az = π/2 – local_az – observer_lon

    pal_az = (numpy.pi/2 - local_az - observer_lon) % (2*numpy.pi)
    
    #pal_az = (numpy.pi/2 - local_az) % (2*numpy.pi)

    zenith=numpy.array([ zenith])
    local_az=numpy.array([local_az])
    # Altitude is the complement of the zenith distance.
    alt = numpy.pi/2 - zenith
    cos_alt = numpy.cos(alt)
    sin_alt = numpy.sin(alt)
    # Build the horizontal (local) unit vectors.
    # We assume the pixel is defined in an ENU system (East, North, Up):
    v_local = numpy.stack((cos_alt * numpy.cos(pal_az),
                        cos_alt * numpy.sin(pal_az),
                        sin_alt), axis=-1)

    

    t = Time(mjd, format='mjd', scale='utc')
    lst = t.sidereal_time('apparent', longitude=observer_lon*u.rad)

    # Compute local sidereal time:
    GMST = gmst_from_mjd(mjd)
    LST = (GMST + observer_lon) % (2*numpy.pi)

   
    
    # Build the rotation matrix from horizontal to the (apparent) equatorial frame.
    R_lst = numpy.array([[ numpy.cos(LST), -numpy.sin(LST), 0],
                      [ numpy.sin(LST),  numpy.cos(LST), 0],
                      [ 0, 0, 1]])
    R_lat = numpy.array([[-numpy.sin(observer_lat), 0, numpy.cos(observer_lat)],
                      [0, -1, 0],
                      [numpy.cos(observer_lat), 0, numpy.sin(observer_lat)]])
    R_equ = R_lst @ R_lat
    # Rotate the local unit vectors.
    v_eq_app = v_local @ R_equ.T

    ra_app  = numpy.arctan2(v_eq_app[...,1], v_eq_app[...,0]) % (2*numpy.pi)
    dec_app = numpy.arcsin(v_eq_app[...,2])

   
    # Precess from the apparent equatorial frame to J2000.
    R_prec = precession_matrix(mjd)
    v_j2000 = v_eq_app @ R_prec.T
    
    # Convert the 3D vectors to spherical coordinates.
    ra  = numpy.arctan2(v_j2000[...,1], v_j2000[...,0]) % (2*numpy.pi)
    dec = numpy.arcsin(v_j2000[...,2])

    
    return dec[0], ra[0]

def decode_uniq(uniq):
    """
    Decode a UNIQ value according to the MOC standard.
    For a given UNIQ value:
      order = floor( ln(uniq/4) / ln(4) )
      nside = 2**order
      ipix  = uniq - 4 * (4**order)
    
    Returns (order, nside, ipix)
    """
    order = int(numpy.floor(numpy.log(uniq / 4) / numpy.log(4)))
    nside = 2**order
    ipix  = int(uniq - 4 * (4**order))
    return order, nside, ipix

def reproject_moc_local_to_equatorial(moc_map, lat_deg, lon_deg, mjd, local_offset=0.0, transfer_fn=local_to_j2000_single):
    """
    Reproject a HEALPix MOC map defined in local horizon coordinates (zenith/azimuth)
    into an equatorial (RA/Dec) MOC map.
    
    For each input pixel:
      - Decode its UNIQ value into (order, nside, ipix) using the MOC standard.
      - Get the pixel center angles (theta, phi) using healpy.pix2ang.
        (Here, theta is taken as the zenith distance and phi as the local azimuth.)
      - Optionally adjust phi by local_offset if needed.
      - Convert (zenith, local azimuth) to (RA, Dec) using local_to_j2000_batch.
      - Convert RA/Dec to a HEALPix pixel index (using the same nside) in equatorial coordinates.
      - Re-encode the pixel as a new UNIQ value.
    
    The resulting arrays of UNIQ values and data are then sorted.
    
    Parameters:
      moc_map   : HealpixMap (from mhealpy) representing the MOC in local coordinates.
      lat_deg   : Observer's geodetic latitude in degrees.
      lon_deg   : Observer's geodetic longitude in degrees (positive eastward).
      mjd       : Observation time (MJD, assumed on the TT scale).
      local_offset : Optional extra rotation (radians) to adjust your local azimuth.
                     (Default is 0, meaning local azimuth is defined as 0 = East.)
    
    Returns:
      equatorial_moc : A new HealpixMap in equatorial (RA/Dec) coordinates.
    """
    observer_lat = numpy.deg2rad(lat_deg)
    observer_lon = numpy.deg2rad(lon_deg)
    
    input_uniq = moc_map.uniq  # 1D array of UNIQ values.
    new_uniq_list = []
    new_value_list = []
    
    for i, uniq in enumerate(input_uniq):
        # Decode UNIQ properly:
        order, nside, ipix = decode_uniq(uniq)
        if not (nside & (nside - 1) == 0):
            raise ValueError("Decoded nside is not a power of 2!")
        
        # Obtain the pixel center in local coordinates.
        theta_local, phi_local = mhealpy.pix2ang(nside, ipix, nest=True)
        local_az = (phi_local + local_offset) % (2*numpy.pi)
        zenith = theta_local
        
        # Convert local coordinates to RA/Dec.
        dec, ra = transfer_fn(zenith,
                               local_az,
                               mjd,
                               observer_lat,
                               observer_lon)

       
        
        assert numpy.all((ra >= 0) & (ra < 2*numpy.pi)), "RA out of bounds"
        assert numpy.all((dec >= -numpy.pi/2) & (dec <= numpy.pi/2)), "Dec out of bounds"
        
        # For HEALPix in equatorial coordinates:
        theta_eq = numpy.pi/2 - dec  # colatitude for HEALPix.
        phi_eq = ra
        new_ipix = mhealpy.ang2pix(nside, theta_eq, phi_eq, nest=True)
        # Clip to valid range (0 to npix-1).
        max_ipix = mhealpy.nside2npix(nside)
        #print(".. max ipix", max_ipix)
        assert 0 <= new_ipix < max_ipix, f"new_ipix ({new_ipix}) out of bounds for nside {nside} (max {max_ipix})"
        if new_ipix >= max_ipix:
            new_ipix = max_ipix - 1

        assert(new_ipix<max_ipix)
        #print(".... new ipix ", new_ipix)
        new_uniq = 4 * (4**order) + new_ipix
        new_uniq_list.append(new_uniq)
        # Get pixel value.
        if hasattr(moc_map, 'data'):
            val = moc_map.data[i]
        else:
            val = 1

        new_value_list.append(val)
    
    new_uniq_array = numpy.array(new_uniq_list, dtype=numpy.int64)
    new_value_array = numpy.array(new_value_list)
    
    # Sort the new UNIQ array and corresponding values.
    sort_idx = numpy.argsort(new_uniq_array)
    new_uniq_array = new_uniq_array[sort_idx]
    new_value_array = new_value_array[sort_idx]
    
    # Remove duplicates if needed.
    if new_uniq_array.size > 0:
        uniq_final, ind = numpy.unique(new_uniq_array, return_index=True)
        new_value_array = new_value_array[ind]
    else:
        uniq_final = new_uniq_array

    print(min(uniq_final))
    print(max(uniq_final))
    
    print(len(new_value_array), len(uniq_final))

    print(uniq_final)

    for ii in range(len(uniq_final)-1):
        assert(uniq_final[ii+1]-uniq_final[ii]>0)

    equatorial_moc = HealpixMap(new_value_array, uniq_final, density=True)
    return equatorial_moc
###### plotting functions for sphere (s2), employing a flexible grid to save computing while still having smooth contours

def _transform_to_world(ax_object, coords, projection_type="zen_azi"):
    """
    Does the coordinate transformation to world coordaintes given a certain projection_type
    """

    if(projection_type=="zen_azi"):
        healpy_phi_theta_coords=coords*180.0/numpy.pi

        # theta
        healpy_phi_theta_coords[:,1]=90.0-healpy_phi_theta_coords[:,0]

        # phi
        healpy_phi_theta_coords[:,0]=coords[:,1]*180.0/numpy.pi
    else:
        raise Exception("Unknown projection type ", projection_type)

    world_coords=ax_object.wcs.all_world2pix(healpy_phi_theta_coords,1)

    return world_coords

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
        
        ### flip = "geo" -> left to right, flip="astro" -> right to left
        super().__init__(fig, rect,
                         rot = fixed_rot,
                         flip="geo", # this ordering assures that azimuth spans from left to right
                         **kwargs)

    def graticule(self, 
                  dpar = 45, 
                  dmer = 60, 
                  grid = True,
                  ticks = True, 
                  show_zenith_label=True,
                  show_zenith_axis=True,
                  zenith_axislabel_minpad=2.0,
                  azimuth_axislabel_minpad=0.0,
                  show_azimuth_label=True,
                  show_azimuth_axis=True,
                  azi_label_color="white",
                  tick_format = 'd', 
                  frame = True, 
                  zen_azi_mode="zen_azi",
                  text_size=12,
                  **kwargs):
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

        if(grid):
            self.grid(grid, **kwargs)

        if(ticks):
            self.coords[0].set_ticks(spacing=dmer * u.deg)
            self.coords[1].set_ticks(spacing=dpar * u.deg)



        if(show_azimuth_axis):
            self.coords[0].set_major_formatter(tick_format)
            self.coords[0].set_ticklabel_visible(ticks)
            self.coords[0].set_ticklabel(color='white', size=text_size)

        if(show_zenith_axis):
            self.coords[1].set_major_formatter(tick_format)
            self.coords[1].set_ticklabel_visible(ticks)
            self.coords[1].set_ticklabel(color='black', size=text_size)

        if frame:
            self.coords.frame.set_linewidth(kwargs.get('linewidth', 3))
            self.coords.frame.set_color(kwargs.get('color', 'black'))
        else:
            self.coords.frame.set_linewidth(0)            
            
        if(zen_azi_mode=="zen_azi"):
            if(show_azimuth_axis):
                if(show_azimuth_label):
                    self.coords[0].set_axislabel("azimuth [deg]", color=azi_label_color, minpad=azimuth_axislabel_minpad) # shift the label a little to the left

            if(show_zenith_axis):

                ## monkeypatching ticklabels for zenith
                self.coords[1].ticklabels.add = add.__get__(self.coords[1].ticklabels)

                if(show_zenith_label):
                    self.coords[1].set_axislabel("zenith [deg]", minpad=zenith_axislabel_minpad) # shift the label a little to the left
                
                
        
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

    def proj_plot(self, *args, **kwargs):
        """
        Calls matplotlib.plot in world coordinates, and does an internal transformation first.
        Internally the healpy ax uses coordinates similar to dec/ra (-90 - 90 / 0-360), so have to take of that here.
        """
        assert(len(args)>=2)

        projection_type="zen_azi"

        if("projection_type" in kwargs):
            projection_type=kwargs["projection_type"]

        assert(projection_type=="zen_azi"), "For now only zen_azi is supported"

        x=args[0]
        y=args[1]

        new_x=x
        new_y=y
        if(type(x)==list):
            new_x=numpy.array(x)
        if(type(y)==list):
            new_y=numpy.array(y)

        assert(new_x.ndim==1)
        assert(new_y.ndim==1)

        combined_coords=numpy.concatenate([new_x[:,None], new_y[:,None]], axis=1)

        world_coords=_transform_to_world(self, combined_coords, projection_type=projection_type)

        further_args=args[2:]
        
        new_kwargs=kwargs.copy()
        if("projection_type" in new_kwargs):
            del new_kwargs["projection_type"]

        return self.plot(world_coords[:,0], world_coords[:,1], *further_args, **new_kwargs)

        
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
                  show_zenith_label=True,
                  show_zenith_axis=True,
                  zenith_axislabel_minpad=4.0,
                  azimuth_axislabel_minpad=0.0,
                  show_azimuth_label=True,
                  show_azimuth_axis=False,
                  azi_label_color="white",
                  tick_format = 'd', 
                  frame = True, 
                  zen_azi_mode="zen_azi",
                  text_size=12,
                  **kwargs):
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
            self.coords[0].set_ticklabel(color='white', size=text_size)

        if(show_zenith_axis):
            self.coords[1].set_major_formatter(tick_format)
            self.coords[1].set_ticklabel_visible(ticks)
            self.coords[1].set_ticklabel(color='black', size=text_size)

        if frame:
            self.coords.frame.set_linewidth(kwargs.get('linewidth', 3))
            self.coords.frame.set_color(kwargs.get('color', 'black'))
        else:
            self.coords.frame.set_linewidth(0)    

       
        ## monkeypatching ticklabels for zenith
        self.coords[1].ticklabels.add = add.__get__(self.coords[1].ticklabels)

        if(show_azimuth_axis):
            if(show_azimuth_label):
                self.coords[0].set_axislabel("azimuth [deg]", color=azi_label_color, minpad=azimuth_axislabel_minpad) # shift the label a little to the left

        if(show_zenith_axis):
            if(show_zenith_label):
                self.coords[1].set_axislabel("zenith [deg]", minpad=zenith_axislabel_minpad) # shift the label a little to the left


    def proj_plot(self, *args, **kwargs):
        """
        Calls matplotlib.plot in world coordinates, and does an internal transformation first.
        Internally the healpy ax uses dec/ra, so have to take of that here.
        """
        assert(len(args)>=2)

        projection_type="zen_azi"

        if("projection_type" in kwargs):
            projection_type=kwargs["projection_type"]

        assert(projection_type=="zen_azi"), "For now only zen_azi is supported"

        x=args[0]
        y=args[1]

        new_x=x
        new_y=y
        if(type(x)==list):
            new_x=numpy.array(x)
        if(type(y)==list):
            new_y=numpy.array(y)

        assert(new_x.ndim==1)
        assert(new_y.ndim==1)

        combined_coords=numpy.concatenate([new_x[:,None], new_y[:,None]], axis=1)

        world_coords=_transform_to_world(self, combined_coords, projection_type=projection_type)

        further_args=args[2:]
        
        new_kwargs=kwargs.copy()
        if("projection_type" in new_kwargs):
            del new_kwargs["projection_type"]

        return self.plot(world_coords[:,0], world_coords[:,1], *further_args, **new_kwargs)

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
    
    print(used_samples[:,0].min(),used_samples[:,0].max(), numpy.pi-used_samples[:,0].max())

    sample_pix = mhealpy.ang2pix(mhealpy.MAX_NSIDE, used_samples[:,0], used_samples[:,1], nest = True)
    
    moc_map=HealpixMap.moc_histogram(mhealpy.MAX_NSIDE, sample_pix, max_entries_per_pixel, nest=True)
    
    pix_ids=numpy.arange(moc_map.npix)
    ang_vals=moc_map.pix2ang(pix_ids)
    ang_vals=numpy.concatenate([ang_vals[0][:,None], ang_vals[1][:,None]], axis=1)
   
    per_pixel_nside, _ = mhealpy.uniq2nest(moc_map.pix2uniq(pix_ids))
    per_pixel_areas= mhealpy.nside2pixarea(per_pixel_nside)
    
    return ang_vals, per_pixel_areas, moc_map

def get_multiresolution_evals(pdf, 
                              conditional_input=None,
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

    data_summary_repeated=None
    if(conditional_input is not None):
        data_summary_repeated=conditional_input

        if(type(conditional_input)==list):  
            if(conditional_input[0].ndim==2):
                assert(conditional_input[0].shape[0]==1), "Only a single conditional input item must be given!"
            data_summary_repeated=[ci.repeat_interleave(samplesize, dim=0) if ci.ndim==2 else ci[None,:].repeat_interleave(samplesize, dim=0) for ci in conditional_input]
        else:
            if(conditional_input.ndim==2):
                assert(conditional_input.shape[0]==1), "Only a single conditional input item must be given!"
            data_summary_repeated=conditional_input.repeat_interleave(samplesize, dim=0) if conditional_input.ndim==2 else conditional_input[None,:].repeat_interleave(samplesize, dim=0)
        
    samples,_,_,_=pdf.sample(samplesize=samplesize, conditional_input=data_summary_repeated)

    eval_positions, eval_areas, moc_map=get_meshed_positions_and_areas(samples,max_entries_per_pixel=max_entries_per_pixel)

    assert(pdf.pdf_defs_list[sub_pdf_index]=="s2"), ("Trying to get multiresolution for s2 subdimension, but subdimension %d is of type %s" % (sub_pdf_index, pdf.pdf_defs_list[sub_pdf_index]))

    if(use_density_if_possible and (sub_pdf_index==0)):
        xyz_positions=pdf.transform_target_into_returnable_params(torch.from_numpy(eval_positions).to(samples))

        if(data_summary_repeated is not None):
            
            moc_size=xyz_positions.shape[0]
            if(type(data_summary_repeated)==list):  
                assert(moc_size<=data_summary_repeated[0].shape[0])
                data_summary_repeated=[ci[:moc_size] for ci in data_summary_repeated]
            else:
                assert(moc_size<=data_summary_repeated.shape[0])
                data_summary_repeated=data_summary_repeated[:moc_size]

        log_pdf,_,_=pdf(xyz_positions, force_embedding_coordinates=True, conditional_input=data_summary_repeated)
        
        log_pdf=log_pdf.cpu().detach().numpy()
        pdf_evals=numpy.exp(log_pdf)

    else:

        
        ipix_vals=moc_map.ang2pix(samples[:,0].cpu().numpy(), samples[:,1].cpu().numpy())
        sorted_pixel_vals=numpy.sort(ipix_vals)

        unique_indices, counts=numpy.unique(sorted_pixel_vals, return_counts=True)
       
        pdf_evals=numpy.zeros(len(eval_areas))
        pdf_evals[unique_indices]=counts/float(sum(counts))#eval_areas[unique_indices]
        pdf_evals=pdf_evals/eval_areas

        # no log_pdf in sample_based evaluation
        log_pdf=None

    return eval_positions, log_pdf, pdf_evals, eval_areas, moc_map


def plot_multiresolution_healpy(pdf,
                                fig=None, 
                                ax_to_plot=None,
                                samplesize=10000,
                                conditional_input=None, 
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
                                zoom_contained_prob_mass=0.97,
                                zoom_correction=1.0,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False): 
    
    """
    Visualizes an S2 pdf, or a certain S2 subpart of a PDF using an adaptive healpix grid from mhealpy. Useful if the PDF
    is very small and higher nside becomes computationally expensive. Can also overlay smooth contours on this irregular grid
    using the *meander* package.

    """
    assert("s2" in pdf.pdf_defs_list ), "Requires that at least one s2 sub-manifold exists."

    eval_positions, _, pdf_evals, eval_areas, moc_map=get_multiresolution_evals(pdf, 
                                                                             sub_pdf_index=sub_pdf_index,
                                                                    samplesize=samplesize,
                                                                    conditional_input=conditional_input,
                                                                    max_entries_per_pixel=max_entries_per_pixel,
                                                                    use_density_if_possible=use_density_if_possible)
    
    if( (~numpy.isfinite(eval_positions)).sum()>0):
        raise Exception("No finite eval positions?!")

    if( (pdf_evals==numpy.inf).sum()>0):
        raise Exception("INFs!?")

    if( (pdf_evals==-numpy.inf).sum()>0):
        raise Exception("- INFs!?")

    ## correct nans
    pdf_evals=numpy.where( (~numpy.isfinite(pdf_evals)), 1e-20, pdf_evals)
    
    ax=_plot_multiresolution_healpy(eval_positions,
                                    pdf_evals,
                                    eval_areas,
                                    moc_map=moc_map,
                                    fig=fig, 
                                    ax_to_plot=ax_to_plot,
                                    draw_pixels=draw_pixels,
                                    log_scale=log_scale,
                                    cbar=cbar,
                                    cbar_kwargs=cbar_kwargs,
                                    graticule=graticule,
                                    graticule_kwargs=graticule_kwargs,
                                    draw_contours=draw_contours,
                                    contour_probs=contour_probs,
                                    contour_colors=contour_colors, # None -> pick colors from color scheme
                                    zoom=zoom,
                                    zoom_contained_prob_mass=zoom_contained_prob_mass,
                                    zoom_correction=zoom_correction,
                                    projection_type=projection_type, # zen_azi or dec_ra
                                    declination_trafo_function=declination_trafo_function,
                                    show_grid=show_grid) 

    return ax

def _plot_multiresolution_healpy(eval_positions,
                                pdf_evals,
                                eval_areas,
                                moc_map=None,
                                fig=None, 
                                ax_to_plot=None,
                                draw_pixels=True,
                                log_scale=True,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                zoom_contained_prob_mass=0.97,
                                zoom_correction=1.0,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False,
                                mjd=None,
                                lon=None,
                                lat=None,
                                zenazi_to_decra_fn=None):  # function that accepts zen,azi, lon, lat, mjd and spits out dec,ra
    
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
        assert(len(eval_positions)==moc_map.npix)

    if(zoom):
        
        tot_sums=pdf_evals*eval_areas
        sorta=numpy.argsort(tot_sums)[::-1] # large to small
        ## add largest areas
        
        contained_mask=numpy.cumsum(tot_sums[sorta])<zoom_contained_prob_mass
        contained_positions=eval_positions[sorta][contained_mask]
        
        if(contained_mask.sum()>0):
            # calculate zoom center

            xyz_positions,_=sphere_base.sphere_base(dimension=2).spherical_to_eucl_embedding(torch.from_numpy(eval_positions), 0.0)
            xyz_positions=xyz_positions.cpu().numpy()
            
            weighted_sum=numpy.mean(pdf_evals[:,None]*eval_areas[:,None]*xyz_positions,axis=0)
            mean_coords=weighted_sum/numpy.sqrt(numpy.sum(weighted_sum**2))
            mean_coords,_=sphere_base.sphere_base(dimension=2).eucl_to_spherical_embedding(torch.from_numpy(mean_coords[None,:]), 0.0)
            mean_coords=mean_coords[0].cpu().numpy()

            ## max position
            max_index=numpy.argmax(pdf_evals)
            max_coords=eval_positions[max_index]

            # calculate zoom diameter
            zen_range=numpy.quantile(contained_positions[:,0], [0.05,0.95])
            azi_range=numpy.quantile(contained_positions[:,1], [0.05,0.95])


            # zoom can be on 2pi boundary.. make sure this is taken into account in diameter
            #### this only works if the mean is accurate enough
            
            if( (mean_coords[1]>=azi_range[0]) and (mean_coords[1]<=azi_range[1])):
                azi_diff=azi_range[1]-azi_range[0]
            else:
                azi_diff=azi_range[0]+2*numpy.pi-azi_range[1]

            azi_diff=azi_range[1]-azi_range[0]
            
            zen_diff=zen_range[1]-zen_range[0]

            # take larger value as approximate diameter
            zoom_diameter=max(zen_diff, azi_diff)*2.0*zoom_correction

            print("mean coords before")
            if(projection_type=="dec_ra"):

                if(zenazi_to_decra_fn is not None):
                    transfer_fn=zenazi_to_decra_fn

                else:
                    transfer_fn=local_to_j2000_batch
                assert(mjd is not None)
                assert(lat is not None)
                assert(lon is not None)

                dec, ra=transfer_fn(mean_coords[0], mean_coords[1], mjd, numpy.deg2rad(lat), numpy.deg2rad(lon))

                #print("lat deglat",lat, numpy.deg2rad(lat))

                #quasi_ra=ra
                
                print("DEC", dec)
                quasi_dec=numpy.pi/2.0-dec
                

                print("quasi decara", quasi_dec, ra)
                print("MJD .... ", mjd)
                ## transform to quasi zen/azi (back to 0-180/0-360)
                mean_coords=[quasi_dec, ra]

                

        else:
            zoom_diameter=2*numpy.pi
        
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
            single_ax=replace_axes_with_gridspec(ax_to_plot, new_layout=(1,1))

            try:
                if(zoom):
                    ax=OrthviewAzimuth(fig, single_ax.get_position(), zoom_center=mean_coords, zoom_diameter=zoom_diameter)
                else:
                    ax=MollviewAzimuth(fig, single_ax.get_position())
            except:
                print("ORTH/MOLLview failed (due to nans probably...")
                return single_ax
            fig.add_axes(ax)
            single_ax.remove() # remove original ax_to_plot
           
        else:
            ax=ax_to_plot
        
    if(draw_pixels):
        moc_map.density(density=True)
        moc_map[numpy.arange(len(eval_positions))]=pdf_evals
        print("MOC MAP pdf evals... ", min(pdf_evals), max(pdf_evals), "non fin sum", (~numpy.isfinite(pdf_evals).sum()), file=sys.stderr)
        
        ### replace moc map if drawing in dec/RA

        if(projection_type=="dec_ra"):
            if(zenazi_to_decra_fn is not None):
                transfer_fn=zenazi_to_decra_fn

            else:
                transfer_fn=local_to_j2000_batch

            moc_map=reproject_moc_local_to_equatorial(moc_map, lat, lon, mjd, transfer_fn=transfer_fn)
            
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
            default_cbar_kwargs["label"]=r"PDF value"

            for extra_kwarg in cbar_kwargs:
                default_cbar_kwargs[extra_kwarg]=cbar_kwargs[extra_kwarg]
            try:
                fig.colorbar(img, ax = ax, **default_cbar_kwargs)
            except:
                print("ERror in colorbar......", file=sys.stderr)
                print("MOC MAP pdf evals... ", min(pdf_evals), max(pdf_evals), file=sys.stderr)
                return ax
    contours=None
    if(draw_contours):
        if(contour_probs is not None):

            used_contour_colors=matplotlib.cm.tab10.colors

            if(contour_colors is not None):
                used_contour_colors=contour_colors

            assert(len(used_contour_colors)>=len(contour_probs))

            used_contour_colors=used_contour_colors[:len(contour_probs)]

            ret=ContourGenerator(
                                 ax, 
                                 "zen_azi",
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

            ## max out at 60 degrees for full sky
            graticule_default_kwargs["dmer"]=min(desirable_dist_between_graticules, 60.0)
            graticule_default_kwargs["dpar"]=min(desirable_dist_between_graticules, 60.0)

        for extra_kwarg in graticule_kwargs:
            graticule_default_kwargs[extra_kwarg]=graticule_kwargs[extra_kwarg]
       
        ax.graticule(**graticule_default_kwargs)

    if(show_grid):
        moc_map.plot_grid(ax, linewidth = .1, color = 'white');


    return ax

