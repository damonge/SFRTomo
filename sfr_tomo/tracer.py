import pyccl as ccl
from scipy.interpolate import interp1d
import numpy as np


class IvTracer(ccl.Tracer):
    """Specific :class:`Tracer` associated with the cosmic infrared
    background intensity at a specific frequency v (Iv). 
    The radial kernel for this tracer is

    .. math::
       W(\\chi) = \\frac{\\chi^{2} S_\\nu^{eff}}{K}.

    Any angular power spectra computed with this tracer, should use
    a three-dimensional power spectrum involving the CIB emissivity
    density in units of
    :math:`{\\rm Jy}\\,{\\rm Mpc}^{-1}\\,{\\rm srad}^{-1}` (or
    multiples thereof).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        snu_z (array): effective source flux for one frequency in units of
            :math:`{\\rm Jy}\\,{\\rm L_{Sun}}^{-1}\\.
        z_arr (array): redshift values to compute chi_z
        zmin (float): minimum redshift down to which we define the
            kernel.
        zmax (float): maximum redshift up to which we define the
            kernel. zmax = 6 by default (reionization)
    """
    def __init__(self, cosmo, snu_z, z_arr, z_min=0., z_max=6.):
        self.chi_max = ccl.comoving_radial_distance(cosmo, 1./(1+z_max))
        self.chi_min = ccl.comoving_radial_distance(cosmo, 1./(1+z_min))
        chi_z = ccl.comoving_radial_distance(cosmo, 1./(1+z_arr))
        # Transform to MJy (units in data)
        snu_inter = interp1d(chi_z, snu_z*1E-6, kind='linear',
                             bounds_error=False, fill_value="extrapolate")
        chi_arr = np.linspace(self.chi_min, self.chi_max, len(snu_z))
        snu_arr = snu_inter(chi_arr)
        K = 1.0e-10  # Kennicutt constant in units of M_sun/yr/L_sun
        w_arr = chi_arr**2*snu_arr/K
        self._trc = []
        self.add_tracer(cosmo, kernel=(chi_arr, w_arr))
