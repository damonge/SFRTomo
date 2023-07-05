import numpy as np
from scipy.integrate import simps
import pyccl as ccl


class HaloProfileCIBM21(ccl.halos.HaloProfile):
    """ CIB profile implementing the model by Maniyar et al.
    (A&A 645, A40 (2021)).

    The parametrization for the mean profile is:

    .. math::
        \\rho_{\\rm SFR}(z) = \\rho_{\\rm SFR}^{\\rm cen}(z)+
        \\rho_{\\rm SFR}^{\\rm sat}(z),

    where the star formation rate (SFR) density from centrals and satellites is
    modelled as:

    .. math::
        \\rho_{\\rm SFR}^{\\rm cen}(z) = \\int_{}^{}\\frac{dn}{dm},
        SFR(M_{\\rm h},z)dm,

    .. math::
        \\rho_{\\rm SFR}^{\\rm sat}(z) = \\int_{}^{}\\frac{dN}{dm},
        (\\int_{M_{\\rm min}}^{M}\\frac{dN_{\\rm sub}}{dm}SFR(M_{\\rm sub},z)dm),
        dm.

    Here, :math:`dN_{\\rm sub}/dm` is the subhalo mass function,
    and the SFR is parametrized as

    .. math::
        SFR(M,z) = \\eta(M,z)\\,
        BAR(M,z),

    where the mass dependence of the efficiency :math:'\\eta' is lognormal

    .. math::
        \\eta(M,z) = \\eta_{\\rm max},
        \\exp\\left[-\\frac{\\log_{10}^2(M/M_{\\rm eff})},
        {2\\sigma_{LM}^2(z)}\\right],
        
    with :math:'\\sigma_{LM}' defined as the redshift dependant logarithmic scatter in mass
    
    .. math::
        \\sigma_{LM}(z) = \\left\\{
        \\begin{array}{cc}
           \\sigma_{LM0} & M < M_{\\rm eff} \\\\
           \\sigma_{LM0} - \\tau max(0,z_{\\rm c}-z) & M \\geq M_{\\rm eff}
        \\end{array}
        \\right.,

    and :math:'BAR' is the Baryon Accretion Rate

    .. math::
        BAR(M,z) = \\frac{\\Omega_{\\rm b}}{\\Omega_{\\rm m}},
        MGR(M,z),

    where :math:'MGR' is the Mass Growth Rate
    
        MGR(M,z) = 46.1\\left(\\frac{M}{10^{12}M_{\\odot}}\\right)^{1.1},
        \\left(1+1.11z\\right)\sqrt{\\Omega_{\\rm m}(1+z)^{3}+\\Omega_{\\rm \\Lambda}}.

    Args:
        cosmo (:obj:`Cosmology`): cosmology object containing
            the cosmological parameters
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        log10meff (float): log10 of the most efficient mass.
        etamax (float) : star formation efficiency of the most efficient mass
        sigLM0 (float): logarithmic scatter in mass.
        tau (float) : rate at which :math:'\\sigma_{LM}' evolves with redshift.
        zc (float) : redshift below which :math:'\\sigma_{LM}' evolves with redshift.
        Mmin (float): minimum subhalo mass.
    """
    name = 'CIBM21'
    _lnten = 2.30258509299

    def __init__(self, cosmo, c_M_relation, log10meff=12.94, etamax=0.42,
                 sigLM0=1.75, tau=1.17, zc=1.5, Mmin=1E5, fsub=0.134,
                 fast_integ=False,
                 log10M0=11.34, log10Mz=0.692,
                 eps0=0.005, epsz=0.689, beta0=3.344, betaz=-2.079,
                 gamma0=0.966, gammaz=0.0, emerge=False):
        if not isinstance(c_M_relation, ccl.halos.Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")
            
        self.Omega_b = cosmo['Omega_b']
        self.Omega_m = cosmo['Omega_c'] + cosmo['Omega_b']
        self.Omega_L = 1 - cosmo['Omega_m']
        self.l10meff = log10meff
        self.etamax = etamax
        self.sigLM0 = sigLM0
        self.tau = tau
        self.zc = zc
        self.Mmin = Mmin
        self.fsub = fsub
        self.fast_integ = fast_integ
        self.log10M0 = log10M0
        self.log10Mz = log10Mz
        self.eps0 = eps0
        self.epsz = epsz
        self.beta0 = beta0
        self.betaz = betaz
        self.gamma0 = gamma0
        self.gammaz = gammaz
        self.emerge = emerge
        self.pNFW = ccl.halos.HaloProfileNFW(c_M_relation)
        super(HaloProfileCIBM21, self).__init__()

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent):
        """Subhalo mass function of Tinker & Wetzel (2010ApJ...719...88T)

        Args:
            Msub (float or array_like): sub-halo mass (in solar masses).
            Mparent (float): parent halo mass (in solar masses).

        Returns:
            float or array_like: average number of subhalos.
        """
        return 0.30*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)


    def update_parameters(self, log10meff=None, etamax=None,
                          sigLM0=None, tau=None, zc=None, Mmin=None,
                          fsub=None, log10M0=None, log10Mz=None,
                          eps0=None, epsz=None, beta0=None, betaz=None,
                          gamma0=None, gammaz=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.

        Args:
            log10meff (float): log10 of the most efficient mass.
            etamax (float) : star formation efficiency of the most efficient mass
            sigLM0 (float): logarithmic scatter in mass.
            tau (float) : rate at which :math:'\\sigma_{LM}' evolves with redshift.
            zc (float) : redshift below which :math:'\\sigma_{LM}' evolves with redshift.
            Mmin (float): minimum subhalo mass (in solar masses).
        """
        if log10meff is not None:
            self.l10meff = log10meff
        if etamax is not None:
            self.etamax = etamax
        if sigLM0 is not None:
            self.sigLM0 = sigLM0
        if tau is not None:
            self.tau = tau
        if zc is not None:
            self.zc = zc
        if Mmin is not None:
            self.Mmin = Mmin
        if fsub is not None:
            self.fsub = fsub
        if log10M0 is not None:
            self.log10M0 = log10M0
        if log10Mz is not None:
            self.log10Mz = log10Mz
        if eps0 is not None:
            self.eps0 = eps0
        if epsz is not None:
            self.epsz = epsz
        if beta0 is not None:
            self.beta0 = beta0
        if betaz is not None:
            self.betaz = betaz
        if gamma0 is not None:
            self.gamma0 = gamma0
        if gammaz is not None:
            self.gammaz = gammaz

    def sigLM(self, M, a):
        z = 1/a - 1
        if hasattr(M, "__len__"):
            sig = np.zeros_like(M)
            smallM = np.log10(M) < self.l10meff
            sig[smallM] = self.sigLM0
            sig[~smallM] = self.sigLM0 - self.tau * max(0, self.zc-z)
            return sig
        else:
            if np.log10(M) < self.l10meff:
                return self.sigLM0
            else :
                return self.sigLM0 - self.tau * max(0, self.zc-z)

    def _efficiency(self, M, a):
        if self.emerge:
            t = 1-a
            M1 = 10**(self.log10M0+self.log10Mz*t)
            eps = self.eps0+self.epsz*t
            beta = self.beta0+self.betaz*t
            gamma = self.gamma0+self.gammaz*t
            mr = M/M1
            eta = 2*eps/(mr**gamma+1/mr**beta)
        else:
            eta = self.etamax * np.exp(-0.5*((np.log(M) -
                                              self._lnten*self.l10meff)/
                                             self.sigLM(M, a))**2)
        return eta

    def _SFR(self, M, a):
        z = 1/a - 1
        # Efficiency - eta
        eta = self._efficiency(M, a)
        # Baryonic Accretion Rate - BAR
        MGR = 46.1 * (M*1e-12)**1.1 * (1+1.11*z) * np.sqrt(self.Omega_m*(1+z)**3 + self.Omega_L)
        BAR = self.Omega_b/self.Omega_m * MGR
        return eta * BAR

    def _SFRcen(self, M, a):
        SFRcen = self._SFR(M*(1-self.fsub), a)
        return SFRcen

    def _SFRsat(self, M, a):
        if self.fast_integ:
            SFRsat = np.zeros_like(M)
            goodM = M >= self.Mmin
            M_use = (1-self.fsub)*M[goodM, None]
            nm = max(2, 3*int(np.log10(np.max(M_use)/self.Mmin)))
            Msub = np.geomspace(self.Mmin, np.max(M_use), nm+1)[None, :]
            # All these arrays are of shape [nM_parent, nM_sub]
            dnsubdlnm = self.dNsub_dlnM_TinkerWetzel10(Msub, M_use)
            SFRI = self._SFR(Msub.flatten(), a)[None, :]
            SFRII = self._SFR(M_use, a)*Msub/M_use
            Ismall = SFRI < SFRII
            SFR = SFRI*Ismall + SFRII*(~Ismall)
            integ = dnsubdlnm*SFR*(M_use >= Msub)
            SFRsat[goodM] = simps(integ, x=np.log(Msub))
            return SFRsat
            
        SFRsat = np.zeros_like(M) 
        for iM, Mhalo in enumerate(M*(1-self.fsub)):
            if Mhalo > self.Mmin:
                nm = max(2, int(np.log10(Mhalo/self.Mmin)*10))
                Msub = np.geomspace(self.Mmin, Mhalo, nm+1)
                dnsubdlnm = self.dNsub_dlnM_TinkerWetzel10(Msub, Mhalo)
                SFRI = self._SFR(Msub, a)
                SFRII = self._SFR(Mhalo, a)*(Msub/Mhalo)
                SFRsub = np.minimum(SFRI, SFRII)
                integ = dnsubdlnm*SFRsub
                SFRsat[iM] = simps(integ, x=np.log(Msub))
        return SFRsat
    
    def _real(self, cosmo, r, M, a, mass_def):
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        SFRs = self._SFRsat(M_use, a)
        ur = self.pNFW._real(cosmo, r_use, M_use,
                             a, mass_def)/M_use[:, None]

        prof = SFRs[:, None]*ur
        
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        SFRc = self._SFRcen(M_use, a)
        SFRs = self._SFRsat(M_use, a)
        uk = 1#self.pNFW._fourier(cosmo, k_use, M_use,
              #                  a, mass_def)/M_use[:, None]
        prof = SFRc[:, None]+SFRs[:, None]*uk
        
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        
        SFRc = self._SFRcen(M_use, a)
        SFRs = self._SFRsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use,
                                a, mass_def)/M_use[:, None]

        prof = SFRs[:, None]*uk
        prof = 2*SFRc[:, None]*prof + prof**2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
