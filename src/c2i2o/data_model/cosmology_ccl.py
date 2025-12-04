



from pydantic import BaseModel, Field, create_model




class Cosmology_CCL(BaseModel):

    Omega_c: float = Field(..., description="Cold dark matter density fraction", ge=0., le=1.)
    Omega_b: float = Field(..., description="Baryonic matter density fraction", ge=0., le=1.)    
    Omega_k: float = Field(0.0, description="Curvature density fraction density fraction", ge=-1., le=1.)    
    Omega_g: float|None = Field(None, description="Density in relativistic species except massless neutrinos", ge=0., le=1.)    
    h: float = Field(0.0, description="Hubble constant divided by 100 km/s/Mpc", ge=0., le=2.)    
    n_s: float = Field(0.96, description="Primordial scalar perturbation spectral index", ge=0., le=2.)    
    sigma8: float|None = Field(0.96, description="Variance of matter density perturbations at an 8 Mpc/h scale.", ge=0., le=2.) 
    A_s: float|None = Field(0.96, description="Power spectrum normalization.", ge=0., le=2.) 
    Neff: float = Field(3.044, description="Effective number of massless neutrinos present.", ge=3., le=3.2) 
    m_nu: float = Field(0.0, description="Mass in eV of the massive neutrinos present.")
    w0: float = Field(-1.0, description="First order term of dark energy equation of state.", ge=-2. le=1.)
    wa: float = Field(0.0, description="Second order term of dark energy equation of state.", ge=-2. le=2.)
    T_CMB: float = Field(2.7255, description="The CMB temperature today.")    
    T_ncdm: float= Field(0.71611, description="Non-CDM temperature in units of photon temperature.")
    
    mass_split: str = Field('normal', description="Type of massive neutrinos.")
    transfer_function: str = Field('boltzmann_camb', description="The transfer function to use.")
    matter_power_spectrum: str = Field('halofit', description="The matter power spectrum to use.")

    baryonic_effects: Baryons | None = Field(None, description="The baryonic effects model to use.")
    mg_parametrization: ModifiedGravity=None, Field(None, description="The modified gravity parametrization to use.")

    extra_parameters: dict = {}


    def build_cosmology(self) -> pyccl.cosmology.Cosmology:

        kwargs_dict = self.dict()

        baryonic_effects_ = kwargs_dict.pop('baryonic_effects')
        if baryonic_effects_ is not None:
            kwargs_dict['baryonic_effects'] = self.build_baryonic_effects(baryonic_effects_)

        mg_parametrization_ = kwargs_dict.pop('mg_parametrization')
        if mg_parametrization_ is not None:
            kwargs_dict['mg_parametrization'] = self.build_mg_parametrization(mg_parametrization_)
        
        return pyccl.cosmology.Cosmology(**kwargs_dict)
        
        

    
