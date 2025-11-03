"""
SACRAMENTO Soil Moisture Accounting Model (SAC-SMA)

The Sacramento model is a continuous soil moisture accounting model developed by
the National Weather Service (NWS) for river forecasting in the United States.
It was originally developed by Burnash et al. (1973) for the Sacramento River basin.

Mathematical Foundation:
========================

The model divides the soil profile into upper and lower zones:

1. Upper Zone:
   - Upper Zone Tension Water (UZTWC): supplies evapotranspiration
   - Upper Zone Free Water (UZFWC): generates surface and interflow runoff
   
2. Lower Zone:
   - Lower Zone Tension Water (LZTWC): supplies evapotranspiration
   - Lower Zone Primary Free Water (LZFPC): generates primary baseflow
   - Lower Zone Supplementary Free Water (LZFSC): generates supplementary baseflow

Water Balance Equations:
------------------------

1. Evapotranspiration:
   - From UZTWC: E1 = min(UZTWC, PET * UZTWC/UZTWM)
   - From UZFWC: E2 = min(UZFWC, (PET - E1) * UZFWC/UZFWM)
   - From LZTWC: E3 = min(LZTWC, (PET - E1 - E2) * LZTWC/LZTWM)
   - From LZFPC: E4 = min(LZFPC, (PET - E1 - E2 - E3) * LZFPC/(LZFPM+LZFSM))
   - From LZFSC: E5 = min(LZFSC, (PET - E1 - E2 - E3 - E4) * LZFSC/(LZFPM+LZFSM))

2. Runoff Generation:
   Surface runoff: ROIMP = P * PCTIM (impervious area)
   
   Interflow: RSSUR = UZFWC * UZK if UZFWC > 0
   
   Primary baseflow: RBFP = LZFPC * LZPK if LZFPC > 0
   
   Supplementary baseflow: RBFS = LZFSC * LZSK if LZFSC > 0

3. Percolation:
   PERC = LZFPM * (LZFPC/LZFPM) * (1 + ZPERC * (ADIMC/UZTWM + LZTWC/LZTWM))

Parameters:
-----------
UZTWM : Upper zone tension water maximum (mm)
UZFWM : Upper zone free water maximum (mm)
LZTWM : Lower zone tension water maximum (mm)
LZFPM : Lower zone primary free water maximum (mm)
LZFSM : Lower zone supplementary free water maximum (mm)
UZK   : Upper zone free water lateral depletion rate (1/day)
LZPK  : Lower zone primary free water depletion rate (1/day)
LZSK  : Lower zone supplementary free water depletion rate (1/day)
ZPERC : Maximum percolation rate coefficient
REXP  : Exponent of the percolation equation
PCTIM : Fraction of impervious area
ADIMP : Additional impervious area
PFREE : Fraction of water percolating directly to lower zone free water

References:
-----------
Burnash, R.J.C., Ferral, R.L., & McGuire, R.A. (1973). 
A generalized streamflow simulation system: Conceptual models for digital computers. 
US Department of Commerce, National Weather Service, and State of California.

Author: Burnash et al. (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import Dict
import warnings


class SacramentoModel:
    """
    Sacramento Soil Moisture Accounting (SAC-SMA) model implementation.
    
    This is a simplified version of the full SAC-SMA model.
    
    Parameters:
    -----------
    UZTWM : float
        Upper zone tension water capacity (mm, default: 80)
    UZFWM : float
        Upper zone free water capacity (mm, default: 40)
    LZTWM : float
        Lower zone tension water capacity (mm, default: 150)
    LZFPM : float
        Lower zone primary free water capacity (mm, default: 100)
    LZFSM : float
        Lower zone supplementary free water capacity (mm, default: 50)
    UZK : float
        Upper zone depletion rate (1/day, default: 0.3)
    LZPK : float
        Lower zone primary depletion rate (1/day, default: 0.01)
    LZSK : float
        Lower zone supplementary depletion rate (1/day, default: 0.05)
    ZPERC : float
        Maximum percolation rate (default: 40)
    REXP : float
        Exponent of percolation equation (default: 2.0)
    PCTIM : float
        Impervious fraction (0-1, default: 0.01)
    ADIMP : float
        Additional impervious area (0-1, default: 0.0)
    PFREE : float
        Fraction percolating directly to lower zone (0-1, default: 0.1)
    """
    
    def __init__(self,
                 UZTWM: float = 80.0,
                 UZFWM: float = 40.0,
                 LZTWM: float = 150.0,
                 LZFPM: float = 100.0,
                 LZFSM: float = 50.0,
                 UZK: float = 0.3,
                 LZPK: float = 0.01,
                 LZSK: float = 0.05,
                 ZPERC: float = 40.0,
                 REXP: float = 2.0,
                 PCTIM: float = 0.01,
                 ADIMP: float = 0.0,
                 PFREE: float = 0.1):
        
        # Model parameters
        self.UZTWM = UZTWM  # Upper zone tension water max
        self.UZFWM = UZFWM  # Upper zone free water max
        self.LZTWM = LZTWM  # Lower zone tension water max
        self.LZFPM = LZFPM  # Lower zone primary free water max
        self.LZFSM = LZFSM  # Lower zone supplementary free water max
        self.UZK = UZK      # Upper zone depletion
        self.LZPK = LZPK    # Lower zone primary depletion
        self.LZSK = LZSK    # Lower zone supplementary depletion
        self.ZPERC = ZPERC  # Percolation rate
        self.REXP = REXP    # Percolation exponent
        self.PCTIM = PCTIM  # Impervious area
        self.ADIMP = ADIMP  # Additional impervious
        self.PFREE = PFREE  # Direct percolation fraction
        
        # State variables (initial conditions - 60% filled)
        self.UZTWC = UZTWM * 0.6  # Upper zone tension water content
        self.UZFWC = UZFWM * 0.6  # Upper zone free water content
        self.LZTWC = LZTWM * 0.6  # Lower zone tension water content
        self.LZFPC = LZFPM * 0.6  # Lower zone primary free water content
        self.LZFSC = LZFSM * 0.6  # Lower zone supplementary free water content
        
    def evapotranspiration(self, PET: float) -> float:
        """
        Calculate actual evapotranspiration from all zones.
        
        Parameters:
        -----------
        PET : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        E : float
            Actual evapotranspiration (mm)
        """
        E_total = 0.0
        PET_remaining = PET
        
        # E1: From upper zone tension water
        if self.UZTWM > 0:
            E1 = min(self.UZTWC, PET_remaining * self.UZTWC / self.UZTWM)
            self.UZTWC -= E1
            E_total += E1
            PET_remaining -= E1
        
        # E2: From upper zone free water
        if self.UZFWM > 0 and PET_remaining > 0:
            E2 = min(self.UZFWC, PET_remaining * self.UZFWC / self.UZFWM)
            self.UZFWC -= E2
            E_total += E2
            PET_remaining -= E2
        
        # E3: From lower zone tension water
        if self.LZTWM > 0 and PET_remaining > 0:
            E3 = min(self.LZTWC, PET_remaining * self.LZTWC / self.LZTWM)
            self.LZTWC -= E3
            E_total += E3
            PET_remaining -= E3
        
        # E4 & E5: From lower zone free water (proportional)
        if PET_remaining > 0:
            LZFWM_total = self.LZFPM + self.LZFSM
            if LZFWM_total > 0:
                E4 = min(self.LZFPC, PET_remaining * self.LZFPC / LZFWM_total)
                self.LZFPC -= E4
                E_total += E4
                PET_remaining -= E4
                
                if PET_remaining > 0:
                    E5 = min(self.LZFSC, PET_remaining * self.LZFSC / LZFWM_total)
                    self.LZFSC -= E5
                    E_total += E5
        
        return E_total
    
    def infiltration(self, P: float) -> float:
        """
        Calculate infiltration to upper zone.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
            
        Returns:
        --------
        surface_runoff : float
            Surface runoff from impervious area (mm)
        """
        # Impervious area runoff
        surface_runoff = P * self.PCTIM
        
        # Infiltration to pervious area
        P_pervious = P * (1 - self.PCTIM)
        
        # Fill upper zone tension water first
        space_UZTW = max(0, self.UZTWM - self.UZTWC)
        to_UZTW = min(P_pervious, space_UZTW)
        self.UZTWC += to_UZTW
        P_pervious -= to_UZTW
        
        # Remaining goes to upper zone free water
        if P_pervious > 0:
            space_UZFW = max(0, self.UZFWM - self.UZFWC)
            to_UZFW = min(P_pervious, space_UZFW)
            self.UZFWC += to_UZFW
            P_pervious -= to_UZFW
            
            # Excess becomes surface runoff
            if P_pervious > 0:
                surface_runoff += P_pervious
        
        return surface_runoff
    
    def percolation(self) -> float:
        """
        Calculate percolation from upper to lower zone.
        
        Returns:
        --------
        perc : float
            Percolation amount (mm)
        """
        # Calculate deficit in lower zone
        deficit_lower = max(0, (self.LZTWM - self.LZTWC) + 
                           (self.LZFPM - self.LZFPC) + 
                           (self.LZFSM - self.LZFSC))
        
        if deficit_lower <= 0:
            return 0.0
        
        # Calculate percolation rate based on upper zone contents
        if self.UZTWM > 0:
            ratio = (self.UZTWC / self.UZTWM) + (self.UZFWC / self.UZFWM) * 0.5
        else:
            ratio = 0.0
        
        # Percolation equation
        if ratio > 0:
            perc = self.ZPERC * (ratio ** self.REXP)
        else:
            perc = 0.0
        
        # Limit percolation to available water and deficit
        perc = min(perc, self.UZFWC + self.UZTWC * 0.5)
        perc = min(perc, deficit_lower)
        
        # Remove from upper zone (preferentially from free water)
        if perc > 0:
            from_UZFW = min(perc, self.UZFWC)
            self.UZFWC -= from_UZFW
            perc_remaining = perc - from_UZFW
            
            if perc_remaining > 0:
                from_UZTW = min(perc_remaining, self.UZTWC)
                self.UZTWC -= from_UZTW
        
        # Add to lower zone
        # Direct percolation to free water
        to_free = perc * self.PFREE
        
        # Split free water between primary and supplementary
        if self.LZFPM + self.LZFSM > 0:
            ratio_primary = self.LZFPM / (self.LZFPM + self.LZFSM)
            to_LZFP = to_free * ratio_primary
            to_LZFS = to_free * (1 - ratio_primary)
            
            self.LZFPC = min(self.LZFPM, self.LZFPC + to_LZFP)
            self.LZFSC = min(self.LZFSM, self.LZFSC + to_LZFS)
        
        # Remaining to tension water
        to_tension = perc * (1 - self.PFREE)
        self.LZTWC = min(self.LZTWM, self.LZTWC + to_tension)
        
        return perc
    
    def generate_runoff(self) -> Dict[str, float]:
        """
        Generate runoff components.
        
        Returns:
        --------
        dict : Runoff components
        """
        # Interflow from upper zone free water
        interflow = self.UZFWC * self.UZK
        self.UZFWC -= interflow
        self.UZFWC = max(0, self.UZFWC)
        
        # Primary baseflow from lower zone primary free water
        baseflow_p = self.LZFPC * self.LZPK
        self.LZFPC -= baseflow_p
        self.LZFPC = max(0, self.LZFPC)
        
        # Supplementary baseflow from lower zone supplementary free water
        baseflow_s = self.LZFSC * self.LZSK
        self.LZFSC -= baseflow_s
        self.LZFSC = max(0, self.LZFSC)
        
        return {
            'interflow': interflow,
            'baseflow_primary': baseflow_p,
            'baseflow_supplementary': baseflow_s
        }
    
    def run_timestep(self, P: float, PET: float) -> Dict[str, float]:
        """
        Run one timestep of the Sacramento model.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        PET : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        dict : Model outputs
        """
        # 1. Evapotranspiration
        E = self.evapotranspiration(PET)
        
        # 2. Infiltration and surface runoff
        surface_runoff = self.infiltration(P)
        
        # 3. Percolation
        perc = self.percolation()
        
        # 4. Generate runoff
        runoff_components = self.generate_runoff()
        
        # Total discharge
        Q = (surface_runoff + 
             runoff_components['interflow'] + 
             runoff_components['baseflow_primary'] + 
             runoff_components['baseflow_supplementary'])
        
        return {
            'Q': Q,
            'surface': surface_runoff,
            'interflow': runoff_components['interflow'],
            'baseflow_p': runoff_components['baseflow_primary'],
            'baseflow_s': runoff_components['baseflow_supplementary'],
            'E': E,
            'UZTWC': self.UZTWC,
            'UZFWC': self.UZFWC,
            'LZTWC': self.LZTWC,
            'LZFPC': self.LZFPC,
            'LZFSC': self.LZFSC
        }
    
    def run(self, P: np.ndarray, PET: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run Sacramento model for multiple timesteps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm)
        PET : np.ndarray
            Potential evapotranspiration time series (mm)
            
        Returns:
        --------
        dict : Model outputs
        """
        n_steps = len(P)
        
        # Initialize output arrays
        Q = np.zeros(n_steps)
        surface = np.zeros(n_steps)
        interflow = np.zeros(n_steps)
        baseflow_p = np.zeros(n_steps)
        baseflow_s = np.zeros(n_steps)
        E = np.zeros(n_steps)
        UZTWC = np.zeros(n_steps)
        UZFWC = np.zeros(n_steps)
        LZTWC = np.zeros(n_steps)
        LZFPC = np.zeros(n_steps)
        LZFSC = np.zeros(n_steps)
        
        for t in range(n_steps):
            result = self.run_timestep(P[t], PET[t])
            Q[t] = result['Q']
            surface[t] = result['surface']
            interflow[t] = result['interflow']
            baseflow_p[t] = result['baseflow_p']
            baseflow_s[t] = result['baseflow_s']
            E[t] = result['E']
            UZTWC[t] = result['UZTWC']
            UZFWC[t] = result['UZFWC']
            LZTWC[t] = result['LZTWC']
            LZFPC[t] = result['LZFPC']
            LZFSC[t] = result['LZFSC']
        
        return {
            'Q': Q,
            'surface': surface,
            'interflow': interflow,
            'baseflow_primary': baseflow_p,
            'baseflow_supplementary': baseflow_s,
            'E': E,
            'UZTWC': UZTWC,
            'UZFWC': UZFWC,
            'LZTWC': LZTWC,
            'LZFPC': LZFPC,
            'LZFSC': LZFSC
        }


def main():
    """
    Example usage of Sacramento model.
    """
    print("=" * 80)
    print("SACRAMENTO Soil Moisture Accounting Model (SAC-SMA)")
    print("=" * 80)
    
    # Set random seed
    np.random.seed(42)
    
    # Generate synthetic data (365 days)
    n_days = 365
    
    # Precipitation
    P = np.random.gamma(2, 5, n_days)
    P[np.random.rand(n_days) > 0.4] = 0
    
    # Evapotranspiration
    t = np.arange(n_days)
    PET = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    
    print("\nInput Data Summary:")
    print(f"  Simulation period: {n_days} days")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Total potential ET: {np.sum(PET):.2f} mm")
    
    # Initialize model
    model = SacramentoModel(
        UZTWM=80.0, UZFWM=40.0,
        LZTWM=150.0, LZFPM=100.0, LZFSM=50.0,
        UZK=0.3, LZPK=0.01, LZSK=0.05,
        ZPERC=40.0, REXP=2.0,
        PCTIM=0.01, ADIMP=0.0, PFREE=0.1
    )
    
    print("\nModel Parameters:")
    print("  Upper Zone:")
    print(f"    UZTWM: {model.UZTWM} mm")
    print(f"    UZFWM: {model.UZFWM} mm")
    print("  Lower Zone:")
    print(f"    LZTWM: {model.LZTWM} mm")
    print(f"    LZFPM: {model.LZFPM} mm")
    print(f"    LZFSM: {model.LZFSM} mm")
    
    # Run model
    print("\nRunning simulation...")
    results = model.run(P, PET)
    
    print("\nSimulation Results:")
    print(f"  Total discharge: {np.sum(results['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results['Q']) / np.sum(P):.3f}")
    print(f"  Total ET: {np.sum(results['E']):.2f} mm")
    
    print("\nRunoff Components:")
    print(f"  Surface runoff: {np.sum(results['surface']):.2f} mm "
          f"({np.sum(results['surface'])/np.sum(results['Q'])*100:.1f}%)")
    print(f"  Interflow: {np.sum(results['interflow']):.2f} mm "
          f"({np.sum(results['interflow'])/np.sum(results['Q'])*100:.1f}%)")
    print(f"  Primary baseflow: {np.sum(results['baseflow_primary']):.2f} mm "
          f"({np.sum(results['baseflow_primary'])/np.sum(results['Q'])*100:.1f}%)")
    print(f"  Supplementary baseflow: {np.sum(results['baseflow_supplementary']):.2f} mm "
          f"({np.sum(results['baseflow_supplementary'])/np.sum(results['Q'])*100:.1f}%)")
    
    # Display first 10 days
    print("\nFirst 10 Days of Simulation:")
    print("Day |   P    |  PET   |   Q    |   E    | UZTWC  | LZTWC  |")
    print("----|--------|--------|--------|--------|--------|--------|")
    for i in range(10):
        print(f"{i+1:3d} | {P[i]:6.2f} | {PET[i]:6.2f} | {results['Q'][i]:6.2f} | "
              f"{results['E'][i]:6.2f} | {results['UZTWC'][i]:6.2f} | "
              f"{results['LZTWC'][i]:6.2f} |")
    
    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
