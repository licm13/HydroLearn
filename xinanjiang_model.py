"""
Xinanjiang (新安江) Hydrological Model

The Xinanjiang model is a conceptual rainfall-runoff model developed by Zhao Ren-jun
in 1973 at Hohai University, China. It is widely used in humid and semi-humid regions
for flood forecasting and water resources management.

Mathematical Foundation:
=======================

1. Evapotranspiration Calculation:
   - Three-layer evapotranspiration structure (upper, lower, and deep layer)
   - EU = EP * (W / WM)                    if W < WM
   - EL = EP * (W - WM) / (WM * (C - 1))  if W >= WM
   - ED = EP * (W - WM - WLM) / (WM * (C - 1) * (C - 1)) if W >= WM + WLM

2. Runoff Generation (Saturation Excess Mechanism):
   The model assumes a parabolic distribution of soil moisture capacity:
   
   f = SM * (1 - (1 - A)^(1/(1+B)))
   
   Where:
   - f: cumulative frequency distribution of soil moisture capacity
   - SM: areal mean soil moisture storage capacity (mm)
   - A: area fraction with soil moisture capacity <= a
   - B: shape parameter of the distribution curve

   Runoff generation:
   R = P + A0 - SM + SM * (1 - (P + A0)/SM)^(1+B)  if P + A0 < SM
   R = P + A0 - SM                                   if P + A0 >= SM

3. Runoff Separation:
   Total runoff is separated into surface runoff (RS), interflow (RI), and groundwater (RG)
   
   RS = PE * FR                    Surface runoff
   RI = PE * (1 - FR) * KI        Interflow
   RG = PE * (1 - FR) * (1 - KI)  Groundwater

4. Flow Concentration:
   - Surface runoff uses unit hydrograph
   - Interflow and groundwater use linear reservoirs
   
   Q(t) = KS * S(t)   for surface
   Q(t) = KI * RI(t)  for interflow  
   Q(t) = KG * RG(t)  for groundwater

Author: Zhao Ren-jun (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import Tuple, Dict
import warnings


class XinanjiangModel:
    """
    Xinanjiang (新安江) rainfall-runoff model implementation.
    
    This model simulates the rainfall-runoff process using a saturation excess 
    runoff generation mechanism, which is particularly suitable for humid regions.
    
    Parameters:
    -----------
    K : float
        Ratio of potential evapotranspiration to pan evaporation (typically 0.7-1.2)
    B : float
        Exponent of the tension water capacity curve (typically 0.1-0.4)
    IMP : float
        Impervious area fraction (0-0.1)
    WM : float
        Average soil moisture storage capacity (mm, typically 120-200)
    WUM : float
        Upper layer soil moisture storage capacity (mm, typically 20-50)
    WLM : float
        Lower layer soil moisture storage capacity (mm, typically 60-90)
    C : float
        Coefficient of deep layer evapotranspiration (typically 0.15-0.20)
    SM : float
        Areal mean free water capacity of surface soil layer (mm, typically 10-50)
    EX : float
        Exponent of the free water capacity curve (typically 1.0-1.5)
    KI : float
        Outflow coefficient of interflow storage (typically 0.2-0.7)
    KG : float
        Outflow coefficient of groundwater storage (typically 0.2-0.7)
    CI : float
        Recession constant of interflow storage (typically 0.5-0.9)
    CG : float
        Recession constant of groundwater storage (typically 0.95-0.998)
    """
    
    def __init__(self, 
                 K: float = 1.0,
                 B: float = 0.3,
                 IMP: float = 0.01,
                 WM: float = 150.0,
                 WUM: float = 30.0,
                 WLM: float = 70.0,
                 C: float = 0.17,
                 SM: float = 30.0,
                 EX: float = 1.5,
                 KI: float = 0.3,
                 KG: float = 0.3,
                 CI: float = 0.7,
                 CG: float = 0.98):
        
        # Model parameters
        self.K = K      # Evapotranspiration coefficient
        self.B = B      # Exponent of tension water distribution curve
        self.IMP = IMP  # Impervious area ratio
        self.WM = WM    # Average tension water capacity
        self.WUM = WUM  # Upper layer tension water capacity
        self.WLM = WLM  # Lower layer tension water capacity
        self.WDM = WM - WUM - WLM  # Deep layer capacity
        self.C = C      # Deep layer evapotranspiration coefficient
        self.SM = SM    # Areal mean free water capacity
        self.EX = EX    # Exponent of free water distribution curve
        self.KI = KI    # Interflow outflow coefficient
        self.KG = KG    # Groundwater outflow coefficient
        self.CI = CI    # Interflow recession constant
        self.CG = CG    # Groundwater recession constant
        
        # State variables (initial conditions)
        self.W = WM * 0.6    # Initial soil moisture (60% of capacity)
        self.S = SM * 0.3    # Initial free water storage
        self.SI = 0.0        # Initial interflow storage
        self.SG = 0.0        # Initial groundwater storage
        
    def evapotranspiration(self, EP: float) -> Tuple[float, float, float]:
        """
        Calculate three-layer evapotranspiration.
        
        EP: Potential evapotranspiration (mm)
        Returns: (EU, EL, ED) - Upper, Lower, Deep layer evapotranspiration
        """
        EP = self.K * EP  # Adjust potential ET
        
        EU = EL = ED = 0.0
        
        # Upper layer evapotranspiration
        if self.W < self.WUM:
            EU = min(EP * self.W / self.WUM, self.W)
        else:
            EU = EP
            
        # Lower layer evapotranspiration
        if self.W >= self.WUM:
            W_temp = self.W - self.WUM
            if W_temp < self.WLM:
                EL = min((EP - EU) * W_temp / self.WLM, W_temp)
            else:
                EL = EP - EU
                
        # Deep layer evapotranspiration
        if self.W >= (self.WUM + self.WLM):
            W_temp = self.W - self.WUM - self.WLM
            if W_temp < self.WDM:
                ED = min(self.C * (EP - EU - EL) * W_temp / self.WDM, W_temp)
            else:
                ED = self.C * (EP - EU - EL)
        
        return EU, EL, ED
    
    def runoff_generation(self, P: float, EP: float) -> float:
        """
        Calculate runoff generation using saturation excess mechanism.
        
        P: Precipitation (mm)
        EP: Potential evapotranspiration (mm)
        Returns: Runoff (mm)
        """
        # Calculate evapotranspiration
        EU, EL, ED = self.evapotranspiration(EP)
        E = EU + EL + ED
        
        # Update soil moisture
        PE = max(0, P - E)  # Net precipitation
        
        # Calculate runoff from pervious area using parabolic curve
        if PE > 0 and self.WM > 0.01:  # Add safety check for division
            A = self.WM * (1.0 - (1.0 - self.W / self.WM) ** (1.0 / (1.0 + self.B)))
            
            if PE + A < self.WM:
                # Partial area generates runoff
                R = PE + A - self.WM + self.WM * (1.0 - (PE + A) / self.WM) ** (1.0 + self.B)
            else:
                # Entire area generates runoff
                R = PE + A - self.WM
                
            # Update soil moisture
            self.W = min(self.WM, self.W + PE - R)
        else:
            R = 0.0
            self.W = max(0.0, self.W - E)
        
        # Add runoff from impervious area
        R_total = R * (1.0 - self.IMP) + P * self.IMP
        
        return R_total
    
    def runoff_separation(self, R: float) -> Tuple[float, float, float]:
        """
        Separate total runoff into surface runoff, interflow, and groundwater.
        
        R: Total runoff (mm)
        Returns: (RS, RI, RG) - Surface runoff, Interflow, Groundwater
        """
        if R <= 0:
            return 0.0, 0.0, 0.0
        
        # Update free water storage
        self.S = self.S + R
        
        # Calculate surface runoff using free water capacity curve
        if self.S <= self.SM:
            # Free water capacity not exceeded - calculate using distribution
            S_ratio = self.S / self.SM
            # Surface runoff occurs when free water exceeds local capacity
            if S_ratio > 0 and self.EX > 0:
                RS = max(0, self.S - self.SM * (1.0 - S_ratio ** (1.0 + self.EX)))
            else:
                RS = 0.0
        else:
            # Free water capacity exceeded
            RS = self.S - self.SM
        
        # Limit surface runoff to available runoff
        RS = min(RS, R)
        RS = max(0, RS)
        
        # Update storage after surface runoff
        self.S = max(0.0, self.S - RS)
        
        # Remaining water for interflow and groundwater
        RSS = max(0, R - RS)
        
        # Separate interflow and groundwater
        if RSS > 0:
            RI = self.KI * RSS
            RG = (1.0 - self.KI) * RSS
        else:
            RI = 0.0
            RG = 0.0
        
        return RS, RI, RG
    
    def flow_routing(self, RS: float, RI: float, RG: float) -> Tuple[float, float, float]:
        """
        Route surface runoff, interflow, and groundwater to outlet.
        
        Uses linear reservoir routing for interflow and groundwater.
        
        Returns: (QS, QI, QG) - Surface flow, Interflow, Groundwater flow
        """
        # Surface runoff (no routing in this simplified version)
        QS = RS
        
        # Interflow routing (linear reservoir)
        self.SI = self.CI * self.SI + RI
        QI = (1.0 - self.CI) * self.SI
        
        # Groundwater routing (linear reservoir)
        self.SG = self.CG * self.SG + RG
        QG = (1.0 - self.CG) * self.SG
        
        return QS, QI, QG
    
    def run_timestep(self, P: float, EP: float) -> Dict[str, float]:
        """
        Run the model for one time step.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        EP : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        dict : Dictionary containing model outputs
            - Q: Total discharge (mm)
            - QS: Surface runoff (mm)
            - QI: Interflow (mm)
            - QG: Groundwater flow (mm)
            - R: Total runoff generated (mm)
            - E: Actual evapotranspiration (mm)
            - W: Soil moisture (mm)
            - S: Free water storage (mm)
        """
        # Calculate evapotranspiration
        EU, EL, ED = self.evapotranspiration(EP)
        E = EU + EL + ED
        
        # Generate runoff
        R = self.runoff_generation(P, EP)
        
        # Separate runoff components
        RS, RI, RG = self.runoff_separation(R)
        
        # Route flows
        QS, QI, QG = self.flow_routing(RS, RI, RG)
        
        # Total discharge
        Q = QS + QI + QG
        
        return {
            'Q': Q,      # Total discharge
            'QS': QS,    # Surface runoff
            'QI': QI,    # Interflow
            'QG': QG,    # Groundwater flow
            'R': R,      # Total runoff generated
            'E': E,      # Evapotranspiration
            'W': self.W, # Soil moisture
            'S': self.S  # Free water storage
        }
    
    def run(self, P: np.ndarray, EP: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the model for multiple time steps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm)
        EP : np.ndarray
            Potential evapotranspiration time series (mm)
            
        Returns:
        --------
        dict : Dictionary containing model outputs as arrays
        """
        n_steps = len(P)
        
        # Initialize output arrays
        Q = np.zeros(n_steps)
        QS = np.zeros(n_steps)
        QI = np.zeros(n_steps)
        QG = np.zeros(n_steps)
        R = np.zeros(n_steps)
        E = np.zeros(n_steps)
        W = np.zeros(n_steps)
        S = np.zeros(n_steps)
        
        # Run model for each time step
        for t in range(n_steps):
            result = self.run_timestep(P[t], EP[t])
            Q[t] = result['Q']
            QS[t] = result['QS']
            QI[t] = result['QI']
            QG[t] = result['QG']
            R[t] = result['R']
            E[t] = result['E']
            W[t] = result['W']
            S[t] = result['S']
        
        return {
            'Q': Q,
            'QS': QS,
            'QI': QI,
            'QG': QG,
            'R': R,
            'E': E,
            'W': W,
            'S': S
        }


def main():
    """
    Example usage of the Xinanjiang model with randomly generated data.
    """
    print("=" * 80)
    print("Xinanjiang (新安江) Hydrological Model - Example")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic input data (365 days)
    n_days = 365
    
    # Precipitation: gamma distribution with shape=2, scale=5 (typical rainfall pattern)
    # Some days have no rain (60% dry days)
    P = np.random.gamma(2, 5, n_days)
    P[np.random.rand(n_days) > 0.4] = 0  # 60% dry days
    
    # Potential evapotranspiration: seasonal variation
    # Higher in summer, lower in winter
    t = np.arange(n_days)
    EP = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)  # Varies between 1-5 mm/day
    
    # Initialize model with default parameters
    model = XinanjiangModel(
        K=1.0,
        B=0.3,
        IMP=0.01,
        WM=150.0,
        WUM=30.0,
        WLM=70.0,
        C=0.17,
        SM=30.0,
        EX=1.5,
        KI=0.3,
        KG=0.3,
        CI=0.7,
        CG=0.98
    )
    
    print("\nModel Parameters:")
    print(f"  K (ET coefficient): {model.K}")
    print(f"  B (Tension water curve exponent): {model.B}")
    print(f"  IMP (Impervious area ratio): {model.IMP}")
    print(f"  WM (Average tension water capacity): {model.WM} mm")
    print(f"  WUM (Upper layer capacity): {model.WUM} mm")
    print(f"  WLM (Lower layer capacity): {model.WLM} mm")
    print(f"  C (Deep layer ET coefficient): {model.C}")
    print(f"  SM (Free water capacity): {model.SM} mm")
    print(f"  EX (Free water curve exponent): {model.EX}")
    
    # Run model
    print("\nRunning model for {} days...".format(n_days))
    results = model.run(P, EP)
    
    # Display results
    print("\nSimulation Results Summary:")
    print(f"  Total Precipitation: {np.sum(P):.2f} mm")
    print(f"  Total Evapotranspiration: {np.sum(results['E']):.2f} mm")
    print(f"  Total Runoff: {np.sum(results['R']):.2f} mm")
    print(f"  Total Discharge: {np.sum(results['Q']):.2f} mm")
    print(f"  Average Soil Moisture: {np.mean(results['W']):.2f} mm")
    print(f"  Runoff Coefficient: {np.sum(results['R']) / np.sum(P):.3f}")
    
    print("\nRunoff Components:")
    print(f"  Surface Runoff: {np.sum(results['QS']):.2f} mm ({np.sum(results['QS'])/np.sum(results['Q'])*100:.1f}%)")
    print(f"  Interflow: {np.sum(results['QI']):.2f} mm ({np.sum(results['QI'])/np.sum(results['Q'])*100:.1f}%)")
    print(f"  Groundwater: {np.sum(results['QG']):.2f} mm ({np.sum(results['QG'])/np.sum(results['Q'])*100:.1f}%)")
    
    # Display first 10 days as example
    print("\nFirst 10 days of simulation:")
    print("Day |   P    |   EP   |   Q    |   E    |   W    |")
    print("----|--------|--------|--------|--------|--------|")
    for i in range(10):
        print(f"{i+1:3d} | {P[i]:6.2f} | {EP[i]:6.2f} | {results['Q'][i]:6.2f} | "
              f"{results['E'][i]:6.2f} | {results['W'][i]:6.2f} |")
    
    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
