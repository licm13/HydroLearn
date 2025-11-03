"""
GR4J (Modèle du Génie Rural à 4 paramètres Journalier)
Rural Engineering 4-Parameter Daily Model

GR4J is a lumped, conceptual rainfall-runoff model developed by Perrin et al. (2003)
at INRAE (formerly Cemagref), France. It is widely used for daily streamflow simulation
and forecasting.

Mathematical Foundation:
========================

The model operates in several steps:

1. Production Store (Soil Moisture Accounting):
   
   Net rainfall: Pn = max(0, P - E)
   Net evapotranspiration: En = max(0, E - P)
   
   Reservoir level S: 
   - If Pn > 0:
     Ps = (X1 * (1 - (S/X1)^2) * tanh(Pn/X1)) / (1 + (S/X1) * tanh(Pn/X1))
     S = S + Ps
   
   - If En > 0:
     Es = (S * (2 - S/X1) * tanh(En/X1)) / (1 + (1 - S/X1) * tanh(En/X1))
     S = S - Es
   
   Percolation: Perc = S * (1 - (1 + (4/9 * S/X1)^4)^(-1/4))
   S = S - Perc

2. Unit Hydrograph Split:
   
   Total water to route: Pr = Pn - Ps + Perc
   
   90% routed through UH1 (unit hydrograph 1)
   10% routed through UH2 (unit hydrograph 2)

3. Routing Store:
   
   Input to routing: Q9 (from UH1)
   
   If Q9 > 0:
     Qr = (X3 * (1 - (R/X3)^2) * tanh(Q9/X3)) / (1 + (R/X3) * tanh(Q9/X3))
     R = R + Qr
   
   Outflow: Qd = R * (1 - (1 + (X2/R)^4)^(-1/4))
   R = R - Qd

4. Groundwater Exchange:
   
   F = X2 * (R / X3)^(7/2)

5. Final Discharge:
   
   Q1 (from UH2) + max(0, Qd + F)

Parameters:
-----------
X1 : Maximum capacity of production store (mm, typically 100-1200)
X2 : Groundwater exchange coefficient (mm, typically -5 to 5)
X3 : Maximum capacity of routing store (mm, typically 20-300)
X4 : Time base of unit hydrograph (days, typically 1.0-4.0)

References:
-----------
Perrin, C., Michel, C., & Andréassian, V. (2003). 
Improvement of a parsimonious model for streamflow simulation. 
Journal of Hydrology, 279(1-4), 275-289.

Author: Perrin et al. (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import Tuple, Dict
import warnings


class GR4J:
    """
    GR4J hydrological model implementation.
    
    Parameters:
    -----------
    X1 : float
        Maximum capacity of production store (mm, default: 350)
    X2 : float
        Groundwater exchange coefficient (mm, default: 0.0)
    X3 : float
        Maximum capacity of routing store (mm, default: 90)
    X4 : float
        Time base of unit hydrograph (days, default: 1.7)
    """
    
    def __init__(self, X1: float = 350.0, X2: float = 0.0, 
                 X3: float = 90.0, X4: float = 1.7):
        # Model parameters
        self.X1 = X1  # Production store capacity
        self.X2 = X2  # Water exchange coefficient
        self.X3 = X3  # Routing store capacity
        self.X4 = X4  # Unit hydrograph time base
        
        # State variables
        self.S = X1 * 0.5   # Production store level (initial: 50%)
        self.R = X3 * 0.3   # Routing store level (initial: 30%)
        
        # Unit hydrographs
        self._compute_unit_hydrographs()
        
        # UH ordinates storage
        self.UH1_queue = np.zeros(len(self.UH1))
        self.UH2_queue = np.zeros(len(self.UH2))
        
    def _compute_unit_hydrographs(self):
        """
        Compute unit hydrograph ordinates (SH1 and SH2).
        
        UH1: Fast response unit hydrograph (time base = X4)
        UH2: Slow response unit hydrograph (time base = 2*X4)
        """
        # UH1 (fast component) - time base X4
        n1 = int(np.ceil(self.X4))
        SH1 = np.zeros(n1)
        
        for t in range(1, n1 + 1):
            if t <= self.X4:
                SH1[t-1] = (t / self.X4) ** 2.5
            else:
                SH1[t-1] = 1.0
        
        # Convert S-curve to unit hydrograph
        self.UH1 = np.zeros(n1)
        self.UH1[0] = SH1[0]
        for t in range(1, n1):
            self.UH1[t] = SH1[t] - SH1[t-1]
        
        # UH2 (slow component) - time base 2*X4
        n2 = int(np.ceil(2 * self.X4))
        SH2 = np.zeros(n2)
        
        for t in range(1, n2 + 1):
            if t <= self.X4:
                SH2[t-1] = 0.5 * (t / self.X4) ** 2.5
            elif t <= 2 * self.X4:
                SH2[t-1] = 1.0 - 0.5 * (2 - t / self.X4) ** 2.5
            else:
                SH2[t-1] = 1.0
        
        # Convert S-curve to unit hydrograph
        self.UH2 = np.zeros(n2)
        self.UH2[0] = SH2[0]
        for t in range(1, n2):
            self.UH2[t] = SH2[t] - SH2[t-1]
    
    def _production_store(self, P: float, E: float) -> Tuple[float, float]:
        """
        Production store (soil moisture accounting).
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        E : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        Pn : float
            Net rainfall (mm)
        Perc : float
            Percolation (mm)
        """
        # Net precipitation and evapotranspiration
        if P >= E:
            Pn = P - E
            En = 0.0
        else:
            Pn = 0.0
            En = E - P
        
        # Production store update
        if Pn > 0:
            # Add water to store
            S_norm = self.S / self.X1
            if Pn > 0.01:  # Avoid numerical issues
                part1 = self.X1 * (1 - S_norm ** 2) * np.tanh(Pn / self.X1)
                part2 = 1 + S_norm * np.tanh(Pn / self.X1)
                if abs(part2) > 0.001:  # Safety check for division by zero
                    Ps = part1 / part2
                else:
                    Ps = 0.0
            else:
                Ps = 0.0
            
            self.S = self.S + Ps
        else:
            Ps = 0.0
        
        if En > 0:
            # Remove water from store
            S_norm = self.S / self.X1
            if En > 0.01:  # Avoid numerical issues
                part1 = self.S * (2 - S_norm) * np.tanh(En / self.X1)
                part2 = 1 + (1 - S_norm) * np.tanh(En / self.X1)
                Es = part1 / part2
            else:
                Es = 0.0
            
            self.S = max(0, self.S - Es)
        
        # Percolation from production store
        S_norm = self.S / self.X1
        if S_norm > 0:
            Perc = self.S * (1 - (1 + (4 * S_norm / 9) ** 4) ** (-0.25))
        else:
            Perc = 0.0
        
        self.S = max(0, self.S - Perc)
        
        # Effective rainfall for routing
        Pr = Pn - Ps + Perc
        
        return Pr, Ps
    
    def _routing(self, Q9: float, Q1: float) -> float:
        """
        Routing store and groundwater exchange.
        
        Parameters:
        -----------
        Q9 : float
            Input from UH1 (90% of effective rainfall)
        Q1 : float
            Input from UH2 (10% of effective rainfall)
            
        Returns:
        --------
        Q : float
            Total discharge (mm)
        """
        # Routing store update
        if Q9 > 0:
            R_norm = self.R / self.X3
            if Q9 > 0.01:
                part1 = self.X3 * (1 - R_norm ** 2) * np.tanh(Q9 / self.X3)
                part2 = 1 + R_norm * np.tanh(Q9 / self.X3)
                Qr = part1 / part2
            else:
                Qr = 0.0
            
            self.R = self.R + Qr
        
        # Outflow from routing store
        R_norm = self.R / self.X3
        if R_norm > 0 and self.R > 0.01:  # Add safety check for division by zero
            Qd = self.R * (1 - (1 + (self.X2 / self.R) ** 4) ** (-0.25))
        else:
            Qd = 0.0
        
        self.R = max(0, self.R - Qd)
        
        # Groundwater exchange
        R_norm = self.R / self.X3
        F = self.X2 * (R_norm ** 3.5)
        
        # Total discharge
        Q = max(0, Qd + F) + Q1
        
        return Q
    
    def run_timestep(self, P: float, E: float) -> Dict[str, float]:
        """
        Run one timestep of GR4J model.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        E : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        dict : Model outputs
        """
        # 1. Production store
        Pr, Ps = self._production_store(P, E)
        
        # 2. Split effective rainfall
        # 90% goes through UH1, 10% through UH2
        Pr_90 = 0.9 * Pr
        Pr_10 = 0.1 * Pr
        
        # 3. Unit hydrograph convolution
        # Update queues
        self.UH1_queue = np.roll(self.UH1_queue, 1)
        self.UH1_queue[0] = Pr_90
        
        self.UH2_queue = np.roll(self.UH2_queue, 1)
        self.UH2_queue[0] = Pr_10
        
        # Calculate outputs
        Q9 = np.sum(self.UH1_queue * self.UH1)
        Q1 = np.sum(self.UH2_queue * self.UH2)
        
        # 4. Routing
        Q = self._routing(Q9, Q1)
        
        return {
            'Q': Q,
            'S': self.S,
            'R': self.R,
            'Pr': Pr
        }
    
    def run(self, P: np.ndarray, E: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run GR4J model for multiple timesteps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm/day)
        E : np.ndarray
            Potential evapotranspiration time series (mm/day)
            
        Returns:
        --------
        dict : Model outputs
        """
        n_steps = len(P)
        
        Q = np.zeros(n_steps)
        S = np.zeros(n_steps)
        R = np.zeros(n_steps)
        
        for t in range(n_steps):
            result = self.run_timestep(P[t], E[t])
            Q[t] = result['Q']
            S[t] = result['S']
            R[t] = result['R']
        
        return {
            'Q': Q,
            'S': S,
            'R': R
        }


def main():
    """
    Example usage of GR4J model.
    """
    print("=" * 80)
    print("GR4J (Modèle du Génie Rural à 4 paramètres Journalier)")
    print("Rural Engineering 4-Parameter Daily Model")
    print("=" * 80)
    
    # Set random seed
    np.random.seed(42)
    
    # Generate synthetic data (2 years)
    n_days = 730
    
    # Precipitation: realistic pattern
    P = np.random.gamma(2, 5, n_days)
    P[np.random.rand(n_days) > 0.4] = 0  # 60% dry days
    
    # Evapotranspiration: seasonal pattern
    t = np.arange(n_days)
    E = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    
    print("\nInput Data Summary:")
    print(f"  Simulation period: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Total potential ET: {np.sum(E):.2f} mm")
    
    # Initialize model with default parameters
    model = GR4J(X1=350.0, X2=0.0, X3=90.0, X4=1.7)
    
    print("\nModel Parameters:")
    print(f"  X1 (Production capacity): {model.X1:.1f} mm")
    print(f"  X2 (Water exchange): {model.X2:.2f} mm")
    print(f"  X3 (Routing capacity): {model.X3:.1f} mm")
    print(f"  X4 (Unit hydrograph time): {model.X4:.2f} days")
    
    # Run model
    print("\nRunning simulation...")
    results = model.run(P, E)
    
    print("\nSimulation Results:")
    print(f"  Total discharge: {np.sum(results['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results['Q']):.2f} mm/day")
    print(f"  Mean discharge: {np.mean(results['Q']):.2f} mm/day")
    print(f"  Average production store: {np.mean(results['S']):.2f} mm")
    print(f"  Average routing store: {np.mean(results['R']):.2f} mm")
    
    # Display first 10 days
    print("\nFirst 10 Days of Simulation:")
    print("Day |   P    |   E    |   Q    |   S    |   R    |")
    print("----|--------|--------|--------|--------|--------|")
    for i in range(10):
        print(f"{i+1:3d} | {P[i]:6.2f} | {E[i]:6.2f} | {results['Q'][i]:6.2f} | "
              f"{results['S'][i]:6.2f} | {results['R'][i]:6.2f} |")
    
    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
