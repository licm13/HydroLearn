"""
Tank Model (タンクモデル) - Hydrological Rainfall-Runoff Model

The Tank model is a conceptual rainfall-runoff model developed by Sugawara (1961, 1995) 
in Japan. It represents the catchment as a series of vertically arranged tanks (reservoirs)
that simulate different runoff components (surface runoff, interflow, and baseflow).

Mathematical Foundation:
========================

The model consists of multiple tanks stacked vertically, where:
- Each tank has side outlets and a bottom outlet
- Side outlets represent different types of runoff (surface, interflow, baseflow)
- Bottom outlets represent percolation to lower tanks

For each tank i:

1. Storage Update:
   S_i(t+1) = S_i(t) + Input_i - ∑(Outflow_j) - Percolation_i
   
2. Side Outflow (j-th outlet):
   Q_j(t) = a_j * max(0, S_i(t) - h_j)
   
   Where:
   - Q_j: discharge from outlet j (mm/time)
   - a_j: discharge coefficient for outlet j (1/time)
   - S_i: storage in tank i (mm)
   - h_j: height threshold for outlet j (mm)

3. Bottom Outflow (Percolation):
   Perc_i(t) = b_i * S_i(t)
   
   Where:
   - Perc_i: percolation from tank i (mm/time)
   - b_i: percolation coefficient (1/time)

4. Total Discharge:
   Q_total(t) = ∑∑(Q_ij(t))  for all tanks and outlets

The number of tanks and outlets can vary:
- 1D: Single tank (simple)
- 2D: Two tanks (surface + baseflow)
- 3D: Three tanks (surface + interflow + baseflow)
- 4D: Four tanks (complete model with multiple runoff components)

Author: Sugawara (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings


class Tank:
    """
    Individual tank component for the Tank Model.
    
    Parameters:
    -----------
    side_outlets : list of tuples
        List of (coefficient, height_threshold) for side outlets
        [(a1, h1), (a2, h2), ...]
    bottom_coef : float
        Bottom percolation coefficient (b)
    initial_storage : float
        Initial storage in the tank (mm)
    """
    
    def __init__(self, 
                 side_outlets: List[Tuple[float, float]], 
                 bottom_coef: float = 0.0,
                 initial_storage: float = 0.0):
        self.side_outlets = side_outlets  # [(a1, h1), (a2, h2), ...]
        self.bottom_coef = bottom_coef    # b
        self.storage = initial_storage    # S
        
    def run_timestep(self, inflow: float) -> Tuple[float, List[float]]:
        """
        Run one timestep for this tank.
        
        Parameters:
        -----------
        inflow : float
            Input to the tank (mm)
            
        Returns:
        --------
        percolation : float
            Percolation to the next tank (mm)
        side_outflows : list
            Outflows from each side outlet (mm)
        """
        # Add inflow to storage
        self.storage += inflow
        
        # Calculate side outflows
        side_outflows = []
        for a, h in self.side_outlets:
            # Outflow = a * max(0, S - h)
            outflow = a * max(0, self.storage - h)
            side_outflows.append(outflow)
            self.storage -= outflow
            self.storage = max(0, self.storage)  # Ensure non-negative
        
        # Calculate bottom percolation
        percolation = self.bottom_coef * self.storage
        self.storage -= percolation
        self.storage = max(0, self.storage)  # Ensure non-negative
        
        return percolation, side_outflows


class TankModel:
    """
    Tank Model implementation with configurable number of tanks.
    
    This is a flexible implementation that supports 1D, 2D, 3D, and 4D configurations.
    
    Parameters:
    -----------
    tank_configs : list of dict
        Configuration for each tank from top to bottom
        Each dict should contain:
        - 'side_outlets': list of (coef, height) tuples
        - 'bottom_coef': percolation coefficient
        - 'initial_storage': initial storage (optional)
    """
    
    def __init__(self, tank_configs: List[Dict]):
        self.tanks = []
        for config in tank_configs:
            tank = Tank(
                side_outlets=config['side_outlets'],
                bottom_coef=config.get('bottom_coef', 0.0),
                initial_storage=config.get('initial_storage', 0.0)
            )
            self.tanks.append(tank)
        
        self.n_tanks = len(self.tanks)
        
    def run_timestep(self, precipitation: float, 
                     evapotranspiration: float = 0.0) -> Dict[str, float]:
        """
        Run one timestep of the tank model.
        
        Parameters:
        -----------
        precipitation : float
            Precipitation (mm)
        evapotranspiration : float
            Evapotranspiration (mm) - reduces top tank storage
            
        Returns:
        --------
        dict : Dictionary containing:
            - Q_total: Total discharge (mm)
            - Q_tanks: List of discharges from each tank
            - Q_outlets: List of all outlet discharges
            - storages: Current storage in each tank
        """
        # Apply ET to top tank (simple approach)
        net_input = max(0, precipitation - evapotranspiration)
        
        inflow = net_input
        total_discharge = 0.0
        all_outflows = []
        
        # Process each tank from top to bottom
        for i, tank in enumerate(self.tanks):
            percolation, side_outflows = tank.run_timestep(inflow)
            
            # Sum discharge from this tank
            tank_discharge = sum(side_outflows)
            total_discharge += tank_discharge
            all_outflows.extend(side_outflows)
            
            # Percolation becomes input to next tank
            inflow = percolation
        
        # Get current storages
        storages = [tank.storage for tank in self.tanks]
        
        return {
            'Q_total': total_discharge,
            'Q_outlets': all_outflows,
            'storages': storages,
            'percolation_loss': inflow  # Percolation from bottom tank
        }
    
    def run(self, P: np.ndarray, EP: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run the model for multiple timesteps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm)
        EP : np.ndarray, optional
            Evapotranspiration time series (mm)
            
        Returns:
        --------
        dict : Dictionary containing model outputs
        """
        n_steps = len(P)
        
        if EP is None:
            EP = np.zeros(n_steps)
        
        # Initialize output arrays
        Q_total = np.zeros(n_steps)
        storages = np.zeros((n_steps, self.n_tanks))
        
        # Run model
        for t in range(n_steps):
            result = self.run_timestep(P[t], EP[t])
            Q_total[t] = result['Q_total']
            storages[t, :] = result['storages']
        
        return {
            'Q': Q_total,
            'storages': storages
        }


class TankModel1D(TankModel):
    """
    1D Tank Model - Single tank configuration
    
    Simple model with one tank representing the entire catchment.
    Suitable for simple runoff simulation.
    
    Parameters:
    -----------
    a1 : float
        Discharge coefficient for surface runoff (1/day, typical: 0.1-0.5)
    h1 : float
        Height threshold for surface runoff (mm, typical: 10-30)
    b1 : float
        Bottom percolation coefficient (1/day, typical: 0.01-0.1)
    """
    
    def __init__(self, a1: float = 0.3, h1: float = 20.0, b1: float = 0.05):
        config = [{
            'side_outlets': [(a1, h1)],
            'bottom_coef': b1,
            'initial_storage': 10.0
        }]
        super().__init__(config)
        self.a1 = a1
        self.h1 = h1
        self.b1 = b1


class TankModel2D(TankModel):
    """
    2D Tank Model - Two tanks configuration
    
    Two tanks representing surface and subsurface components.
    - Tank 1: Surface runoff (fast response)
    - Tank 2: Baseflow (slow response)
    
    Parameters:
    -----------
    a11 : float
        Discharge coefficient for surface runoff from Tank 1
    h11 : float
        Height threshold for surface runoff from Tank 1
    a21 : float
        Discharge coefficient for baseflow from Tank 2
    h21 : float
        Height threshold for baseflow from Tank 2
    b1 : float
        Percolation coefficient from Tank 1 to Tank 2
    b2 : float
        Bottom percolation coefficient from Tank 2
    """
    
    def __init__(self, 
                 a11: float = 0.3, h11: float = 20.0,
                 a21: float = 0.1, h21: float = 5.0,
                 b1: float = 0.1, b2: float = 0.01):
        configs = [
            {
                'side_outlets': [(a11, h11)],
                'bottom_coef': b1,
                'initial_storage': 15.0
            },
            {
                'side_outlets': [(a21, h21)],
                'bottom_coef': b2,
                'initial_storage': 30.0
            }
        ]
        super().__init__(configs)


class TankModel3D(TankModel):
    """
    3D Tank Model - Three tanks configuration (Standard Configuration)
    
    Three tanks representing different runoff components:
    - Tank 1: Surface runoff (fast, direct runoff)
    - Tank 2: Interflow (intermediate response)
    - Tank 3: Baseflow (slow, groundwater)
    
    This is the most commonly used configuration.
    
    Parameters:
    -----------
    Tank 1 (Surface):
        a11, h11 : Surface runoff (fast)
        a12, h12 : Overflow
        b1 : Percolation to Tank 2
    
    Tank 2 (Interflow):
        a21, h21 : Interflow (medium)
        b2 : Percolation to Tank 3
    
    Tank 3 (Baseflow):
        a31, h31 : Baseflow (slow)
        b3 : Deep percolation loss
    """
    
    def __init__(self,
                 # Tank 1 - Surface
                 a11: float = 0.5, h11: float = 30.0,
                 a12: float = 0.8, h12: float = 50.0,
                 b1: float = 0.1,
                 # Tank 2 - Interflow
                 a21: float = 0.2, h21: float = 10.0,
                 b2: float = 0.05,
                 # Tank 3 - Baseflow
                 a31: float = 0.05, h31: float = 5.0,
                 b3: float = 0.001):
        
        configs = [
            # Tank 1: Surface runoff with two outlets
            {
                'side_outlets': [(a11, h11), (a12, h12)],
                'bottom_coef': b1,
                'initial_storage': 20.0
            },
            # Tank 2: Interflow
            {
                'side_outlets': [(a21, h21)],
                'bottom_coef': b2,
                'initial_storage': 30.0
            },
            # Tank 3: Baseflow
            {
                'side_outlets': [(a31, h31)],
                'bottom_coef': b3,
                'initial_storage': 50.0
            }
        ]
        super().__init__(configs)


def main():
    """
    Example usage of Tank models with randomly generated data.
    """
    print("=" * 80)
    print("Tank Model - Hydrological Model Examples")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic input data (365 days)
    n_days = 365
    
    # Precipitation: realistic pattern with dry and wet periods
    P = np.random.gamma(2, 5, n_days)
    P[np.random.rand(n_days) > 0.4] = 0  # 60% dry days
    
    # Potential evapotranspiration: seasonal variation
    t = np.arange(n_days)
    EP = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    
    print("\nInput Data Summary:")
    print(f"  Simulation period: {n_days} days")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Total ET: {np.sum(EP):.2f} mm")
    
    # ========================================================================
    # 1D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("1D Tank Model (Single Tank)")
    print("=" * 80)
    
    model_1d = TankModel1D(a1=0.3, h1=20.0, b1=0.05)
    results_1d = model_1d.run(P, EP)
    
    print("\nModel Parameters:")
    print(f"  a1 (discharge coef): {model_1d.a1}")
    print(f"  h1 (height threshold): {model_1d.h1} mm")
    print(f"  b1 (percolation coef): {model_1d.b1}")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_1d['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results_1d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_1d['Q']):.2f} mm/day")
    print(f"  Average storage: {np.mean(results_1d['storages']):.2f} mm")
    
    # ========================================================================
    # 2D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("2D Tank Model (Two Tanks - Surface + Baseflow)")
    print("=" * 80)
    
    model_2d = TankModel2D(
        a11=0.3, h11=20.0,
        a21=0.1, h21=5.0,
        b1=0.1, b2=0.01
    )
    results_2d = model_2d.run(P, EP)
    
    print("\nModel Parameters:")
    print("  Tank 1 (Surface):")
    print(f"    a11: 0.3, h11: 20.0 mm")
    print(f"    b1: 0.1")
    print("  Tank 2 (Baseflow):")
    print(f"    a21: 0.1, h21: 5.0 mm")
    print(f"    b2: 0.01")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_2d['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results_2d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_2d['Q']):.2f} mm/day")
    print(f"  Average Tank 1 storage: {np.mean(results_2d['storages'][:, 0]):.2f} mm")
    print(f"  Average Tank 2 storage: {np.mean(results_2d['storages'][:, 1]):.2f} mm")
    
    # ========================================================================
    # 3D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("3D Tank Model (Three Tanks - Surface + Interflow + Baseflow)")
    print("=" * 80)
    
    model_3d = TankModel3D(
        # Tank 1
        a11=0.5, h11=30.0,
        a12=0.8, h12=50.0,
        b1=0.1,
        # Tank 2
        a21=0.2, h21=10.0,
        b2=0.05,
        # Tank 3
        a31=0.05, h31=5.0,
        b3=0.001
    )
    results_3d = model_3d.run(P, EP)
    
    print("\nModel Parameters:")
    print("  Tank 1 (Surface):")
    print(f"    Outlet 1: a11=0.5, h11=30.0 mm")
    print(f"    Outlet 2: a12=0.8, h12=50.0 mm")
    print(f"    Bottom: b1=0.1")
    print("  Tank 2 (Interflow):")
    print(f"    Outlet 1: a21=0.2, h21=10.0 mm")
    print(f"    Bottom: b2=0.05")
    print("  Tank 3 (Baseflow):")
    print(f"    Outlet 1: a31=0.05, h31=5.0 mm")
    print(f"    Bottom: b3=0.001")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_3d['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results_3d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_3d['Q']):.2f} mm/day")
    print(f"  Average Tank 1 storage: {np.mean(results_3d['storages'][:, 0]):.2f} mm")
    print(f"  Average Tank 2 storage: {np.mean(results_3d['storages'][:, 1]):.2f} mm")
    print(f"  Average Tank 3 storage: {np.mean(results_3d['storages'][:, 2]):.2f} mm")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    
    print("\n{:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Total Q (mm)", "Runoff Coef", "Peak Q (mm/day)"))
    print("-" * 60)
    print("{:<15} {:<15.2f} {:<15.3f} {:<15.2f}".format(
        "1D Tank", np.sum(results_1d['Q']), 
        np.sum(results_1d['Q'])/np.sum(P), np.max(results_1d['Q'])))
    print("{:<15} {:<15.2f} {:<15.3f} {:<15.2f}".format(
        "2D Tank", np.sum(results_2d['Q']), 
        np.sum(results_2d['Q'])/np.sum(P), np.max(results_2d['Q'])))
    print("{:<15} {:<15.2f} {:<15.3f} {:<15.2f}".format(
        "3D Tank", np.sum(results_3d['Q']), 
        np.sum(results_3d['Q'])/np.sum(P), np.max(results_3d['Q'])))
    
    # Display first 10 days
    print("\n" + "=" * 80)
    print("First 10 Days of 3D Tank Model Simulation")
    print("=" * 80)
    print("\nDay |   P    |   EP   |   Q    | Tank1  | Tank2  | Tank3  |")
    print("----|--------|--------|--------|--------|--------|--------|")
    for i in range(10):
        print(f"{i+1:3d} | {P[i]:6.2f} | {EP[i]:6.2f} | {results_3d['Q'][i]:6.2f} | "
              f"{results_3d['storages'][i,0]:6.2f} | {results_3d['storages'][i,1]:6.2f} | "
              f"{results_3d['storages'][i,2]:6.2f} |")
    
    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print("=" * 80)
    
    return results_1d, results_2d, results_3d


if __name__ == "__main__":
    main()
