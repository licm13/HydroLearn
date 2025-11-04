"""
HBV-96 (Hydrologiska Byråns Vattenbalansavdelning) Hydrological Model
HBV-96 水文模型

The HBV model is a conceptual rainfall-runoff model developed by the Swedish 
Meteorological and Hydrological Institute (SMHI) by Bergström (1976, 1992).
It is one of the most widely used hydrological models globally, particularly 
suitable for Nordic climate conditions but adaptable to various regions.

Mathematical Foundation / 数学基础:
=======================

1. Snow Accumulation and Melt / 积雪累积与融化:
   
   If T < TT:  (Temperature below threshold / 温度低于阈值)
       Snow accumulation: SNOW = SNOW + P
       Liquid water: Pn = 0
   
   If T >= TT: (Temperature above threshold / 温度高于或等于阈值)
       Snowmelt: M = CFMAX * (T - TT)
       Pn = P + min(M, SNOW)
       SNOW = SNOW - min(M, SNOW)
   
   Where / 其中:
   - T: air temperature (°C / 气温)
   - TT: threshold temperature (°C / 阈值温度)
   - CFMAX: degree-day factor (mm/°C/day / 度日因子)
   - P: precipitation (mm / 降水)
   - Pn: liquid water input to soil (mm / 输入土壤的液态水)

2. Soil Moisture Accounting / 土壤水分核算:
   
   Actual evapotranspiration / 实际蒸散发:
   Ea = Ep * min(SM/FC, 1) * (SM/(LP*FC))
   
   Recharge to groundwater / 地下水补给:
   If SM > FC:
       R = (SM - FC) * BETA
   Else:
       R = 0
   
   Soil moisture update / 土壤水分更新:
   SM = SM + Pn - Ea - R
   
   Where / 其中:
   - SM: soil moisture (mm / 土壤水分)
   - FC: field capacity (mm / 田间持水量)
   - LP: limit for potential evapotranspiration (fraction / 潜在蒸散发限制系数)
   - BETA: shape coefficient (non-linear factor / 形状系数)
   - Ep: potential evapotranspiration (mm / 潜在蒸散发)
   - Ea: actual evapotranspiration (mm / 实际蒸散发)

3. Response Function (Upper and Lower Reservoirs) / 响应函数(上层和下层水库):
   
   Upper reservoir / 上层水库:
   Q0 = K0 * max(0, SUZ - UZL)  (Fast response / 快速响应)
   Q1 = K1 * SUZ                 (Slow response / 慢速响应)
   Percolation: PERC = PERC_MAX * min(1, SUZ/FC)
   SUZ = SUZ + R - Q0 - Q1 - PERC
   
   Lower reservoir / 下层水库:
   Q2 = K2 * SLZ                 (Baseflow / 基流)
   SLZ = SLZ + PERC - Q2
   
   Total discharge / 总径流:
   Q = Q0 + Q1 + Q2
   
   Where / 其中:
   - SUZ: upper zone storage (mm / 上层蓄水量)
   - SLZ: lower zone storage (mm / 下层蓄水量)
   - K0, K1, K2: recession coefficients (1/day / 消退系数)
   - UZL: threshold for fast response (mm / 快速响应阈值)
   - PERC_MAX: maximum percolation rate (mm/day / 最大下渗率)

Parameters / 参数:
-----------
TT : float (°C)
    Threshold temperature for snow/rain (典型值: -1.0 to 2.0)
CFMAX : float (mm/°C/day)
    Degree-day factor for snowmelt (典型值: 2.0 to 5.0)
FC : float (mm)
    Field capacity, maximum soil moisture storage (典型值: 100 to 400)
LP : float (-)
    Limit for potential evapotranspiration (典型值: 0.5 to 1.0)
BETA : float (-)
    Shape coefficient for recharge (典型值: 1.0 to 6.0)
K0 : float (1/day)
    Recession coefficient for fast flow (典型值: 0.1 to 0.5)
K1 : float (1/day)
    Recession coefficient for slow flow (典型值: 0.01 to 0.1)
K2 : float (1/day)
    Recession coefficient for baseflow (典型值: 0.001 to 0.05)
UZL : float (mm)
    Threshold for fast response (典型值: 0 to 50)
PERC : float (mm/day)
    Maximum percolation rate (典型值: 0 to 3)

References / 参考文献:
-----------
Bergström, S. (1976). Development and application of a conceptual runoff model 
for Scandinavian catchments. SMHI Reports RHO, No. 7.

Bergström, S. (1992). The HBV model - its structure and applications. 
SMHI Reports Hydrology, No. 4.

Author: Bergström (Original), Educational implementation for HydroLearn
Date: 2024
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os


class HBVModel:
    """
    HBV-96 rainfall-runoff model implementation for educational purposes.
    HBV-96 降雨径流模型的教学实现。
    
    This is a simplified version of the HBV model focusing on the core concepts
    for teaching hydrological modeling principles.
    这是HBV模型的简化版本，专注于水文建模原理的核心概念教学。
    
    Parameters / 参数:
    -----------
    TT : float, optional
        Threshold temperature (°C), default=0.0
        阈值温度（摄氏度），默认值=0.0
    CFMAX : float, optional
        Degree-day factor (mm/°C/day), default=3.5
        度日因子（毫米/摄氏度/天），默认值=3.5
    FC : float, optional
        Field capacity (mm), default=250.0
        田间持水量（毫米），默认值=250.0
    LP : float, optional
        Limit for potential ET (-), default=0.7
        潜在蒸散发限制系数，默认值=0.7
    BETA : float, optional
        Shape coefficient (-), default=2.0
        形状系数，默认值=2.0
    K0 : float, optional
        Fast flow recession coefficient (1/day), default=0.3
        快速流消退系数（1/天），默认值=0.3
    K1 : float, optional
        Slow flow recession coefficient (1/day), default=0.05
        慢速流消退系数（1/天），默认值=0.05
    K2 : float, optional
        Baseflow recession coefficient (1/day), default=0.01
        基流消退系数（1/天），默认值=0.01
    UZL : float, optional
        Threshold for fast response (mm), default=20.0
        快速响应阈值（毫米），默认值=20.0
    PERC : float, optional
        Maximum percolation rate (mm/day), default=2.0
        最大下渗率（毫米/天），默认值=2.0
    """
    
    def __init__(
        self,
        TT: float = 0.0,
        CFMAX: float = 3.5,
        FC: float = 250.0,
        LP: float = 0.7,
        BETA: float = 2.0,
        K0: float = 0.3,
        K1: float = 0.05,
        K2: float = 0.01,
        UZL: float = 20.0,
        PERC: float = 2.0
    ):
        """Initialize HBV model with parameters / 使用参数初始化HBV模型"""
        # Store parameters / 存储参数
        self.TT = TT
        self.CFMAX = CFMAX
        self.FC = FC
        self.LP = LP
        self.BETA = BETA
        self.K0 = K0
        self.K1 = K1
        self.K2 = K2
        self.UZL = UZL
        self.PERC = PERC
        
        # Initialize state variables / 初始化状态变量
        self.reset_states()
        
    def reset_states(self) -> None:
        """
        Reset all state variables to initial conditions.
        将所有状态变量重置为初始条件。
        """
        self.SNOW: float = 0.0     # Snow storage (mm) / 雪储存量
        self.SM: float = 50.0      # Soil moisture (mm) / 土壤水分
        self.SUZ: float = 10.0     # Upper zone storage (mm) / 上层蓄水量
        self.SLZ: float = 10.0     # Lower zone storage (mm) / 下层蓄水量
        
    def snow_routine(self, P: float, T: float) -> Tuple[float, float]:
        """
        Snow accumulation and melt routine.
        积雪累积与融化程序。
        
        Parameters / 参数:
        -----------
        P : float
            Precipitation (mm) / 降水量（毫米）
        T : float
            Air temperature (°C) / 气温（摄氏度）
            
        Returns / 返回:
        --------
        Pn : float
            Liquid water input to soil (mm) / 输入土壤的液态水（毫米）
        melt : float
            Snowmelt amount (mm) / 融雪量（毫米）
        """
        if T < self.TT:
            # Temperature below threshold: accumulate snow / 温度低于阈值：积雪
            self.SNOW += P
            Pn = 0.0
            melt = 0.0
        else:
            # Temperature above threshold: melt snow / 温度高于阈值：融雪
            melt = self.CFMAX * (T - self.TT)
            melt = min(melt, self.SNOW)
            self.SNOW -= melt
            Pn = P + melt
            
        return Pn, melt
    
    def soil_routine(self, Pn: float, Ep: float) -> Tuple[float, float, float]:
        """
        Soil moisture accounting routine.
        土壤水分核算程序。
        
        Parameters / 参数:
        -----------
        Pn : float
            Liquid water input (mm) / 液态水输入（毫米）
        Ep : float
            Potential evapotranspiration (mm) / 潜在蒸散发（毫米）
            
        Returns / 返回:
        --------
        Ea : float
            Actual evapotranspiration (mm) / 实际蒸散发（毫米）
        recharge : float
            Recharge to response function (mm) / 对响应函数的补给（毫米）
        SM_new : float
            Updated soil moisture (mm) / 更新后的土壤水分（毫米）
        """
        # Add precipitation to soil moisture / 将降水加入土壤水分
        self.SM += Pn
        
        # Actual evapotranspiration / 实际蒸散发
        # Limited by soil moisture availability / 受土壤水分可用性限制
        SM_ratio = min(self.SM / self.FC, 1.0)
        LP_ratio = min(self.SM / (self.LP * self.FC), 1.0)
        Ea = Ep * SM_ratio * LP_ratio
        Ea = min(Ea, self.SM)
        self.SM -= Ea
        
        # Recharge to groundwater / 地下水补给
        # Non-linear relationship with soil moisture / 与土壤水分的非线性关系
        if self.SM > self.FC:
            recharge = (self.SM - self.FC)
            self.SM = self.FC
        else:
            SM_ratio = self.SM / self.FC
            recharge = Pn * (SM_ratio ** self.BETA)
            recharge = min(recharge, self.SM)
            self.SM -= recharge
            
        return Ea, recharge, self.SM
    
    def response_routine(self, recharge: float) -> Tuple[float, float, float, float]:
        """
        Response function (upper and lower reservoirs).
        响应函数（上层和下层水库）。
        
        Parameters / 参数:
        -----------
        recharge : float
            Recharge from soil (mm) / 来自土壤的补给（毫米）
            
        Returns / 返回:
        --------
        Q0 : float
            Fast runoff component (mm) / 快速径流分量（毫米）
        Q1 : float
            Slow runoff component (mm) / 慢速径流分量（毫米）
        Q2 : float
            Baseflow component (mm) / 基流分量（毫米）
        Q_total : float
            Total discharge (mm) / 总径流量（毫米）
        """
        # Add recharge to upper zone / 将补给加入上层蓄水区
        self.SUZ += recharge
        
        # Upper zone outflows / 上层蓄水区出流
        # Fast response (only above threshold) / 快速响应（仅高于阈值时）
        Q0 = self.K0 * max(0.0, self.SUZ - self.UZL)
        # Slow response / 慢速响应
        Q1 = self.K1 * self.SUZ
        
        # Percolation to lower zone / 向下层蓄水区的渗透
        perc = self.PERC * min(1.0, self.SUZ / self.FC)
        perc = min(perc, self.SUZ)
        
        # Update upper zone storage / 更新上层蓄水量
        self.SUZ = self.SUZ - Q0 - Q1 - perc
        self.SUZ = max(0.0, self.SUZ)
        
        # Add percolation to lower zone / 将渗透水加入下层蓄水区
        self.SLZ += perc
        
        # Lower zone outflow (baseflow) / 下层蓄水区出流（基流）
        Q2 = self.K2 * self.SLZ
        
        # Update lower zone storage / 更新下层蓄水量
        self.SLZ = self.SLZ - Q2
        self.SLZ = max(0.0, self.SLZ)
        
        # Total discharge / 总径流
        Q_total = Q0 + Q1 + Q2
        
        return Q0, Q1, Q2, Q_total
    
    def run_timestep(
        self, 
        P: float, 
        T: float, 
        Ep: float
    ) -> Dict[str, float]:
        """
        Run one timestep of the HBV model.
        运行HBV模型的一个时间步。
        
        Parameters / 参数:
        -----------
        P : float
            Precipitation (mm) / 降水量（毫米）
        T : float
            Air temperature (°C) / 气温（摄氏度）
        Ep : float
            Potential evapotranspiration (mm) / 潜在蒸散发（毫米）
            
        Returns / 返回:
        --------
        result : dict
            Dictionary with simulation results / 包含模拟结果的字典
            - 'Q': total discharge (mm) / 总径流（毫米）
            - 'Q0': fast runoff (mm) / 快速径流（毫米）
            - 'Q1': slow runoff (mm) / 慢速径流（毫米）
            - 'Q2': baseflow (mm) / 基流（毫米）
            - 'Ea': actual ET (mm) / 实际蒸散发（毫米）
            - 'SM': soil moisture (mm) / 土壤水分（毫米）
            - 'SNOW': snow storage (mm) / 雪储存（毫米）
            - 'SUZ': upper zone storage (mm) / 上层蓄水（毫米）
            - 'SLZ': lower zone storage (mm) / 下层蓄水（毫米）
        """
        # 1. Snow routine / 积雪程序
        Pn, melt = self.snow_routine(P, T)
        
        # 2. Soil routine / 土壤程序
        Ea, recharge, SM = self.soil_routine(Pn, Ep)
        
        # 3. Response routine / 响应程序
        Q0, Q1, Q2, Q_total = self.response_routine(recharge)
        
        return {
            'Q': Q_total,
            'Q0': Q0,
            'Q1': Q1,
            'Q2': Q2,
            'Ea': Ea,
            'SM': SM,
            'SNOW': self.SNOW,
            'SUZ': self.SUZ,
            'SLZ': self.SLZ,
            'melt': melt
        }
    
    def run(
        self,
        P: np.ndarray,
        T: np.ndarray,
        Ep: np.ndarray,
        warmup: int = 365
    ) -> Dict[str, np.ndarray]:
        """
        Run the HBV model for a time series.
        为时间序列运行HBV模型。
        
        Parameters / 参数:
        -----------
        P : np.ndarray
            Precipitation time series (mm) / 降水时间序列（毫米）
        T : np.ndarray
            Temperature time series (°C) / 温度时间序列（摄氏度）
        Ep : np.ndarray
            Potential ET time series (mm) / 潜在蒸散发时间序列（毫米）
        warmup : int, optional
            Number of warmup timesteps, default=365
            预热时间步数，默认值=365
            
        Returns / 返回:
        --------
        results : dict
            Dictionary with arrays of simulation results / 包含模拟结果数组的字典
        """
        # Check input dimensions / 检查输入维度
        n = len(P)
        if len(T) != n or len(Ep) != n:
            raise ValueError("Input arrays must have the same length / 输入数组必须具有相同的长度")
        
        # Reset states / 重置状态
        self.reset_states()
        
        # Initialize output arrays / 初始化输出数组
        Q = np.zeros(n)
        Q0 = np.zeros(n)
        Q1 = np.zeros(n)
        Q2 = np.zeros(n)
        Ea = np.zeros(n)
        SM = np.zeros(n)
        SNOW = np.zeros(n)
        SUZ = np.zeros(n)
        SLZ = np.zeros(n)
        melt = np.zeros(n)
        
        # Run simulation / 运行模拟
        for t in range(n):
            result = self.run_timestep(P[t], T[t], Ep[t])
            Q[t] = result['Q']
            Q0[t] = result['Q0']
            Q1[t] = result['Q1']
            Q2[t] = result['Q2']
            Ea[t] = result['Ea']
            SM[t] = result['SM']
            SNOW[t] = result['SNOW']
            SUZ[t] = result['SUZ']
            SLZ[t] = result['SLZ']
            melt[t] = result['melt']
        
        # Apply warmup period / 应用预热期
        if warmup > 0 and warmup < n:
            Q[:warmup] = np.nan
        
        return {
            'Q': Q,
            'Q0': Q0,
            'Q1': Q1,
            'Q2': Q2,
            'Ea': Ea,
            'SM': SM,
            'SNOW': SNOW,
            'SUZ': SUZ,
            'SLZ': SLZ,
            'melt': melt
        }


def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency.
    计算纳什效率系数。
    
    NSE = 1 - Σ(Qobs - Qsim)² / Σ(Qobs - mean(Qobs))²
    
    Parameters / 参数:
    -----------
    observed : np.ndarray
        Observed values / 观测值
    simulated : np.ndarray
        Simulated values / 模拟值
        
    Returns / 返回:
    --------
    nse : float
        Nash-Sutcliffe Efficiency / 纳什效率系数
    """
    # Remove NaN values / 移除NaN值
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    obs = observed[mask]
    sim = simulated[mask]
    
    if len(obs) == 0:
        return np.nan
    
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    nse = 1 - (numerator / denominator)
    return nse


def calculate_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    计算均方根误差。
    
    Parameters / 参数:
    -----------
    observed : np.ndarray
        Observed values / 观测值
    simulated : np.ndarray
        Simulated values / 模拟值
        
    Returns / 返回:
    --------
    rmse : float
        Root Mean Square Error / 均方根误差
    """
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    obs = observed[mask]
    sim = simulated[mask]
    
    if len(obs) == 0:
        return np.nan
    
    rmse = np.sqrt(np.mean((obs - sim) ** 2))
    return rmse


# Demo section for educational purposes / 教学演示部分
if __name__ == '__main__':
    """
    Educational demo: HBV model simulation with synthetic data.
    教学演示：使用合成数据进行HBV模型模拟。
    """
    print("=" * 70)
    print("HBV Model Educational Demo / HBV模型教学演示")
    print("=" * 70)
    
    # Set random seed for reproducibility / 设置随机种子以确保可重现性
    np.random.seed(42)
    
    # Generate synthetic data (2 years daily) / 生成合成数据（2年日数据）
    n_days = 730
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Synthetic precipitation (mm/day) / 合成降水（毫米/天）
    # Higher in winter, lower in summer / 冬季较高，夏季较低
    P = np.random.gamma(2, 3, n_days)
    seasonal = np.sin(2 * np.pi * np.arange(n_days) / 365) * 2 + 3
    P = P * seasonal
    
    # Synthetic temperature (°C) / 合成温度（摄氏度）
    # Varies seasonally / 季节性变化
    T = 10 + 15 * np.sin(2 * np.pi * (np.arange(n_days) - 80) / 365) + np.random.normal(0, 2, n_days)
    
    # Synthetic potential ET (mm/day) / 合成潜在蒸散发（毫米/天）
    # Higher in summer / 夏季较高
    Ep = 2 + 3 * np.sin(2 * np.pi * (np.arange(n_days) - 80) / 365) + np.random.normal(0, 0.3, n_days)
    Ep = np.maximum(Ep, 0)
    
    # Initialize and run HBV model / 初始化并运行HBV模型
    print("\nInitializing HBV model with default parameters...")
    print("使用默认参数初始化HBV模型...")
    
    model = HBVModel(
        TT=0.0,
        CFMAX=3.5,
        FC=250.0,
        LP=0.7,
        BETA=2.0,
        K0=0.3,
        K1=0.05,
        K2=0.01,
        UZL=20.0,
        PERC=2.0
    )
    
    print("\nRunning simulation...")
    print("运行模拟...")
    results = model.run(P, T, Ep, warmup=365)
    
    # Calculate statistics / 计算统计量
    Q_valid = results['Q'][~np.isnan(results['Q'])]
    print(f"\nSimulation Statistics / 模拟统计:")
    print(f"  Mean discharge / 平均流量: {np.mean(Q_valid):.2f} mm/day")
    print(f"  Max discharge / 最大流量: {np.max(Q_valid):.2f} mm/day")
    print(f"  Min discharge / 最小流量: {np.min(Q_valid):.2f} mm/day")
    
    # Create visualization / 创建可视化图表
    print("\nGenerating visualization...")
    print("生成可视化图表...")
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('HBV Model Simulation Results / HBV模型模拟结果', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Plot 1: Precipitation and Temperature / 降水和温度
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.bar(dates, P, alpha=0.6, color='blue', label='Precipitation / 降水')
    ax1_twin.plot(dates, T, color='red', linewidth=1, label='Temperature / 温度')
    ax1.set_ylabel('Precipitation (mm/day)\n降水 (毫米/天)', color='blue')
    ax1_twin.set_ylabel('Temperature (°C)\n温度 (摄氏度)', color='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1.set_xlim(dates[0], dates[-1])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Meteorological Inputs / 气象输入')
    
    # Plot 2: Discharge components / 径流组分
    ax2 = axes[1]
    ax2.fill_between(dates, 0, results['Q2'], alpha=0.5, color='green', label='Baseflow / 基流 (Q2)')
    ax2.fill_between(dates, results['Q2'], results['Q2'] + results['Q1'], alpha=0.5, color='orange', label='Slow flow / 慢速流 (Q1)')
    ax2.fill_between(dates, results['Q2'] + results['Q1'], results['Q'], alpha=0.5, color='red', label='Fast flow / 快速流 (Q0)')
    ax2.set_ylabel('Discharge (mm/day)\n径流 (毫米/天)')
    ax2.set_xlim(dates[0], dates[-1])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Discharge Components / 径流组分')
    
    # Plot 3: Soil moisture and snow / 土壤水分和积雪
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    ax3.plot(dates, results['SM'], color='brown', linewidth=1.5, label='Soil Moisture / 土壤水分')
    ax3.axhline(y=model.FC, color='brown', linestyle='--', alpha=0.5, label='Field Capacity / 田间持水量')
    ax3_twin.fill_between(dates, 0, results['SNOW'], alpha=0.4, color='lightblue', label='Snow / 积雪')
    ax3.set_ylabel('Soil Moisture (mm)\n土壤水分 (毫米)', color='brown')
    ax3_twin.set_ylabel('Snow Storage (mm)\n积雪 (毫米)', color='lightblue')
    ax3.tick_params(axis='y', labelcolor='brown')
    ax3_twin.tick_params(axis='y', labelcolor='lightblue')
    ax3.set_xlim(dates[0], dates[-1])
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Soil Moisture and Snow / 土壤水分和积雪')
    
    # Plot 4: Storage states / 蓄水状态
    ax4 = axes[3]
    ax4.plot(dates, results['SUZ'], color='blue', linewidth=1.5, label='Upper Zone / 上层蓄水区')
    ax4.plot(dates, results['SLZ'], color='darkblue', linewidth=1.5, label='Lower Zone / 下层蓄水区')
    ax4.set_ylabel('Storage (mm)\n蓄水量 (毫米)')
    ax4.set_xlim(dates[0], dates[-1])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Reservoir States / 水库状态')
    
    # Plot 5: Water balance / 水量平衡
    ax5 = axes[4]
    ax5.plot(dates, P, alpha=0.7, linewidth=1, label='Precipitation / 降水', color='blue')
    ax5.plot(dates, results['Ea'], alpha=0.7, linewidth=1, label='Actual ET / 实际蒸散发', color='green')
    ax5.plot(dates, results['Q'], alpha=0.7, linewidth=1, label='Discharge / 径流', color='red')
    ax5.set_ylabel('Flux (mm/day)\n通量 (毫米/天)')
    ax5.set_xlabel('Date / 日期')
    ax5.set_xlim(dates[0], dates[-1])
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Water Balance Components / 水量平衡组分')
    
    # Format x-axis / 格式化x轴
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure / 保存图表
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'hbv_model_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to / 图表已保存至: {output_path}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully! / 演示成功完成！")
    print("=" * 70)
