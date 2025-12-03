"""
Physics-Informed Neural Network (PINN) for Hydrology
物理信息神经网络 (PINN) 水文模型

A PINN combines the power of neural networks with the constraints of physics laws.
PINN结合了神经网络的能力和物理定律的约束。

What is a PINN?
===============
Imagine you're teaching a student (the neural network) to predict river flow.

Method 1 - Pure Data-Driven: "Here's what happened before, just memorize it."
  Problem: The student might predict impossible things (water appearing from nowhere!)

Method 2 - PINN: "Here's what happened before, BUT you must follow this rule:
            Water In - Water Out = Change in Storage"
  Benefit: The predictions are physically realistic!

Mathematical Foundation:
========================

Water Balance Equation (The Core Physics Law):
  P - E - Q = dS/dt

  Where:
  - P: Precipitation (rain falling into the catchment) [mm/day]
  - E: Evapotranspiration (water evaporating) [mm/day]
  - Q: Discharge (river flow leaving the catchment) [mm/day]
  - S: Storage (water stored in soil, groundwater) [mm]
  - dS/dt: Change in storage over time [mm/day]

This is like a bank account:
  Money In - Money Out = Change in Savings
  Rain In - (Evaporation + River Flow) = Change in Soil Water

Model Architecture:
===================
1. Input Layer: [P, E, S_prev] → Receives precipitation, evapotranspiration, and previous storage
2. Hidden Layers: Multiple layers with activation functions (learning complex patterns)
3. Output Layer: [Q, dS] → Predicts discharge and storage change

Loss Function (How we train):
==============================
Total Loss = Data Loss + Physics Loss

Data Loss: How close are predictions to observations?
  L_data = Mean((Q_predicted - Q_observed)²)

Physics Loss: How well does it satisfy water balance?
  L_physics = Mean((P - E - Q_predicted - dS_predicted)²)

The physics loss prevents the model from learning nonsense!

References:
-----------
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
involving nonlinear partial differential equations.
Journal of Computational Physics, 378, 686-707.

Author: HydroLearn Teaching Team
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import warnings

# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HydrologyPINN(nn.Module):
    """
    Physics-Informed Neural Network for Rainfall-Runoff Modeling.
    物理信息神经网络用于降雨-径流建模。

    This model learns to predict discharge (Q) and storage change (dS)
    while respecting the water balance equation: P - E - Q = dS/dt

    Parameters:
    -----------
    hidden_layers : list of int
        Number of neurons in each hidden layer (default: [64, 32, 16])
        For freshmen: Think of these as "thinking layers" - more layers = more complex patterns

    activation : str
        Activation function: 'tanh', 'relu', or 'sigmoid' (default: 'tanh')
        For freshmen: This is how each neuron decides to "fire" (like brain neurons!)

    physics_weight : float
        Weight for physics loss in total loss (default: 0.1)
        Higher value = model follows physics more strictly
        For freshmen: Think of this as "how much should the student care about physics rules vs. memorizing data?"

    learning_rate : float
        Learning rate for optimizer (default: 0.001)
        For freshmen: How big steps we take when learning. Too big = overshoot, too small = too slow

    Example:
    --------
    >>> # Create model
    >>> model = HydrologyPINN(hidden_layers=[64, 32], physics_weight=0.1)
    >>>
    >>> # Train model
    >>> model.fit(P_train, E_train, Q_train, epochs=100)
    >>>
    >>> # Predict
    >>> Q_pred = model.predict(P_test, E_test)
    """

    def __init__(self,
                 hidden_layers: list = [64, 32, 16],
                 activation: str = 'tanh',
                 physics_weight: float = 0.1,
                 learning_rate: float = 0.001):

        super(HydrologyPINN, self).__init__()

        # Store hyperparameters
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers

        # Normalization parameters (will be set during training)
        self.P_mean, self.P_std = 0.0, 1.0
        self.E_mean, self.E_std = 0.0, 1.0
        self.Q_mean, self.Q_std = 0.0, 1.0
        self.S_mean, self.S_std = 0.0, 1.0

        # Build the neural network architecture
        # Input: [P, E, S_prev] = 3 features
        # Output: [Q, dS] = 2 predictions
        layers = []
        input_size = 3

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))

            # Add activation function
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}. Use 'tanh', 'relu', or 'sigmoid'")

            input_size = hidden_size

        # Output layer (no activation - we want real values)
        layers.append(nn.Linear(input_size, 2))  # Output: [Q, dS]

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

        # Initialize optimizer (will be set during training)
        self.optimizer = None

        # Training history
        self.history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': []
        }

    def forward(self, P: torch.Tensor, E: torch.Tensor, S_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters:
        -----------
        P : torch.Tensor
            Precipitation [batch_size, 1]
        E : torch.Tensor
            Evapotranspiration [batch_size, 1]
        S_prev : torch.Tensor
            Previous storage [batch_size, 1]

        Returns:
        --------
        Q : torch.Tensor
            Predicted discharge [batch_size, 1]
        dS : torch.Tensor
            Predicted storage change [batch_size, 1]
        """
        # Normalize inputs (very important for neural networks!)
        P_norm = (P - self.P_mean) / self.P_std if self.P_std > 1e-8 else (P - self.P_mean)
        E_norm = (E - self.E_mean) / self.E_std if self.E_std > 1e-8 else (E - self.E_mean)
        S_norm = (S_prev - self.S_mean) / self.S_std if self.S_std > 1e-8 else (S_prev - self.S_mean)

        # Concatenate inputs
        x = torch.cat([P_norm, E_norm, S_norm], dim=1)

        # Pass through network
        output = self.network(x)

        # Split output into Q and dS
        Q_norm = output[:, 0:1]
        dS_norm = output[:, 1:2]

        # Denormalize outputs
        Q = Q_norm * (self.Q_std + 1e-8) + self.Q_mean
        dS = dS_norm * (self.S_std + 1e-8) + self.S_mean

        # Ensure Q is non-negative (discharge can't be negative!)
        Q = torch.relu(Q)

        return Q, dS

    def compute_loss(self, P: torch.Tensor, E: torch.Tensor, S_prev: torch.Tensor,
                     Q_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = data loss + physics loss.

        This is the magic of PINN! We penalize both:
        1. Deviation from observed data (data loss)
        2. Violation of physics laws (physics loss)

        Parameters:
        -----------
        P, E, S_prev : torch.Tensor
            Input features
        Q_obs : torch.Tensor
            Observed discharge (ground truth)

        Returns:
        --------
        total_loss : torch.Tensor
            Combined loss
        data_loss : torch.Tensor
            MSE between predicted and observed Q
        physics_loss : torch.Tensor
            Violation of water balance equation
        """
        # Forward pass
        Q_pred, dS_pred = self.forward(P, E, S_prev)

        # Data Loss: How close are we to observations?
        # MSE (Mean Squared Error)
        data_loss = torch.mean((Q_pred - Q_obs) ** 2)

        # Physics Loss: Do we satisfy the water balance?
        # Water balance: P - E - Q = dS/dt
        # Residual should be close to zero!
        water_balance_residual = P - E - Q_pred - dS_pred
        physics_loss = torch.mean(water_balance_residual ** 2)

        # Total loss (weighted combination)
        total_loss = data_loss + self.physics_weight * physics_loss

        return total_loss, data_loss, physics_loss

    def fit(self,
            P: np.ndarray,
            E: np.ndarray,
            Q_obs: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True,
            validation_split: float = 0.2) -> Dict:
        """
        Train the PINN model.

        For freshmen: This is like studying for an exam!
        - Each epoch = one complete review of all the material
        - Batch size = how many examples we study at once before updating our understanding
        - Validation split = keeping some questions aside to test if we really learned or just memorized

        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series [n_samples]
        E : np.ndarray
            Evapotranspiration time series [n_samples]
        Q_obs : np.ndarray
            Observed discharge time series [n_samples]
        epochs : int
            Number of training epochs (default: 100)
        batch_size : int
            Batch size for training (default: 32)
        verbose : bool
            Print training progress (default: True)
        validation_split : float
            Fraction of data to use for validation (default: 0.2)

        Returns:
        --------
        history : dict
            Training history with losses
        """
        # Compute normalization statistics (using training data only)
        n_train = int(len(P) * (1 - validation_split))

        self.P_mean, self.P_std = np.mean(P[:n_train]), np.std(P[:n_train])
        self.E_mean, self.E_std = np.mean(E[:n_train]), np.std(E[:n_train])
        self.Q_mean, self.Q_std = np.mean(Q_obs[:n_train]), np.std(Q_obs[:n_train])

        # Initialize storage by integrating the water balance equation with observed data
        S = np.zeros(len(P))
        S[0] = np.max(Q_obs[:n_train]) * 0.5  # Initial guess for storage
        for t in range(len(P) - 1):
            # Use observed data to estimate storage change
            dS_observed = P[t] - E[t] - Q_obs[t]
            S[t + 1] = S[t] + dS_observed
            S[t + 1] = max(0, S[t + 1])  # Storage cannot be negative

        # Update normalization stats based on the calculated storage
        self.S_mean, self.S_std = np.mean(S[:n_train]), np.std(S[:n_train])

        # Prepare previous storage as an input feature
        S_prev = np.roll(S, 1)
        S_prev[0] = S[0]

        # Convert to PyTorch tensors
        P_tensor = torch.FloatTensor(P[:n_train]).reshape(-1, 1)
        E_tensor = torch.FloatTensor(E[:n_train]).reshape(-1, 1)
        S_tensor = torch.FloatTensor(S_prev[:n_train]).reshape(-1, 1)
        Q_tensor = torch.FloatTensor(Q_obs[:n_train]).reshape(-1, 1)

        # Create dataset and dataloader
        dataset = TensorDataset(P_tensor, E_tensor, S_tensor, Q_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Validation data
        if validation_split > 0:
            P_val = torch.FloatTensor(P[n_train:]).reshape(-1, 1)
            E_val = torch.FloatTensor(E[n_train:]).reshape(-1, 1)
            S_val = torch.FloatTensor(S_prev[n_train:]).reshape(-1, 1)
            Q_val = torch.FloatTensor(Q_obs[n_train:]).reshape(-1, 1)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Training loop
        if verbose:
            print("=" * 70)
            print("Starting PINN Training")
            print("=" * 70)
            print(f"Training samples: {n_train}")
            print(f"Validation samples: {len(P) - n_train}")
            print(f"Epochs: {epochs}")
            print(f"Batch size: {batch_size}")
            print(f"Physics weight: {self.physics_weight}")
            print("=" * 70)

        for epoch in range(epochs):
            # Training phase
            self.train()
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0

            for batch_P, batch_E, batch_S, batch_Q in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()

                # Compute loss
                total_loss, data_loss, physics_loss = self.compute_loss(
                    batch_P, batch_E, batch_S, batch_Q
                )

                # Backward pass
                total_loss.backward()

                # Update weights
                self.optimizer.step()

                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += physics_loss.item()

            # Average losses
            n_batches = len(dataloader)
            epoch_loss /= n_batches
            epoch_data_loss /= n_batches
            epoch_physics_loss /= n_batches

            # Store history
            self.history['total_loss'].append(epoch_loss)
            self.history['data_loss'].append(epoch_data_loss)
            self.history['physics_loss'].append(epoch_physics_loss)

            # Validation phase
            if validation_split > 0 and (epoch + 1) % 10 == 0:
                self.eval()
                with torch.no_grad():
                    val_total_loss, val_data_loss, val_physics_loss = self.compute_loss(
                        P_val, E_val, S_val, Q_val
                    )

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {epoch_loss:.6f} "
                          f"(Data: {epoch_data_loss:.6f}, Physics: {epoch_physics_loss:.6f}) | "
                          f"Val Loss: {val_total_loss.item():.6f}")
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {epoch_loss:.6f} "
                      f"(Data: {epoch_data_loss:.6f}, Physics: {epoch_physics_loss:.6f})")

        if verbose:
            print("=" * 70)
            print("Training completed!")
            print("=" * 70)

        return self.history

    def predict(self, P: np.ndarray, E: np.ndarray,
                S_initial: Optional[float] = None) -> np.ndarray:
        """
        Predict discharge for given inputs.

        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series
        E : np.ndarray
            Evapotranspiration time series
        S_initial : float, optional
            Initial storage (default: use mean from training)

        Returns:
        --------
        Q_pred : np.ndarray
            Predicted discharge
        """
        self.eval()

        # Initialize storage
        if S_initial is None:
            S_initial = self.S_mean

        S = np.zeros(len(P))
        S[0] = S_initial
        Q_pred = np.zeros(len(P))

        with torch.no_grad():
            for t in range(len(P)):
                # Get previous storage
                S_prev = S[t]

                # Prepare inputs
                P_t = torch.FloatTensor([[P[t]]])
                E_t = torch.FloatTensor([[E[t]]])
                S_t = torch.FloatTensor([[S_prev]])

                # Predict
                Q_t, dS_t = self.forward(P_t, E_t, S_t)

                # Store results
                Q_pred[t] = Q_t.item()

                # Update storage for next timestep
                if t < len(P) - 1:
                    S[t + 1] = S_prev + dS_t.item()
                    S[t + 1] = max(0, S[t + 1])  # Storage can't be negative

        return Q_pred

    def calculate_metrics(self, Q_obs: np.ndarray, Q_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.

        For freshmen: These metrics tell us "how good" our predictions are!
        - NSE: 1.0 = perfect, 0.0 = as good as just using average, <0 = worse than average
        - RMSE: Average error in mm/day (lower is better)
        - PBIAS: Percent bias (0% = perfect, positive = underestimate, negative = overestimate)

        Parameters:
        -----------
        Q_obs : np.ndarray
            Observed discharge
        Q_pred : np.ndarray
            Predicted discharge

        Returns:
        --------
        metrics : dict
            Dictionary with NSE, RMSE, PBIAS, R2
        """
        # Remove NaN values
        valid_idx = ~(np.isnan(Q_obs) | np.isnan(Q_pred))
        Q_obs_clean = Q_obs[valid_idx]
        Q_pred_clean = Q_pred[valid_idx]

        if len(Q_obs_clean) == 0:
            return {'NSE': -np.inf, 'RMSE': np.inf, 'PBIAS': np.inf, 'R2': -np.inf}

        # Nash-Sutcliffe Efficiency
        numerator = np.sum((Q_obs_clean - Q_pred_clean) ** 2)
        denominator = np.sum((Q_obs_clean - np.mean(Q_obs_clean)) ** 2)
        NSE = 1 - numerator / (denominator + 1e-8)

        # Root Mean Square Error
        RMSE = np.sqrt(np.mean((Q_obs_clean - Q_pred_clean) ** 2))

        # Percent Bias
        PBIAS = 100 * np.sum(Q_obs_clean - Q_pred_clean) / (np.sum(Q_obs_clean) + 1e-8)

        # R-squared
        correlation = np.corrcoef(Q_obs_clean, Q_pred_clean)[0, 1]
        R2 = correlation ** 2

        return {
            'NSE': NSE,
            'RMSE': RMSE,
            'PBIAS': PBIAS,
            'R2': R2
        }


def create_pinn_plots(model: HydrologyPINN, P: np.ndarray, E: np.ndarray,
                      Q_obs: np.ndarray, Q_pred: np.ndarray,
                      save_dir: str = "figures"):
    """
    Create comprehensive visualization plots for PINN results.

    Parameters:
    -----------
    model : HydrologyPINN
        Trained PINN model
    P, E : np.ndarray
        Input precipitation and evapotranspiration
    Q_obs : np.ndarray
        Observed discharge
    Q_pred : np.ndarray
        Predicted discharge
    save_dir : str
        Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create date index
    n_days = len(P)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Figure 1: Training history (Loss curves)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('PINN Training History / PINN训练历史', fontsize=16, fontweight='bold')

    epochs = range(1, len(model.history['total_loss']) + 1)

    # Total loss
    axes[0].plot(epochs, model.history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch / 训练轮次', fontweight='bold')
    axes[0].set_ylabel('Loss / 损失', fontweight='bold')
    axes[0].set_title('Total Loss / 总损失', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Data loss vs Physics loss
    axes[1].plot(epochs, model.history['data_loss'], 'r-', linewidth=2, label='Data Loss / 数据损失')
    axes[1].plot(epochs, model.history['physics_loss'], 'g-', linewidth=2, label='Physics Loss / 物理损失')
    axes[1].set_xlabel('Epoch / 训练轮次', fontweight='bold')
    axes[1].set_ylabel('Loss / 损失', fontweight='bold')
    axes[1].set_title('Data Loss vs Physics Loss / 数据损失 vs 物理损失', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Prediction results
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('PINN Prediction Results / PINN预测结果', fontsize=16, fontweight='bold')

    # Precipitation
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('Precipitation / 降水\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)

    # Discharge comparison
    axes[1].plot(dates, Q_obs, 'k-', linewidth=2, label='Observed / 观测值', alpha=0.7)
    axes[1].plot(dates, Q_pred, 'r--', linewidth=2, label='PINN Prediction / PINN预测')
    axes[1].fill_between(dates, Q_obs, alpha=0.2, color='black')
    axes[1].fill_between(dates, Q_pred, alpha=0.2, color='red')
    axes[1].set_ylabel('Discharge / 径流\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Residuals (errors)
    residuals = Q_obs - Q_pred
    axes[2].plot(dates, residuals, 'purple', linewidth=1.5)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].fill_between(dates, residuals, alpha=0.3, color='purple')
    axes[2].set_ylabel('Residuals / 残差\n(mm/day)', fontweight='bold')
    axes[2].set_xlabel('Date / 日期', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Scatter plot and metrics
    metrics = model.calculate_metrics(Q_obs, Q_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('PINN Performance Metrics / PINN性能指标', fontsize=16, fontweight='bold')

    # Scatter plot
    axes[0].scatter(Q_obs, Q_pred, alpha=0.6, s=30, c='blue', edgecolors='black', linewidth=0.5)

    # 1:1 line
    max_val = max(np.max(Q_obs), np.max(Q_pred))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line / 1:1线')

    # Add metrics text
    metrics_text = f"NSE: {metrics['NSE']:.3f}\n"
    metrics_text += f"R²: {metrics['R2']:.3f}\n"
    metrics_text += f"RMSE: {metrics['RMSE']:.3f} mm/day\n"
    metrics_text += f"PBIAS: {metrics['PBIAS']:.2f}%"

    axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes,
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                verticalalignment='top', fontweight='bold', fontsize=11)

    axes[0].set_xlabel('Observed Discharge / 观测径流 (mm/day)', fontweight='bold')
    axes[0].set_ylabel('Predicted Discharge / 预测径流 (mm/day)', fontweight='bold')
    axes[0].set_title('Observed vs Predicted / 观测 vs 预测', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Metrics bar chart
    metric_names = ['NSE', 'R²', 'PBIAS/100']
    metric_values = [metrics['NSE'], metrics['R2'], metrics['PBIAS']/100]
    colors = ['green' if v > 0.5 else 'orange' if v > 0 else 'red' for v in metric_values]

    bars = axes[1].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Metric Value / 指标值', fontweight='bold')
    axes[1].set_title('Performance Metrics / 性能指标', fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        axes[1].text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPINN visualizations saved to '{save_dir}/'")


def main():
    """
    Example usage of the PINN model with synthetic data.
    """
    print("=" * 80)
    print("Physics-Informed Neural Network (PINN) for Hydrology")
    print("物理信息神经网络 (PINN) 水文模型")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data (2 years)
    n_days = 730
    t = np.arange(n_days)

    # Precipitation (with seasonal pattern)
    seasonal_factor = 1.5 + 0.8 * np.sin(2 * np.pi * t / 365 + np.pi)
    P_base = np.random.gamma(1.5, 4, n_days) * seasonal_factor
    dry_prob = 0.6 + 0.2 * np.sin(2 * np.pi * t / 365)
    P = np.where(np.random.rand(n_days) < dry_prob, 0, P_base)

    # Add extreme events
    extreme_events = np.random.choice(n_days, size=5, replace=False)
    P[extreme_events] += np.random.gamma(5, 10, 5)

    # Evapotranspiration (with seasonal pattern)
    E_mean = 4.0
    E_amplitude = 2.5
    E = E_mean + E_amplitude * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.3, n_days)
    E = np.maximum(E, 0.5)

    # Generate "true" discharge using a simple model (for demonstration)
    # In reality, this would be observed data
    Q_true = np.zeros(n_days)
    S = np.zeros(n_days)
    S[0] = 100.0  # Initial storage

    for i in range(n_days):
        # Simple water balance
        inflow = max(0, P[i] - E[i])
        outflow = 0.1 * S[i]  # Linear reservoir
        Q_true[i] = outflow
        if i < n_days - 1:
            S[i + 1] = S[i] + inflow - outflow
            S[i + 1] = max(0, S[i + 1])

    # Add noise to create "observed" data
    Q_obs = Q_true + np.random.normal(0, 0.1 * np.std(Q_true), n_days)
    Q_obs = np.maximum(Q_obs, 0)

    print("\nData Summary:")
    print(f"  Period: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Mean precipitation: {np.mean(P):.2f} mm/day")
    print(f"  Mean evapotranspiration: {np.mean(E):.2f} mm/day")
    print(f"  Mean observed discharge: {np.mean(Q_obs):.2f} mm/day")

    # Create and train PINN model
    print("\nCreating PINN model...")
    model = HydrologyPINN(
        hidden_layers=[64, 32, 16],
        activation='tanh',
        physics_weight=0.1,
        learning_rate=0.001
    )

    print("\nTraining PINN model...")
    history = model.fit(
        P, E, Q_obs,
        epochs=200,
        batch_size=32,
        verbose=True,
        validation_split=0.2
    )

    # Make predictions
    print("\nMaking predictions...")
    Q_pred = model.predict(P, E)

    # Calculate metrics
    metrics = model.calculate_metrics(Q_obs, Q_pred)

    print("\nModel Performance:")
    print(f"  NSE (Nash-Sutcliffe Efficiency): {metrics['NSE']:.3f}")
    print(f"  R² (Coefficient of Determination): {metrics['R2']:.3f}")
    print(f"  RMSE (Root Mean Square Error): {metrics['RMSE']:.3f} mm/day")
    print(f"  PBIAS (Percent Bias): {metrics['PBIAS']:.2f}%")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_pinn_plots(model, P, E, Q_obs, Q_pred, save_dir="figures")

    print("\n" + "=" * 80)
    print("PINN demonstration completed successfully!")
    print("Check the 'figures' directory for visualizations.")
    print("=" * 80)

    return model, Q_pred, metrics


if __name__ == "__main__":
    main()
