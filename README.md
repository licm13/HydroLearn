# Hydrological Models / 水文模型

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### Overview

This repository contains Python implementations of classic hydrological rainfall-runoff models with comprehensive mathematical explanations, detailed comments, and practical examples. These models are widely used in hydrology for streamflow simulation, flood forecasting, and water resources management.

### Implemented Models

#### 1. **Xinanjiang Model (新安江模型)**
- **Origin**: Developed by Zhao Ren-jun (1973) at Hohai University, China
- **Type**: Conceptual, saturation excess mechanism
- **Best for**: Humid and semi-humid regions
- **Key Features**:
  - Three-layer evapotranspiration structure
  - Parabolic distribution of soil moisture capacity
  - Separates runoff into surface, interflow, and groundwater components

#### 2. **Tank Model (タンクモデル)**
- **Origin**: Developed by Sugawara (1961, 1995) in Japan
- **Type**: Conceptual, multiple reservoirs
- **Variants**: 1D, 2D, 3D (standard), 4D
- **Best for**: Various catchment types with different runoff components
- **Key Features**:
  - Vertically stacked tanks (reservoirs)
  - Side outlets for different runoff types
  - Bottom outlets for percolation
  - Flexible configuration

#### 3. **GR4J Model (Modèle du Génie Rural à 4 paramètres Journalier)**
- **Origin**: Developed by Perrin et al. (2003) at INRAE, France
- **Type**: Lumped, conceptual
- **Best for**: Daily streamflow simulation
- **Key Features**:
  - Only 4 parameters (parsimonious)
  - Production and routing stores
  - Unit hydrograph approach
  - Widely validated across different climates

#### 4. **Sacramento Model (SAC-SMA)**
- **Origin**: Developed by Burnash et al. (1973) for NWS, USA
- **Type**: Continuous soil moisture accounting
- **Best for**: Operational river forecasting
- **Key Features**:
  - Upper and lower zone structure
  - Tension water and free water storages
  - Multiple runoff components
  - Detailed soil moisture accounting

### Installation

```bash
# Clone the repository
git clone https://github.com/licm13/Hydrological-model.git
cd Hydrological-model

# Install required packages
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

### Quick Start

#### Running Individual Models

**Xinanjiang Model:**
```python
from xinanjiang_model import XinanjiangModel
import numpy as np

# Generate or load data
P = np.random.gamma(2, 5, 365)  # Precipitation (mm/day)
ET = np.ones(365) * 3.0          # Evapotranspiration (mm/day)

# Initialize and run model
model = XinanjiangModel(K=1.0, B=0.3, WM=150.0)
results = model.run(P, ET)

# Access results
discharge = results['Q']         # Total discharge
soil_moisture = results['W']     # Soil moisture
```

**Tank Model (3D):**
```python
from tank_model import TankModel3D

model = TankModel3D()
results = model.run(P, ET)
discharge = results['Q']
```

**GR4J Model:**
```python
from gr4j_model import GR4J

model = GR4J(X1=350.0, X2=0.0, X3=90.0, X4=1.7)
results = model.run(P, ET)
discharge = results['Q']
```

**Sacramento Model:**
```python
from sacramento_model import SacramentoModel

model = SacramentoModel()
results = model.run(P, ET)
discharge = results['Q']
```

#### Running All Examples

```bash
python examples.py
```

This will run comprehensive examples including:
- Model comparison
- Sensitivity analysis
- Storm event simulation
- Seasonal pattern analysis

#### Running Individual Model Demos

```bash
python xinanjiang_model.py
python tank_model.py
python gr4j_model.py
python sacramento_model.py
```

### Project Structure

```
Hydrological-model/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── xinanjiang_model.py      # Xinanjiang model implementation
├── tank_model.py            # Tank model implementation (1D/2D/3D)
├── gr4j_model.py            # GR4J model implementation
├── sacramento_model.py      # Sacramento model implementation
└── examples.py              # Comprehensive examples
```

### Model Parameters

Each model has different parameters that need to be calibrated based on observed data. See the individual model files for detailed parameter descriptions.

**General calibration workflow:**
1. Prepare observed precipitation, evapotranspiration, and streamflow data
2. Initialize model with default or estimated parameters
3. Run model and compare simulated vs. observed discharge
4. Use optimization algorithms (e.g., SCE-UA, NSGA-II) to calibrate parameters
5. Validate with independent data period

### Mathematical Documentation

Each model file contains:
- Detailed mathematical formulations
- Step-by-step equations
- Parameter descriptions and typical ranges
- References to original papers
- Implementation notes

### Data Format

**Input data should be in the following format:**

```python
# NumPy arrays
P = np.array([...])   # Precipitation (mm/day)
ET = np.array([...])  # Potential evapotranspiration (mm/day)

# Or pandas DataFrame
import pandas as pd
data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=365),
    'Precipitation': [...],
    'Evapotranspiration': [...]
})

P = data['Precipitation'].values
ET = data['Evapotranspiration'].values
```

### Example Output

```
Model Comparison Summary
================================================================================

Model                Total Q (mm)    Runoff Coef     Peak Q (mm)     Mean Q (mm)
--------------------------------------------------------------------------------
Xinanjiang           532.45          0.412           15.23           1.46
Tank_3D              498.76          0.386           12.87           1.37
GR4J                 521.34          0.403           14.56           1.43
Sacramento           489.23          0.378           13.21           1.34
```

### Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New model implementations
- Documentation improvements
- Additional examples

### References

1. **Xinanjiang Model**:
   - Zhao, R.J. (1992). The Xinanjiang model applied in China. Journal of Hydrology, 135(1-4), 371-381.

2. **Tank Model**:
   - Sugawara, M. (1995). Tank model. Computer models of watershed hydrology, 165-214.

3. **GR4J Model**:
   - Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.

4. **Sacramento Model**:
   - Burnash, R.J.C., Ferral, R.L., & McGuire, R.A. (1973). A generalized streamflow simulation system: Conceptual models for digital computers. US Department of Commerce, National Weather Service.

### License

This project is open source and available for educational and research purposes.

### Contact

For questions or suggestions, please open an issue on GitHub.

---

<a name="中文"></a>
## 中文

### 概述

本仓库包含经典水文降雨径流模型的Python实现，配有全面的数学解释、详细的注释和实用示例。这些模型广泛应用于水文学中的河流流量模拟、洪水预报和水资源管理。

### 已实现的模型

#### 1. **新安江模型**
- **来源**：由赵人俊教授（1973）在河海大学开发
- **类型**：概念性，蓄满产流机制
- **适用于**：湿润和半湿润地区
- **主要特点**：
  - 三层蒸发结构
  - 土壤蓄水容量的抛物线分布
  - 将径流分为地表径流、壤中流和地下径流

#### 2. **Tank模型（タンクモデル）**
- **来源**：由Sugawara（1961, 1995）在日本开发
- **类型**：概念性，多水库模型
- **变体**：1D、2D、3D（标准）、4D
- **适用于**：具有不同径流成分的各类流域
- **主要特点**：
  - 垂直堆叠的水箱（水库）
  - 侧出口用于不同类型的径流
  - 底部出口用于渗透
  - 灵活的配置

#### 3. **GR4J模型（四参数日模型）**
- **来源**：由Perrin等人（2003）在法国INRAE开发
- **类型**：集总式概念性模型
- **适用于**：日径流模拟
- **主要特点**：
  - 仅4个参数（简约型）
  - 产流库和汇流库
  - 单位线方法
  - 在不同气候条件下广泛验证

#### 4. **Sacramento模型（SAC-SMA）**
- **来源**：由Burnash等人（1973）为美国国家气象局开发
- **类型**：连续土壤水分核算
- **适用于**：业务化河流预报
- **主要特点**：
  - 上层和下层土壤结构
  - 张力水和自由水存储
  - 多种径流成分
  - 详细的土壤水分核算

### 安装

```bash
# 克隆仓库
git clone https://github.com/licm13/Hydrological-model.git
cd Hydrological-model

# 安装依赖包
pip install -r requirements.txt
```

### 依赖要求

- Python 3.7+
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

### 快速开始

#### 运行单个模型

**新安江模型：**
```python
from xinanjiang_model import XinanjiangModel
import numpy as np

# 生成或加载数据
P = np.random.gamma(2, 5, 365)  # 降水 (mm/天)
ET = np.ones(365) * 3.0          # 蒸发 (mm/天)

# 初始化并运行模型
model = XinanjiangModel(K=1.0, B=0.3, WM=150.0)
results = model.run(P, ET)

# 获取结果
discharge = results['Q']         # 总流量
soil_moisture = results['W']     # 土壤湿度
```

**Tank模型（3D）：**
```python
from tank_model import TankModel3D

model = TankModel3D()
results = model.run(P, ET)
discharge = results['Q']
```

**GR4J模型：**
```python
from gr4j_model import GR4J

model = GR4J(X1=350.0, X2=0.0, X3=90.0, X4=1.7)
results = model.run(P, ET)
discharge = results['Q']
```

**Sacramento模型：**
```python
from sacramento_model import SacramentoModel

model = SacramentoModel()
results = model.run(P, ET)
discharge = results['Q']
```

#### 运行所有示例

```bash
python examples.py
```

这将运行包括以下内容的综合示例：
- 模型比较
- 敏感性分析
- 暴雨事件模拟
- 季节模式分析

#### 运行单个模型演示

```bash
python xinanjiang_model.py
python tank_model.py
python gr4j_model.py
python sacramento_model.py
```

### 项目结构

```
Hydrological-model/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略文件
├── xinanjiang_model.py      # 新安江模型实现
├── tank_model.py            # Tank模型实现（1D/2D/3D）
├── gr4j_model.py            # GR4J模型实现
├── sacramento_model.py      # Sacramento模型实现
└── examples.py              # 综合示例
```

### 模型参数

每个模型都有需要基于观测数据进行率定的不同参数。详细的参数描述请参见各个模型文件。

**通用率定流程：**
1. 准备观测的降水、蒸发和流量数据
2. 使用默认或估计参数初始化模型
3. 运行模型并比较模拟与观测流量
4. 使用优化算法（如SCE-UA、NSGA-II）率定参数
5. 使用独立时段的数据进行验证

### 数学文档

每个模型文件包含：
- 详细的数学公式
- 逐步方程
- 参数描述和典型范围
- 原始论文参考
- 实现说明

### 数据格式

**输入数据应采用以下格式：**

```python
# NumPy数组
P = np.array([...])   # 降水 (mm/天)
ET = np.array([...])  # 潜在蒸发 (mm/天)

# 或pandas DataFrame
import pandas as pd
data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=365),
    'Precipitation': [...],
    'Evapotranspiration': [...]
})

P = data['Precipitation'].values
ET = data['Evapotranspiration'].values
```

### 示例输出

```
模型比较总结
================================================================================

模型                 总流量(mm)      径流系数        峰值流量(mm)    平均流量(mm)
--------------------------------------------------------------------------------
新安江模型            532.45          0.412           15.23           1.46
Tank_3D模型          498.76          0.386           12.87           1.37
GR4J模型             521.34          0.403           14.56           1.43
Sacramento模型       489.23          0.378           13.21           1.34
```

### 贡献

欢迎贡献！请随时提交pull request或开issue：
- Bug修复
- 新模型实现
- 文档改进
- 额外示例

### 参考文献

1. **新安江模型**：
   - 赵人俊 (1992). 新安江模型在中国的应用. 水文学报, 135(1-4), 371-381.

2. **Tank模型**：
   - Sugawara, M. (1995). Tank model. Computer models of watershed hydrology, 165-214.

3. **GR4J模型**：
   - Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.

4. **Sacramento模型**：
   - Burnash, R.J.C., Ferral, R.L., & McGuire, R.A. (1973). A generalized streamflow simulation system: Conceptual models for digital computers. US Department of Commerce, National Weather Service.

### 许可证

本项目为开源项目，可用于教育和研究目的。

### 联系方式

如有问题或建议，请在GitHub上开issue。
