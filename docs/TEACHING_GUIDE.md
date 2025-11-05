# HydroLearn Teaching Guide
# HydroLearn 教学指南

## 📚 Overview / 概览

This guide provides comprehensive teaching materials and resources for using the HydroLearn repository in hydrological modeling courses.

本指南为在水文模拟课程中使用HydroLearn代码库提供综合的教学材料和资源。

---

## 🎯 Learning Outcomes / 学习成果

After completing this course, students will be able to:
完成本课程后，学生将能够：

1. **Understand Hydrological Processes** / 理解水文过程
   - Explain the water cycle and water balance equation
   - Describe different runoff generation mechanisms
   - Identify key processes in rainfall-runoff transformation

2. **Apply Hydrological Models** / 应用水文模型
   - Select appropriate models for different regions and purposes
   - Configure and run models with different input data
   - Interpret model outputs and visualizations

3. **Calibrate and Validate Models** / 率定和验证模型
   - Understand the calibration-validation workflow
   - Use optimization algorithms for parameter estimation
   - Evaluate model performance using standard metrics

4. **Analyze and Compare Models** / 分析和比较模型
   - Compare model structures and complexities
   - Assess strengths and weaknesses of different approaches
   - Select models based on data availability and objectives

---

## 📖 Course Structure / 课程结构

### Week 1-2: Introduction to Hydrological Modeling
### 第1-2周：水文模拟导论

**Topics:**
- The hydrological cycle and water balance
- Types of hydrological models
- Model input and output data
- Performance metrics (NSE, RMSE, R²)

**Materials:**
- PowerPoint: `HydroLearn_Teaching_Presentation.pptx` (Slides 1-4)
- Diagram: `water_cycle_diagram.png`
- Reading: README.md Introduction section

**Activities:**
- Review the water cycle diagram
- Run `examples.py` to see all models in action
- Discuss water balance equation: P = ET + Q + ΔS

**Assessment:**
- Quiz on hydrological concepts
- Identify components in water cycle diagram

---

### Week 3-4: Xinanjiang Model (新安江模型)
### 第3-4周：新安江模型

**Topics:**
- Saturation excess runoff mechanism
- Three-layer evapotranspiration structure
- Parabolic distribution curves
- Linear reservoir routing

**Materials:**
- PowerPoint: Slides 6-7
- Flowchart: `xinanjiang_model_flowchart.png`
- Code: `xinanjiang_model.py`

**Key Equations:**
```python
# Three-layer ET
EU = EP * (W / WUM) if W < WUM
EL = EP * (W - WM) / (WM * (C - 1)) if W >= WM

# Runoff generation (saturation excess)
if PE + A < SM:
    R = PE + A - SM + SM * (1 - (PE + A)/SM)^(1+B)
else:
    R = PE + A - SM
```

**Activities:**
1. Run `xinanjiang_model.py`
2. Modify parameters B and SM
3. Observe impact on runoff generation
4. Plot soil moisture dynamics

**Lab Exercise:**
- Implement a simple saturation excess calculator
- Compare with Xinanjiang model output
- Analyze parameter sensitivity

**Assessment:**
- Explain saturation excess mechanism
- Describe role of B parameter in parabolic curve
- Analyze model outputs for given scenario

---

### Week 5-6: HBV Model
### 第5-6周：HBV模型

**Topics:**
- Temperature-driven snow processes
- Degree-day snowmelt method
- Soil moisture accounting
- Three-component runoff generation

**Materials:**
- PowerPoint: Slides 8-9
- Flowchart: `hbv_model_flowchart.png`
- Code: `hbv_model.py`

**Key Equations:**
```python
# Snow accumulation and melt
if T < TT:
    Snow = Snow + P
else:
    Melt = CFMAX * (T - TT)

# Soil moisture and recharge
EA = PET * min(SM / (LP * FC), 1.0)
Recharge = (SM / FC)^BETA * (Rain - EA)

# Three-component runoff
Q0 = K0 * max(SUZ - UZL, 0)  # Quick runoff
Q1 = K1 * SUZ                 # Interflow
Q2 = K2 * SLZ                 # Baseflow
```

**Activities:**
1. Run `hbv_model.py` with synthetic data
2. Explore snow accumulation/melt dynamics
3. Vary temperature threshold (TT) parameter
4. Compare Q0, Q1, Q2 components

**Lab Exercise:**
- Simulate a winter period with snowmelt
- Analyze timing of peak flows
- Compare with/without snow module

**Assessment:**
- Explain degree-day method for snowmelt
- Interpret three runoff components
- Analyze seasonal flow patterns

---

### Week 7: Other Models Overview
### 第7周：其他模型概览

**Models Covered:**
1. **Tank Model** - Multi-reservoir structure
2. **GR4J** - Parsimonious 4-parameter model
3. **Sacramento** - Operational forecasting model
4. **SCS-CN + UH** - Event-based storm model

**Materials:**
- PowerPoint: Slide 10
- Comparison Table: `model_comparison_table.png`
- Code: respective `.py` files

**Activities:**
- Run each model individually
- Compare water balance across models
- Discuss parameter numbers vs. performance

---

### Week 8-9: Model Calibration and Validation
### 第8-9周：模型率定与验证

**Topics:**
- Calibration objectives and strategies
- Optimization algorithms
- Split-sample testing
- Performance evaluation metrics

**Materials:**
- PowerPoint: Slide 11
- Code: `calibration_example.py`

**Key Concepts:**
```
Calibration Period: Use 60-70% of data to find optimal parameters
Validation Period: Test on remaining 30-40% (independent data)

Performance Metrics:
- NSE (Nash-Sutcliffe Efficiency): -∞ to 1 (1 = perfect)
- RMSE (Root Mean Square Error): Lower is better
- R² (Coefficient of Determination): 0 to 1
```

**Activities:**
1. Run `calibration_example.py` with GR4J
2. Modify calibration period split
3. Compare calibration vs. validation NSE
4. Try different objective functions

**Lab Exercise:**
- Calibrate Xinanjiang model on sample dataset
- Document parameter values and NSE
- Validate on independent period
- Compare with classmates' results

**Assessment:**
- Explain why validation is necessary
- Interpret NSE values
- Discuss parameter identifiability

---

### Week 10: Model Comparison and Selection
### 第10周：模型比较与选择

**Topics:**
- Criteria for model selection
- Complexity vs. performance trade-offs
- Regional suitability
- Data requirements

**Materials:**
- PowerPoint: Slides 5, 14
- All model outputs

**Discussion Questions:**
1. When to use saturation excess vs. infiltration excess?
2. Is more parameters always better?
3. How to select a model for your catchment?

**Group Activity:**
- Each group presents one model
- Compare performance on same dataset
- Discuss advantages and limitations

---

## 🛠️ Practical Exercises / 实践练习

### Exercise 1: Water Balance Analysis
### 练习1：水量平衡分析

**Objective:** Understand and verify water balance closure

**Steps:**
1. Run any model (e.g., `hbv_model.py`)
2. Calculate total inputs: P
3. Calculate total outputs: ET + Q
4. Calculate storage change: ΔS
5. Verify: P = ET + Q + ΔS (within error tolerance)

**Deliverable:**
- Plot showing cumulative P, ET, Q over time
- Table with water balance components
- Discussion of balance error sources

---

### Exercise 2: Parameter Sensitivity Analysis
### 练习2：参数敏感性分析

**Objective:** Understand how parameters affect model outputs

**Steps:**
1. Select one model (Xinanjiang or HBV)
2. Choose 3-5 key parameters
3. Vary each parameter by ±20%, ±50%
4. Run model and record peak flow, total runoff, NSE
5. Plot parameter vs. output relationships

**Deliverable:**
- Sensitivity plots for each parameter
- Ranking of parameters by sensitivity
- Discussion of physical interpretation

---

### Exercise 3: Storm Event Simulation
### 练习3：暴雨事件模拟

**Objective:** Compare model responses to storm events

**Steps:**
1. Create synthetic storm event (high precipitation)
2. Run with different initial soil moisture conditions
3. Compare HBV vs. Xinanjiang responses
4. Analyze time to peak, peak magnitude, recession

**Deliverable:**
- Hydrograph comparison plot
- Analysis of runoff generation mechanisms
- Discussion of model differences

---

### Exercise 4: Real-World Application
### 练习4：实际应用

**Objective:** Apply models to real catchment data

**Steps:**
1. Obtain data for a real catchment (instructor provided or public)
2. Prepare input data (P, T, PET, Q_obs)
3. Calibrate 2-3 different models
4. Validate and compare performance
5. Write technical report

**Deliverable:**
- Calibration report with parameter tables
- Validation plots (simulated vs. observed)
- Model comparison and recommendation
- Technical report (5-10 pages)

---

## 📊 Visualization Guide / 可视化指南

### Understanding Model Outputs
### 理解模型输出

Each model generates comprehensive visualizations in the `figures/` directory:

1. **Time Series Plots** / 时间序列图
   - Precipitation (inverted) and discharge
   - Soil moisture dynamics
   - Runoff components (surface, interflow, baseflow)
   - Evapotranspiration

2. **Water Balance Plots** / 水量平衡图
   - Cumulative inputs vs. outputs
   - Storage changes over time
   - Component contributions (pie charts)

3. **Performance Plots** / 性能图
   - Observed vs. simulated scatter plots
   - Flow duration curves
   - Residual analysis

### Reading Flowcharts
### 阅读流程图

The flowcharts (`*_flowchart.png`) show:
- **Blue boxes**: Input data
- **Green boxes**: Process modules
- **Red/Yellow boxes**: State variables
- **Arrows**: Flow of water/information
- **Equations**: Key mathematical relationships

---

## 🎓 Assessment Rubrics / 评估标准

### Lab Report Rubric (100 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Introduction | 10 | Clear problem statement, objectives |
| Methodology | 20 | Model description, parameter selection, calibration approach |
| Results | 30 | Plots, tables, statistical analysis |
| Discussion | 25 | Interpretation, comparison, limitations |
| Conclusions | 10 | Summary, recommendations |
| References | 5 | Proper citations |

### Code Quality Rubric (50 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Functionality | 20 | Code runs without errors |
| Documentation | 15 | Comments, docstrings |
| Organization | 10 | Clear structure, meaningful names |
| Reproducibility | 5 | Clear instructions to run |

---

## 📚 Additional Resources / 额外资源

### Recommended Reading
### 推荐阅读

1. **Textbooks:**
   - Beven, K. (2012). Rainfall-Runoff Modelling: The Primer (2nd ed.)
   - Dingman, S. L. (2015). Physical Hydrology (3rd ed.)

2. **Original Papers:**
   - Zhao, R. J. (1992). The Xinanjiang model applied in China
   - Bergström, S. (1992). The HBV model - its structure and applications
   - Perrin et al. (2003). Improvement of a parsimonious model (GR4J)

3. **Online Resources:**
   - [USGS Water Science School](https://www.usgs.gov/special-topic/water-science-school)
   - [Hydrology Project Training Modules](https://hydrology-project.gov.in/)

### Datasets for Practice
### 练习数据集

1. **Included in Repository:**
   - `data/sample_data.csv` - Real catchment data
   - `data/example_teaching_dataset.csv` - Teaching examples
   - `data/hourly_forcings.csv` - Hourly meteorological data

2. **Public Datasets:**
   - [CAMELS](https://ral.ucar.edu/solutions/products/camels) - Catchment attributes and meteorology
   - [GRDC](https://www.bafg.de/GRDC/) - Global Runoff Data Centre
   - [Local hydrological bureaus / 当地水文局]

---

## 🔧 Troubleshooting / 故障排除

### Common Issues and Solutions

**Issue 1: Chinese characters not displaying**
```python
# Solution: Install proper fonts or use English-only mode
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use fallback font
```

**Issue 2: Model doesn't converge during calibration**
```python
# Solutions:
# 1. Increase number of iterations
# 2. Try different initial parameter guesses
# 3. Check data quality (missing values, outliers)
# 4. Narrow parameter bounds
```

**Issue 3: Water balance error is large**
```python
# Check:
# 1. Initial conditions are reasonable
# 2. All storages are properly initialized
# 3. No numerical instabilities in equations
# 4. Time step is appropriate
```

**Issue 4: NSE is negative**
```python
# This means model is worse than mean of observations
# Solutions:
# 1. Improve calibration
# 2. Check for data errors
# 3. Consider if model is appropriate for catchment
```

---

## 📧 Getting Help / 获取帮助

1. **Check Documentation First**
   - Read model docstrings
   - Review README.md
   - Check this teaching guide

2. **Ask Questions**
   - Open GitHub issue with specific problem
   - Include error messages and relevant code
   - Describe what you've tried

3. **Discuss with Peers**
   - Form study groups
   - Share insights (not solutions!)
   - Learn from each other's approaches

---

## 🏆 Best Practices / 最佳实践

### For Students / 给学生

1. **Start Simple**
   - Run examples first before modifying
   - Understand default parameters
   - Build complexity gradually

2. **Document Everything**
   - Comment your code
   - Keep a lab notebook
   - Record parameter experiments

3. **Validate Results**
   - Check water balance
   - Compare with literature values
   - Question unexpected results

4. **Collaborate Ethically**
   - Discuss concepts freely
   - Write your own code
   - Cite all sources

### For Instructors / 给教师

1. **Prepare Environment**
   - Test all code before class
   - Prepare backup datasets
   - Have solution code ready

2. **Engage Students**
   - Use real-world examples
   - Encourage questions
   - Relate to local hydrology

3. **Assess Fairly**
   - Provide clear rubrics
   - Give formative feedback
   - Allow revision opportunities

4. **Update Materials**
   - Incorporate new research
   - Update datasets
   - Improve visualizations

---

## 📅 Suggested Timeline / 建议时间表

### Semester Course (14 weeks)

| Week | Topic | Activities | Deliverables |
|------|-------|------------|--------------|
| 1-2 | Introduction | Lectures, run examples | Quiz |
| 3-4 | Xinanjiang Model | Lab exercises | Lab report 1 |
| 5-6 | HBV Model | Lab exercises | Lab report 2 |
| 7 | Other Models | Comparison study | Comparison table |
| 8-9 | Calibration | Hands-on calibration | Calibrated models |
| 10 | Model Selection | Group presentations | Presentation |
| 11-12 | Real-world Project | Independent work | Project proposal |
| 13 | Project Work | Independent work | Draft report |
| 14 | Final Presentations | Presentations | Final report |

---

## 🌟 Success Stories / 成功案例

### Previous Student Projects
### 以往学生项目

1. **Flood Forecasting System**
   - Calibrated HBV model for local catchment
   - Developed early warning thresholds
   - Presented to local water authority

2. **Climate Change Impact**
   - Used multiple models with climate projections
   - Assessed future water availability
   - Published in student journal

3. **Comparative Study**
   - Tested all 6 models on same catchment
   - Identified best model for region
   - Created decision-making framework

---

## 📝 Feedback / 反馈

We welcome feedback on this teaching guide and the HydroLearn repository:

- **GitHub Issues**: Report problems or suggest improvements
- **Email**: Contact course instructors
- **Pull Requests**: Contribute your own materials!

---

**Last Updated:** 2024
**Version:** 1.0
**Maintainers:** HydroLearn Teaching Team

---

## Appendix: Quick Reference / 附录：快速参考

### Model Parameter Ranges

**Xinanjiang:**
- K: 0.7-1.2 (ET coefficient)
- B: 0.1-0.4 (tension water curve)
- WM: 120-200 mm (average soil capacity)
- SM: 10-50 mm (free water capacity)

**HBV:**
- TT: -2 to 2°C (snow threshold)
- CFMAX: 1-8 mm/°C/day (degree-day factor)
- FC: 50-500 mm (field capacity)
- BETA: 1-6 (recharge shape)

**GR4J:**
- X1: 100-1200 mm (production store)
- X2: -5 to 5 mm (water exchange)
- X3: 20-300 mm (routing store)
- X4: 1-4 days (unit hydrograph base)

### Common Python Commands

```python
# Run a model
python hbv_model.py

# Run with custom parameters
python -c "from hbv_model import HBVModel; model = HBVModel(FC=250); ..."

# Generate flowcharts
python docs/model_flowcharts.py

# Create presentation
python docs/create_presentation.py

# Run all examples
python examples.py
```

### Performance Metric Interpretation

| NSE Value | Performance |
|-----------|-------------|
| 0.75-1.00 | Very good |
| 0.65-0.75 | Good |
| 0.50-0.65 | Satisfactory |
| 0.40-0.50 | Acceptable |
| < 0.40 | Unsatisfactory |

**Note:** NSE can be negative if model is worse than mean

---

**Happy Modeling! 祝学习愉快！**
