# 甲基化时钟

## 文献来源

[Horvath S. DNA methylation age of human tissues and cell types[J]. Genome biology, 2013, 14(10): 3156.](https://link.springer.com/article/10.1186/gb-2013-14-10-r115)


## 数据来源
- 8000+ 样本 
- Illumina 27K / 450K 甲基化芯片 
- 涵盖 51 种健康组织和细胞类型

## 软件包
- R package glmnet

## 数学模型
- elastic net regression
  - alpha parameter=0.5
  - lambda value was chosen using cross-validation on the training data (lambda = 0.0226)

## 补充知识

| 类别               | 模型名称                  | 适用场景（核心优势）                                                                 | 缺点/注意事项                              | sklearn实现（推荐）                  | 选择优先级（2026工业界/竞赛） |
|--------------------|---------------------------|---------------------------------------------------------------------------------------|--------------------------------------------|--------------------------------------|-------------------------------|
| 经典线性回归       | 普通最小二乘（OLS）       | 特征少（p << n）、无多重共线性、解释性要求高                                          | 高维爆炸、多重共线性时方差巨大             | LinearRegression                     | ★☆☆☆☆（只适合教学/基线）      |
|                    | 统计模型（statsmodels）   | 需要完整的统计推断（p值、置信区间、F检验等）                                          | 不能处理高维                              | statsmodels.api.OLS                  | ★★☆☆☆（学术/报告用）          |
| 带惩罚的线性回归   | Ridge（岭回归）           | 多重共线性严重、不需要变量选择                                                        | 系数不会为0                                | Ridge / RidgeCV                      | ★★★☆☆                         |
|                    | Lasso                     | 高维 + 强变量选择（系数可精确为0）                                                    | 同一组高度相关特征只随机选一个              | Lasso / LassoCV                      | ★★★★☆                         |
|                    | ElasticNet（最推荐！）    | 高维 + 特征间高度相关 + 需要变量选择（工业界/竞赛默认王者）                           | 参数比Ridge多一个（但CV自动搞定）          | ElasticNet / ElasticNetCV            | ★★★★★（首选！）               |
|                    | Adaptive Lasso            | 想进一步减少Lasso的系数估计偏差                                                       | 需要两阶段，稍微复杂                       | 无原生，用sklearn+手动实现           | ★★★☆☆                         |
|                    | SCAD / MCP                | 追求理论上最优的无偏稀疏估计（学术顶会常用）                                          | 非凸，收敛不稳定                           | 无原生（用glmnet_python等）          | ★★☆☆☆                         |
| 鲁棒回归           | HuberRegressor            | 数据有异常值（outliers）                                                              | 计算稍慢                                   | HuberRegressor                       | ★★★★☆（数据脏时必备）         |
|                    | RANSACRegressor           | 极端异常值很多（如传感器数据）                                                        | 随机性强，需要调参                         | RANSACRegressor                      | ★★★☆☆                         |
|                    | TheilSenRegressor         | 小样本+异常值+需要鲁棒斜率估计                                                        | 计算复杂度高                               | TheilSenRegressor                    | ★★☆☆☆                         |
| 分位数回归         | QuantileRegressor         | 预测分位数（中位数回归、对异方差鲁棒、预测区间）                                     | 比普通回归慢                               | QuantileRegressor                    | ★★★★☆（金融风控/销量预测）    |
| 广义线性模型       | Poisson / Gamma / Tweedie | 目标变量不是正态分布（计数、保险金额、正值连续）                                     | 需要指定分布                               | PoissonRegressor 等                  | ★★★★☆（非正态目标时）         |
| 贝叶斯线性回归     | BayesianRidge             | 需要系数后验分布、不确定性量化                                                        | 比ElasticNet慢一点                         | BayesianRidge                        | ★★★★☆（不确定性建模）         |
|                    | ARDRegression             | 自动相关性判定（更强的变量选择）                                                      | 计算慢                                     | ARDRegression                        | ★★★☆☆                         |
| 多目标回归         | MultiTaskElasticNet       | 同时预测多个相关目标（如多指标预测）                                                  | -                                          | MultiTaskElasticNetCV                | ★★★★☆                         |

一句话总结2026年工业界/天池/Kaggle最常用顺序 ElasticNetCV（带pipeline标准化） > Huber > QuantileRegressor > RidgeCV > LassoCV > 其他 几乎所有2023-2025年Kaggle结构化数据前10%方案里，线性模型部分都是ElasticNetCV打底（配合LightGBM/XGBoost stacking）。

## 拓展阅读

[Horvath S, Raj K. DNA methylation-based biomarkers and the epigenetic clock theory of ageing[J]. Nature reviews genetics, 2018, 19(6): 371-384.](https://www.nature.com/articles/s41576-018-0004-3)