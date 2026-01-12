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
  - use 10 fold cross validation to estimate the lambda parameter (lambda = 0.0226)

## R代码

    library(glmnet)
    # use 10 fold cross validation to estimate the lambda parameter 
    # in the training data
    glmnet.Training.CV = cv.glmnet(datMethTraining, F(Age), nfolds=10,alpha=alpha,family="gaussian")
    # The definition of the lambda parameter:
    lambda.glmnet.Training = glmnet.Training.CV$lambda.min
    # Fit the elastic net predictor to the training data
    glmnet.Training = glmnet(datMethTraining, F(Age), family="gaussian", alpha=0.5, nlambda=100)
    # Arrive at an estimate of of DNAmAge
    DNAmAgeBasedOnTraining=inverse.F(predict(glmnet.Training,datMeth,type="response",s=lambda.glmnet.Training))

## python代码

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    
    # 假设您的数据已经准备好：
    # datMethTraining : 训练集的甲基化矩阵 (样本 × CpG位点)，通常是 pandas DataFrame 或 numpy array
    # datMeth         : 要预测的完整/测试集甲基化矩阵 (与训练集相同列顺序)
    # F_Age           : 训练集的 chronological age (真实年龄)，numpy array 或 pandas Series
    # 注意：Horvath 原始论文中常用 log-linear 变换，这里我们先实现最常见的线性版本
    
    # -------------------------------
    # 重要参数对应关系（R glmnet ↔ sklearn ElasticNetCV）
    # -------------------------------
    # R: alpha        → sklearn: l1_ratio     (L1的比例，0=纯岭回归，1=纯Lasso)
    # R: lambda       → sklearn: alpha         (总惩罚强度)
    # 因此：R 中的 alpha=0.5 对应 sklearn 的 l1_ratio=0.5
    
    alpha = 0.5           # 与您 R 代码中的 alpha=0.5 对应（弹性网混合比例）
    
    # 强烈推荐：对甲基化 beta 值做标准化（glmnet 默认 standardize=TRUE）
    # sklearn 不默认做标准化，所以我们手动加 StandardScaler
    scaler = StandardScaler()
    datMethTraining_scaled = scaler.fit_transform(datMethTraining)
    datMeth_scaled         = scaler.transform(datMeth)          # 应用相同的缩放
    
    # -------------------------------
    # 1. 使用 10-fold CV 自动选择最佳 lambda (即 sklearn 的 alpha)
    # -------------------------------
    # ElasticNetCV 内置交叉验证，相当于 cv.glmnet
    cv_model = ElasticNetCV(
        l1_ratio=alpha,                 # 与 R 中的 alpha 参数对应
        cv=10,                          # 10-fold CV
        n_jobs=-1,                      # 使用所有 CPU 核心加速
        random_state=42,
        max_iter=50000,                 # 高维数据建议大一些，避免不收敛
        tol=1e-5,
        selection='random'              # 通常比 'cyclic' 更快收敛
    )
    
    cv_model.fit(datMethTraining_scaled, F_Age)
    
    # 得到 CV 选出的最佳 lambda (即 sklearn 的最佳 alpha)
    lambda_best = cv_model.alpha_
    print(f"Best lambda (alpha) selected by 10-fold CV: {lambda_best:.6f}")
    
    # -------------------------------
    # 2. 用完整训练集 + 最佳 lambda 重新拟合最终模型
    #    （这对应您 R 代码中第二次调用 glmnet）
    # -------------------------------
    # 注意：这里我们直接用 ElasticNetCV 已经拟合好的 coef_ 和 intercept_
    # 如果你想严格模拟 R 的两步流程，也可以再单独 fit 一个 ElasticNet
    final_model = cv_model  # 最常用做法：直接用 CV 拟合的模型
    
    # 或者（更接近 R 的风格）：
    from sklearn.linear_model import ElasticNet
    
    final_model = ElasticNet(
        alpha=lambda_best,           # 使用 CV 选出的最佳惩罚强度
        l1_ratio=alpha,
        max_iter=50000,
        tol=1e-5,
        random_state=42,
        selection='random'
    )
    
    final_model.fit(datMethTraining_scaled, F_Age)
    
    # -------------------------------
    # 3. 预测 DNAmAge
    # -------------------------------
    DNAmAge_pred_scaled = final_model.predict(datMeth_scaled)
    
    # Horvath 原始时钟常用反向 log-linear 变换还原年龄
    # 如果您的数据使用了 F(Age) = log(Age+1) 或类似变换，请在这里反转
    # 最常见 Horvath 变换方式示例（请根据您实际使用的 F 函数调整）：
    
    # 假设 F(x) = log(x + 1)，则反函数 inverse.F(y) = exp(y) - 1
    # DNAmAgeBasedOnTraining = np.exp(DNAmAge_pred_scaled) - 1
    
    # 如果您没做任何非线性变换（最常见情况），直接用预测值即可：
    DNAmAgeBasedOnTraining = DNAmAge_pred_scaled
    
    print("预测的 DNAmAge（前5个样本）：")
    print(DNAmAgeBasedOnTraining[:5])
    
    # -------------------------------
    # 额外：查看选择了多少个 CpG 位点（非零系数个数）
    # -------------------------------
    n_nonzero = np.sum(final_model.coef_ != 0)
    print(f"最终模型中非零系数的 CpG 位点个数: {n_nonzero} / {datMethTraining.shape[1]}")
## R vs python 对照表

| 任务类型             | R glmnet family       | Python sklearn 推荐类                          | 备注                              |
|----------------------|-----------------------|------------------------------------------------|-----------------------------------|
| 普通回归（连续目标） | gaussian             | ElasticNet / ElasticNetCV                      | 默认就是 gaussian                 |
| 二分类               | binomial             | LogisticRegression(penalty='elasticnet')       | 不是 ElasticNet 而是 Logistic     |
| 多分类               | multinomial          | LogisticRegression(multi_class='multinomial')  | —                                 |
| 计数/稀疏正值        | poisson              | PoissonRegressor                               | —                                 |
| 多目标回归           | mgaussian            | MultiTaskElasticNetCV                          | —                                 |
| 生存分析             | cox                  | 无（用 lifelines / scikit-survival）           | glmnet 专有功能                   |

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