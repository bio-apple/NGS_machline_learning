# TensorFlow Decision Forests (TF-DF) 

即可做回归又可做分类。

最主要的就是可以自动处理缺失数据，也可以自动编码分类类型数据。

唯一缺点与scikitlearn相比就是可选模型少，不支持 SVM、KNN、逻辑回归等传统 ML 模型，TF-DF还在发展开发阶段，scikitlearn相对比较成熟。

| 特性 (Feature) | TF-DF 优势 (TF-DF Advantage) | 说明 (Explanation) |
| :--- | :--- | :--- |
| **训练速度（大数据）** | 更快 (C++ 后端 + 多线程) | 使用 Yggdrasil Decision Forests 引擎，专为高性能优化 |
| **更好的默认超参** | 有 benchmark_rank1 等模板 | 开箱即用精度通常优于 scikit-learn 的默认配置 |
| **可扩展性更强** | 支持百万级样本、高维特征 | sklearn 在大数据集下效率低下 |
| **自动处理缺失值** | 支持自动分裂处理缺失 | 不需要手动填补 (sklearn 必须手动处理) |
| **对类别变量原生支持** | 内部处理 one-hot/target encoding | sklearn 要手动编码 |
| **与 TensorFlow 集成好** | 可做深度模型的一部分 (如 Wide & Deep) | sklearn 与 TF 脱节 |
| **解释性工具更强** | 内置 feature importance / 可视化 / inspector | sklearn 要外部包或手动处理 |
| **支持分布式式训练 (未来方向)** | 实验性支持在 TF Serving、TPU 上运行 | sklearn 完全本地单机 |

## 数据来源
Ames Housing 数据集（Ames Housing Dataset）是一个结构化数据集，广泛用于回归建模、特征工程和机器学习教学与基准测试。
它由爱荷华州立大学的 Dean De Cock教授创建，目的是提供比波士顿房价数据集（Boston Housing）更复杂、现代、更符合实际的问题数据。

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

**简介**

| 项目 | 内容                             |
| :--- |:-------------------------------|
| **地点** | 美国爱荷华州 Ames 市                  |
| **年份** | 2006 年首次发布，后续更新                |
| **样本数量** | **2930 条**房屋交易记录（常用版本约 1460 条） |
| **特征数量** | **79** 个解释变量+ 1 个目标变量（房价）      |
| **目标变量** | SalePrice—— 房屋最终销售价格（美元）       |

---

**特征多样性**

| 特征类型       | 示例 |
|:-----------| :--- |
| **数值型**    | 建造年份 (YearBuilt), 总面积 (GrLivArea), 卧室数量 (BedroomAbvGr) |
| **类别型**    | 房屋类型 (HouseStyle), 街道材质 (Street), 屋顶样式 (RoofStyle) |
| **有序型**    | 房屋状况评分 (OverallQual, OverallCond) |
| **布尔/特殊值** | 是否有壁炉 (Fireplaces), 是否有地下室 (BsmtQual) |

## 学习参考链接

https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf

## demo project:

[3.Boston_House_Prices-tensorflow_decision_forests.ipynb](./demo_project/3.Boston_House_Prices-tensorflow_decision_forests.ipynb)