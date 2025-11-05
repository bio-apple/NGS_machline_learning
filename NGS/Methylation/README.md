# REVIEW

[Aref-Eshghi E, Abadi A B, Farhadieh M E, et al. DNA methylation and machine learning: challenges and perspective toward enhanced clinical diagnostics[J]. Clinical Epigenetics, 2025, 17(1): 1-43.](https://link.springer.com/article/10.1186/s13148-025-01967-0)

1.  特征选择 (Feature Selection)

CpG 位点的甲基化状态在本质上并非机械地相互独立；它们在空间上是相互关联的，尤其是在彼此接近时。因此需要探寻CpG位点之间相关性，如果几个CpG位点高度相关，保留一个就好。

2. DNA 甲基化数据通常不遵循正态分布，而是常呈二项分布，这使得在应用标准统计检验之前需要进行特定的转换。（在基因组上的CpG要么是甲基化要么是非甲基化因此是二项式分布），为什么存在甲基化程度
那是因为是多个细胞不同测序，会导致同一位点会存在不程度的甲基化，应该属于在 N 次采样（或 N 个分子）中，观察到 k 个甲基化分子的数量 k 就会遵循参数为 N 和 p 的二项分布 B(N,p)。

3. 

