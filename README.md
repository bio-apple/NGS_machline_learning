# 机器学习

## 1.传统机器学习的数据处理流程

![tradition](./pic/tradition.png)

## 2.深度学习的数据处理流程

为了学习一种好的表示，需要构建具有一定“深度”的模型，并通过学习算法 来让模型自动学习出好的特征表示(从底层特征，到中层特征，再到高层特征)， 
从而最终提升预测模型的准确率.所谓“深度”是指原始数据进行非线性特征转换的次数.深度学习采用的模型主要是*神经网络模型*

![deep_learning](pic/deep_learning.png)

## 3.机器学习分类主要有

|学习类型| 	数据要求          | 	任务类型  |	主要用途|
|-------|----------------|--------|-|
|监督学习	| 大量带标签数据        | 	分类(Classification)、回归(Regression)	 |明确的预测任务|
|无监督学习	| 仅需无标签数据        | 	聚类(Clustering)、降维(DimensionalityReduction)|模式发现、数据结构探索|
|半监督学习	| 少量标签 + 大量无标签数据 | 	分类	   |在标签稀缺但无标签丰富的场景|

| 模块   |	解释	|常用算法	|常见应用|
|------|----------------|--------|-|
| 分类	  |识别某对象属于哪个类别|	SVM、最近邻、随机森林|	垃圾邮件识别、图像识别|
| 回归	  |预测目标变量的连续值|	线性回归、SVR、随机森林回归|	房价预测、温度预测、销售预测|
| 聚类	  |将相似的数据划分到同一组|	K均值、DBSCAN、层次聚类|	客户分群、图像分割、基因表达分析|
| 降维	  |减少特征维度，保留数据主要信息|	PCA、SVD、Kernel PCA|	数据可视化、数据压缩、特征提取|
| 模型选择 |	选择性能最优的模型或参数|	网格搜索、随机搜索、交叉验证|	模型优化、性能评估、超参数调整|
| 预处理  |	对数据进行清理、变换或编码，便于模型使用|	标准化、归一化、编码、缺失值填补|	数据归一化、分类特征处理、缺失值填补|

## 4.[scikit-learn](https://scikit-learn.org) 、[TensorFlow](https://www.tensorflow.org/?hl=zh-cn) and [PyTorch](https://pytorch.ac.cn)

**scikit-learn:A set of python modules for machine learning and data mining.https://scikit-learn.org/stable/**
<pre>
pip3 install -U scikit-learn
</pre>
是一个通用的机器学习库，提供了包括分类、回归、聚类等在内的一系列传统机器学习算法。它更侧重于特征工程，需要用户自行对数据进行处理，如选择特征、压缩维度、转换格式等
适合中小型、实用的机器学习项目，尤其是那些数据量不大但需要手动处理数据并选择合适模型的项目。这类项目往往在CPU上就可以完成，对硬件要求相对较低。

**TensorFlow:an open source machine learning framework for everyone.https://www.tensorflow.org/**

由 Google 公司开发的**深度学习框架**，可以在任意具备CPU或者GPU的设备上运行.
TensorFlow 的计算过程使用数据流图来表示.TensorFlow 的名字来源于其计算过程中的操作对象为多维数组，即张量(Tensor).
TensorFlow 1.0 版本采用静态计算图，2.0 版本之后也支持动态计算图.
<pre>
#直接安装tensorflow_decision_forests会自动兼容安装tensorflow、pandas、numpy
pip3 install tensorflow_decision_forests --upgrade
</pre>

**PyTorch**:由 Facebook、NVIDIA、Twitter等公司开发维护的**深度学习框架**，其前身为Lua语言的Torch.
PyTorch也是基于动态计算图的框架，在需要动态改变神经网络结构的任务中有着明显的优势.

## 6.seaborn: statistical data visualization

<pre>
pip3 install seaborn #会附带安装matplotlib
</pre>

## 5.生物信息学中的机器学习

Google(**DeepVariant、AlphaMissense**)与Illumina(**PrimateAI-3D**)开发的生物信息工具都利用了，
卷积神经网络(**Convolutional Neural Network,CNN 或 ConvNet**)

- [Nucleus](./NGS/Nucleus/README.md)

- [DeepVariant](./NGS/DeepVariant/README.md)

- [AlphaMissense](./NGS/AlphaMissense/README.md)

- [PrimateAI-3D](./NGS/PrimateAI-3D/README.md)
