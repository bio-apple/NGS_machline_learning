# 机器学习-学习笔记

## 1.两个模块，选择模块

TensorFlow:由 Google 公司开发的深度学习框架，可以在任意具 备 CPU 或者 GPU 的设备上运行.
TensorFlow 的计算过程使用数据流图来表示.TensorFlow 的名字来源于其计算过程中的操作对象为多维数组，即张量(Ten-sor).
TensorFlow 1.0 版本采用静态计算图，2.0 版本之后也支持动态计算图.

PyTorch:由 Facebook、NVIDIA、Twitter等公司开发维护的深度学习框架，其前身为Lua语言的Torch.
PyTorch也是基于动态计算图的框架，在需要动态改变神经网络结构的任务中有着明显的优势.

我决定学习**TensorFlow**因为谷歌在生物信息学方向的应用有：

Nucleus:https://github.com/google/nucleus

DeepVariant:https://github.com/google/deepvariant

Poplin R, Chang P C, Alexander D, et al. A universal SNP and small-indel variant caller using deep neural networks[J]. Nature biotechnology, 2018, 36(10): 983-987.

AlphaMissense:https://github.com/google-deepmind/alphamissense

Cheng J, Novati G, Pan J, et al. Accurate proteome-wide missense variant effect prediction with AlphaMissense[J]. Science, 2023, 381(6664): eadg7492.

## 2.TensorFlow，选择CNN卷积神经网络(Convolutional Neural Network，CNN 或 ConvNet)

DeepVariant、AlphaMissense、**PrimateAI-3D**都使用了CNN

Gao H, Hamp T, Ede J, et al. The landscape of tolerated genetic variation in humans and primates[J]. Science, 2023, 380(6648): eabn8153.