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

将变异检测转化为three-state (hom-ref, het, hom-alt) genotype classification problem三分类问题

Poplin R, Chang P C, Alexander D, et al. A universal SNP and small-indel variant caller using deep neural networks[J]. Nature biotechnology, 2018, 36(10): 983-987.

AlphaMissense:https://github.com/google-deepmind/alphamissense

二分类：benign or pathogenic

Cheng J, Novati G, Pan J, et al. Accurate proteome-wide missense variant effect prediction with AlphaMissense[J]. Science, 2023, 381(6664): eadg7492.

## 2.CNN卷积神经网络(Convolutional Neural Network，CNN 或 ConvNet)

DeepVariant、AlphaMissense、**PrimateAI-3D**都使用了CNN

三分类问题：common variants、unknown human variants、pathogenicity

Gao H, Hamp T, Ede J, et al. The landscape of tolerated genetic variation in humans and primates[J]. Science, 2023, 380(6648): eabn8153.


## 3.TensorFlow新手南：Keras

https://www.tensorflow.org/?hl=zh-cn


## 4.ImageNet

项目由 李飞飞（Fei-Fei Li）教授领导，她是斯坦福大学的计算机科学教授，人工智能领域的知名学者。李飞飞教授与她的团队一起创建了ImageNet，目的是为计算机视觉领域提供一个大规模的、带标签的图像数据集，推动机器学习和深度学习技术在图像识别上的发展。
是一个大型的图像数据库，广泛用于计算机视觉领域，特别是在训练和评估深度学习模型（如卷积神经网络，CNN）时。它包含了超过 1400 万张带标签的图像，涵盖了大约 2 万多个不同的类别，包含各种各样的物体，如动物、植物、建筑、交通工具等。

ImageNet 最著名的贡献之一是 ImageNet大规模视觉识别挑战赛（ILSVRC），这是一个年度竞赛，旨在测试图像分类和目标检测模型的性能。该竞赛自 2010年开始，每年都会吸引众多研究者提交他们的模型，推动了深度学习技术（尤其是CNN）的迅速发展。

ImageNet的作用：

数据集：它为计算机视觉任务提供了大量标注数据，供研究人员和开发人员训练和评估机器学习模型。
模型训练：许多成功的CNN模型（如AlexNet、VGG、ResNet等）都是基于ImageNet数据集训练的，这些模型的预训练权重常用于其他任务，如迁移学习。