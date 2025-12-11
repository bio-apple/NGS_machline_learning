## [PrimateAI-3D](https://primateai3d.basespace.illumina.com)

    PrimateAI-3D三分类问题：common variants、unknown human variants、pathogenicity

[Gao H, Hamp T, Ede J, et al. The landscape of tolerated genetic variation in humans and primates[J]. Science, 2023, 380(6648): eabn8153.](https://www.science.org/doi/10.1126/science.abn8197)


## 常见三维数据格式

| 数据类型 | 常见文件格式 | 核心内容 |
| :--- | :--- | :--- |
| **蛋白质结构** | **.pdb, .mmCIF** | 原子坐标 (X, Y, Z)、原子类型、残基名称和链标识符。 |
| **点云 (Point Cloud)** | **.ply, .pcd, .xyz** | 包含大量**点坐标** $(X, Y, Z)$，可能包括颜色 $(R, G, B)$ 或法线信息。 |
| **网格 (Mesh)** | **.obj, .stl, .off** | 包含**顶点坐标**（Vertices）和连接这些顶点的**面信息**（Faces/Triangles）。 |
| **体积数据 (Volumetric)** | **.nii, .mrc, .dicom** | 已经是以**体素形式**组织的 3D 数据，每个体素包含一个数值（如密度或信号强度）。 |

## 处理三维数据的常见 Python 模块列表
| 类别 | 模块名称 | 核心功能和用途 |
| :--- | :--- | :--- |
| **基础数值/计算** | **NumPy / SciPy** | 基础科学计算、高效的数组操作，是所有 3D 数据处理的底层依赖。 |
| **点云/几何处理** | **Open3D** | 功能全面的 3D 数据处理库，支持点云、网格的滤波、配准、重建、分割和可视化。 |
| **网格处理/CAD** | **Trimesh** | 专注于高效处理和分析**三角形网格**，支持体素化、布尔运算、截面切割。 |
| **科学可视化** | **PyVista** | 基于 VTK，专注于高性能的**科学可视化**，适用于网格、点云和体积数据。 |
| **基础可视化** | **Matplotlib (mplot3d)** | 基础的 3D 绘图扩展，用于绘制简单的三维散点图和曲面图。 |
| **深度学习/GNN** | **PyTorch / TensorFlow** | 深度学习框架，用于构建 3D 卷积网络 (3D CNN) 或处理体素化后的张量。 |
| **深度学习/GNN** | **PyTorch Geometric (PyG)** | 专注于图神经网络 (GNN) 和非结构化 3D 数据（如点云）的处理。 |
| **生物结构解析** | **Biopython / MDAnalysis** | 用于读取和解析 PDB/mmCIF 等生物大分子结构文件，提取原子坐标。 |