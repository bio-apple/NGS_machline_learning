# 甲基化时钟

## 文献来源

- Horvath S. DNA methylation age of human tissues and cell types[J]. Genome biology, 2013, 14(10): 3156.

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