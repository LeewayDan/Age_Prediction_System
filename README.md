# 基于DNA甲基化芯片数据进行年龄预测与疾病风险评估的可视化平台

## 概述

基于**MAPLE**模型

## 特点

- **稳健的甲基化年龄预测**：跨不同数据集准确预测表观遗传年龄
- **疾病风险评估**：评估心血管疾病（CVD）和 2 型糖尿病（T2D）的风险评分
- **平台独立性**：兼容多种测序平台和预处理方法
- **技术偏差抑制**：最大程度减少批次效应对预测的影响

## 环境配置

### 前提条件

- Python == 3.10
- R == 4.3.3
- curl == 7.58.0

### 安装

1. 运行安装脚本以创建 conda 环境并安装 Python 和 R 包

   ```bash
   ./SetupEnvironment.sh
   ```

2. 激活 conda 环境

   ```bash
   conda activate maple
   ```

3. 验证 R 包安装情况
   脚本已包含版本检查，最终输出应类似：

   ```bash
   BiocManager installed? TRUE
   Bioconductor release: 
   [1] '3.18'
   Setting options('download.file.method.GEOquery'='auto')
   Setting options('GEOquery.inmemory.gpl'=FALSE)
     Package Installed Version
   1 data.table      TRUE  1.14.6
   2      minfi      TRUE  1.48.0
   3      ENmix      TRUE  1.38.1
   4       gmqn      TRUE   0.1.0
   ```

## 模型检查点

要使用平台的模型进行预测，需要下载模型检查点压缩包，并将解压后的模型参数放入 `checkpoints` 文件夹中。

下载模型参数：[百度网盘](https://pan.baidu.com/s/1bkrgRZ8-6Ul-kwPCp6E2WQ?pwd=2026)

## 预测

1. 预处理原始 DNA 甲基化数据（IDAT → β 矩阵）

   输入：原始 IDAT 文件目录
   输出：预处理后的 β 值矩阵

   ```bash
   Rscript ./raw_process/idat_process.R \
     ./examples/RAW \
     ./examples/input_data/Beta_values.csv
   ```

2. 对预处理后的数据进行预测

   输入：预处理得到的 β 值矩阵，包含测试样本元数据的 CSV 文件
   输出：预测结果 CSV 文件

   ```bash
   python MAPLE_inference.py \
     --input_path ./examples/input_data/Beta_values.csv \
     --sample_info ./examples/input_data/test_meta.csv \
     --output_path ./examples/MAPLE_output.csv
   ```

3. 输出

模型预测结果将生成一个 CSV 文件，包含：

- 预测的表观遗传年龄
- CVD 风险评分
- T2D 风险评分
  