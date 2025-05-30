# Deep-learning-of-performance-of-Li-ion-batteries-and-beyond

本项目旨在利用深度学习方法对锂离子电池及多价金属电池电极的平均电压和电容量性能进行分析与预测。核心编程语言为 Python，适合对电池性能建模感兴趣的开发者参考和使用。

## 主要内容

- 利用CGCNN对电池性能数据进行建模与预测
- 数据预处理、特征工程、模型训练与评估的完整流程
- 结果可视化与分析
- 可扩展到其它类型电池或类似时序/性能预测任务

## 快速开始

1. **克隆本仓库**
   ```bash
   git clone https://github.com/HAYES-YU/Deep-learning-of-performance-of-Li-ion-batteries-and-beyond.git
   cd Deep-learning-of-performance-of-Li-ion-batteries-and-beyond
   ```

2. **安装依赖**

   - [PyTorch](http://pytorch.org)
   - [scikit-learn](http://scikit-learn.org/stable/)
   - [pymatgen](http://pymatgen.org)
   - [numpy](http://numpy.org/)
  
   建议使用conda环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## 依赖环境

- Python 3.7+
- Jupyter Notebook
- 主要依赖库见 `requirements.txt`（如未提供，可根据 notebook 中的 `import` 行手动安装）

## 数据说明

数据来源于开源网站Materials project，可自行下载数据集。

## 参考作者

The CGCNN model is provided by Xie Tian et.al (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301). They also provide their model in github (http://github.com/txie-93/cgcnn).

Using the CGCNN transfer learning model to pridict the voltages of many kinds of metal-ion battery electrodes by Zhang Xiuying, Zhou Jun, Lv Jing, and Shen Lei https://github.com/yingyingeryu/voltages-of-battery-electrodes.

---

