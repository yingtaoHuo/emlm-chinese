# emlm-chinese
基于情感掩码的中文预训练方法
## 相关idea
该任务相关带标注数据较少，尝试在预训练阶段引入下游任务的先验知识。
## 使用方法
进行预训练：
```python
python pretrain.py
```
超参数设置见args.py
运行完后，会生成pretrain_model文件夹，该文件夹下包含预训练后的模型（200000句，16进程数据并行预处理大概一整天，Titan预训练20天，这边可以进行多卡优化）
学习下游任务：
```python
python train.py
```
会有30个epoch的训练过程，相较于预训练，时间会大为缩短，大概半天。结束后，测试集的结果会自动生成。
## 目前支持的数据集
该方法在NLPCC 2013微博数据集上达到目前SOTA效果
## 可下载的相关预训练资料
预训练所需的数据需要自己下载[web_zh_2019](https://pan.baidu.com/share/init?surl=17MVHVtDbvDx30V6mxxSXg)，提取码为2rk6，当然也可以换其他相关数据集
其他领域的[预训练资料](https://zhuanlan.zhihu.com/p/163616279)
