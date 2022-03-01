# emlm-chinese
基于情感掩码的中文预训练方法
## 相关idea
该任务相关带标注数据较少，尝试在预训练阶段引入下游任务的先验知识。
## 使用方法
python pretrain.py % 具体参数设置间args.py
运行完后，会生成pretrain_model文件夹，该文件夹下包含预训练后的模型（10进程数据预处理大概一整天，Titan预训练20天，这边可以进行多卡优化）
python train.py
会有30个epoch的训练过程，相较于预训练，时间会大为缩短，大概半天。结束后，在测试集的结果会自动生成
## 目前支持的数据集
在NLPCC 2013数据集上达到目前SOTA效果
预训练所需的数据需要自己下载[web_zh_2019](https://pan.baidu.com/share/init?surl=17MVHVtDbvDx30V6mxxSXg)，当然也可以换其他相关数据集
