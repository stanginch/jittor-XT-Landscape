
# Jittor 风格及语义引导的风景图片生成比赛

![主要结果](https://s3.bmp.ovh/imgs/2022/04/19/440f015864695c92.png)


## 简介
| 简单介绍项目背景、项目特点

本项目包含了第三届计图挑战赛计图 - 风格及语义引导的风景图片生成赛题的代码实现。

## 安装 
| 介绍基本的硬件需求、运行环境、依赖安装方法

本项目可在 1 张 4090 上运行，训练时间约为 72 小时。

#### 运行环境
- ubuntu 16.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

```

## 训练
｜ 介绍模型训练的方法

单卡训练可运行以下命令：
```
bash scripts/train.sh
```

多卡训练可以运行以下命令：
```
bash scripts/train-multigpu.sh
```

## 推理
｜ 介绍模型推理、测试、或者评估的方法

生成测试集上的结果可以运行以下命令：

```
bash scripts/test.sh
```

## 致谢
| 对参考的论文、开源库予以致谢，可选

此项目基于论文 *A Style-Based Generator Architecture for Generative Adversarial Networks* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。

## 注意事项

点击项目的“设置”，在Description一栏中添加项目描述，需要包含“jittor”字样。同时在Topics中需要添加jittor。

![image-20220419164035639](https://s3.bmp.ovh/imgs/2022/04/19/6a3aa627eab5f159.png)
