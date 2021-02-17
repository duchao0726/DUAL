# Deep Uncertainty-Aware Learning (DUAL)

Code for reproducing most of the results in the paper:

[Exploration in Online Advertising Systems with Deep Uncertainty-Aware Learning](https://arxiv.org/abs/2012.02298). Chao Du, Zhifeng Gao, Shuo Yuan, Lining Gao, Ziyan Li, Yifan Zeng, Xiaoqiang Zhu, Jian Xu, Kun Gai and Kuang-chih Lee, 2020.

:warning: The code is still under development.

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.5 LTS
- CPU: Intel(R) Xeon(R) Platinum 8163
- GPU: N/A
- Python: 2.7.17
- tensorflow: 1.14.0
- numpy: 1.16.6

## Datasets
- The data can be downloaded from [`http://ml.cs.tsinghua.edu.cn/~chaodu/static/files/DUAL-data.tar.gz`](http://ml.cs.tsinghua.edu.cn/~chaodu/static/files/DUAL-data.tar.gz).
- After downloading, run `tar -zxvf DUAL-data.tar.gz` to extract the data files.
- The `data` folder should be placed in the base directory of the project.

## Example

We provide a convenient experiment launcher to produce results using multiple different seeds.

Try `python batchrunner_dnn.py` to train the `DNN` architecture with DUAL using seeds [0 - 31].

If everything goes well, this should reproduce the result "$0.7755 \pm 0.0020$" in Table 1.

## Acknowledgement

Our code is developed based on the [mouna99/dien](https://github.com/mouna99/dien) project.