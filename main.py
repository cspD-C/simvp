"""
main.py - SimVP 训练和测试的主入口文件

功能:
1. 定义命令行参数
2. 创建实验对象
3. 执行训练和测试流程

使用方法:
  python main.py --epochs 1000 --lr 0.001 --dataname mmnist

参数说明:
  --epochs: 训练轮数 (建议 1000-2000)
  --lr: 学习率
  --dataname: 数据集名称 (mmnist 或 taxibj)
  --hid_S: 空间隐藏维度
  --hid_T: 时间隐藏维度
  --N_S: 编码器/解码器层数
  --N_T: Mid_Xnet 层数
"""

import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 函数: create_parser - 创建命令行参数解析器
# ============================================================================
# 作用: 定义所有可配置的训练参数
# 参数分类:
#   1. 基础设置: 设备、GPU、随机种子
#   2. 数据集参数: 批次大小、数据路径、数据集名称
#   3. 模型参数: 输入形状、隐藏维度、网络层数
#   4. 训练参数: 训练轮数、学习率、日志频率
# ============================================================================


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)