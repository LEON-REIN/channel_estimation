# 基于 OFDM 的信道估计研究

## 参数设置
- FFT 总数为 64.  
  参考于 802.11a 的频域结构. 48 个数据子载波, 4 个已知导频, 中间 1 个 DC 置零, 前 6 后 5 个虚拟子载波置零.    
  载波下标: [11, 25, 39, 53]  
  非数据位置下标：[0, 1, 2, 3, 4, 5, 11, 25, 32, 39, 53, 59, 60, 61, 62, 63]

- 调制方式为 OFDM-4QAM.  
  为方便, 0~3 映射为 [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
  
- 用于各估计方式性能测试的总 OFDM 符号数为 1000 个(而神经网络的训练码元有 10000 个).  
  即: 有 4800 个数据码元, 9600 个比特
  
- 信道为 BELLHOP 水声信道模型, 有明显的多径效应; 再加上高斯白噪声.  
  均方根时延拓展为 6.901 ms
  
- OFDM 循环前缀长度取为 30ms, 共 16 点; OFDM 符号总长 150ms, 共 80 点长.  
  载波间隔为 1/120ms = 8.33Hz,  
  基带比特率为 800 bps,   
  星座映射后波特率为 400 baud, 串行码元宽度为 2.5ms, 
  并行码元宽度为 2.5*48 = 120ms.

## 各文件说明  

- 前缀 `1~7` 的 `.py` 文件们  
  按顺序可分别独立执行的程序, 每个文件完成整个通信系统的部分功能. 便于验证各模块的准确性. 
  `data_sets` 文件夹中数据主要由这些文件生成. 
  
- All_in_One.py  
  直接完成整个通信系统. 主要用于集成运行在不同设定的参数(SNR等)下, 各信道估计方式的准确性, 便于比较.  
  
- CENet-V1~3.py  
  是本课设的核心, 主要用于开发线下训练的神经网络. 训练完后可以导出完整模型, 再用于实际的线上应用环境.  
  > 几乎都是只基于 10dB 时的仿真数据, 只为了初步探索合适的神经网络的架构!  
  > 最终版本`CENet/V3.6/20210420-085719/CENet-V3.6.h5`利用了多个信噪比的数据，数据产生利用了
  > `get_dataset.py` 的最后一部分代码，
  > 结果十分逼近完美均衡。  
  
- readme.md  
  即本文件, 用于提醒自己干过啥 QwQ  
  
- CENet 文件夹  
  存放不同版本神经网络的训练过程(基于 `Tensorboard`)，只截取代表性的一两次，其余几百次记录已删除
  
- data_sets 文件夹  
  保存独立执行文件们的输出
  
- imgs 文件夹  
  保存部分结果图片；网络结构图片保存于 CENet 文件夹内
  
- MyUtils 文件夹  
  常用函数的封装，细节说明见其 `__init__.py`；示例用法见 `All_in_One.py` 文件
  
## 备注  

Google colab 很方便! 速度比我笔记本的 GTX1070 快3倍!  

## 使用环境  

- Windows 10; Google Colab
- Tensorflow-GPU 2.3.0 (IMPORTANT!)
- Python 3.7.6
- Anaconda3 (conda 4.8.3)
- CUDA 11.2 (by conda)

  

  












