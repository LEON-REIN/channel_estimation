# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-16  ~  18:27 
# @File       : __init__.py.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#

"""
###################################################################################################
# Introduction of my own toolset                                                                  #
# MyUtils  -->  A Packet for digital signal processing.                                           #
#                                                                                                 #
# qamdemod: Demodulate a array of complex numbers into decimal numbers.                           #
# scatterplot: Plots scatters of complex numbers.                                                 #
# rms: Calculating the RMS Time Lag of one FIR channel.                                           #
# get_dataset: Genrating OFDM symbols & their correct labels by numpy when the SNR specified.     #
# get_onehot: Map a int matrix to a onehot matrix. The shape[0] won't change.                     #
# get_valid_data: Extraction of valid data, from 64 columns to 48 columns.                        #
# get_evaluation: Get the Pe & BER of two matrices which both have 64 columns.                    #
# get_int_from_onehot: Restore the onehot matrix to an integer matrix.                            #
#                                                                                                 #
###################################################################################################
"""

from .qamdemod import *
from .scatterplot import *
from .acc import *
from .rms import *
from .get_dataset import *

__all__ = ['qamdemod', 'scatterplot', 'get_Pe', 'get_BER', 'rms', 'get_dataset', 'get_onehot',
           'get_valid_data', 'get_evaluation', 'get_int_from_onehot']


print(
    r"""
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  -  /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

               佛祖保佑         永无 BUG
    """
)
