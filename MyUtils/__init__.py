# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-16  ~  18:27 
# @File       : __init__.py.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#

"""
A Packet for digital signal processing
"""

from .qamdemod import *
from .scatterplot import *
from .acc import *
from .rms import *

__all__ = ['qamdemod', 'scatterplot', 'get_Pe', 'get_BER', 'rms']

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
