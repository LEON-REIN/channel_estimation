# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-16  ~  19:36 
# @File       : scatterplot.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#

"""
Plotting scatters
"""

import numpy as np
import matplotlib.pyplot as plt


def scatterplot(array: np.ndarray, name="Signal") -> None:
    """
    Plots scatters of complex numbers.
    :param name: str
        The name to show in the figure.
    :param array: np.ndarray, dtype==np.complex
        Array filled up with complex numbers.
    :return: None
    """

    # ax = plt.gca()

    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_position(('data', 0))  # set bottom-axis according to data: where y = 0
    # ax.spines['left'].set_position(('axes', 0.5))  # set left-axis according to 50% of bottom-axis
    plt.title(f"$Constellation\;Diagram\;of\;The\;{name}$")
    plt.xlabel(r"$In-phase\ Component$")
    plt.ylabel(r"$Quadrature\ Component$")
    plt.scatter(array.real, array.imag, s=30, c=[],
                alpha=0.7, marker='o', edgecolors='k')
    plt.show()
