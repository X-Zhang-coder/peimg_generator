# -*- coding: utf-8 -*-
# @Author: ZX
# @Date: 2021-12-25 17:57:09 
# @Last Modified by: ZX
# @Last Modified time: 2021-12-30 23:06:45

"""
This script is to auto-generate PE-loops from Radiant txt data.

Operating environment (required):
    Python environment
    numpy
    matplotlib

If there is no environment, please install python3.x,
and then install modules 'numpy' and 'matplotlib' (Run 'pip install numpy matplotlib' in command line).

Usage:
    Put this file and a series of PE data txt files into the same dir.
    Set parameters of this script.
    Run this script.
    A graph of PE-loops will be output to the same dir.

Welcome to report any bug or new need to the author.

Wish you a smooth experiment!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os
import time

#--------------------------------------------------------------------------------------------------------------------#

#------------------#
# Vital Parameters #
#------------------#

thickness = 0.39    # Total thickness of film (um)

area_test = 0.0005  # Area of electrode input when testing PE (cm2)
area_actual = 0.0005    # Actual area of electroed (cm2)

#--------------------------------------------------------------------------------------------------------------------#

#------------------#
# Style Parameters #
#------------------#

E_range = 'auto'    # Range of electirc field (kV/cm)
                    # Input as [a, b]
                        # e.g. E_range = [-2500, 2500]
                        # e.g. E_range = 'auto'
                    # If use 'auto', the range will be adapted to data

P_range = 'auto'    # Range of polarization intensity (uC/cm2)
                    # Input as [a, b]
                        # e.g. P_range = [-80, 80]
                    # If use 'auto', the range will be adapted to data

output_header = 'auto'  # Prefix name of output image file
                        # Any string or blank is ok
                            # e.g. output_header = 'BFO_1'
                        # If use 'auto', output file will be autonamed by common prefix of inputfiles

image_type = 'svg'  # Filetype of output image
                    # The value can be selected such as 'png', 'jpg' (convenient for watching on a phone)
                    # Or 'svg' (a vector illustration type)

line_width = 1.5   # Width of loop lines

legend_type = 'elecfield'   # Type of legend
                            # If use 'volt', the legend will be the largest voltage of each loop
                            # If use 'elecfield', the legend will be the largest electric field of each loop
                            # If use 'filename', the legend will be the name of each data file
                            # If use None, there will be no legend
                                # e.g. legend_type = None

legend_pos = 'lower right'  # Position of legend in the graph
                            # The value can be ('upper/center/lower') + ('left/right/center' or 'best')
                                # e.g. legend_pos = 'lower right'
                                # e.g. legend_pos = 'center'
                                # e.g. legend_pos = 'best'
                            # If use 'best', the legend will be autoplaced to a reasonable position

legend_size = 10    # Size of legend

#--------------------------------------------------------------------------------------------------------------------#


graph_params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'bold',
        'font.size': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
        }
rcParams.update(graph_params)

plt.xlabel('Electric Filed (kV/cm)', fontdict={'weight':'bold', 'size':16})
plt.ylabel('Polarization (μC/cm²)', fontdict={'weight':'bold', 'size':16})
plt.axhline(y=0, ls='-', c='black', linewidth=1)
plt.axvline(x=0, ls='-', c='black', linewidth=1)

if type(E_range) == list:
    plt.xlim(*E_range)
if type(P_range) == list:
    plt.ylim(*P_range)

def _getCommonPrefix(filenames):
    result = ''
    for i in zip(*filenames):
        if len(set(i)) == 1:
            result += i[0]
        else:
            break
    return result

def selectLegend(legend_type):
    if legend_type is None:
        legend = None
    elif legend_type == 'filename':
        legend = file[:-4]
    elif legend_type == 'volt':
        legend = str(int(max(pe_data[:,0])/5 + 0.5) * 5) + 'V'
    else:
        legend = str(int(max(pe_data[:,0]/thickness*10)/100 + 0.5) * 100) + 'kV/cm'
    return legend

def getData(txt_file):
    with open(txt_file, 'rb') as f:
        lines = f.read()
    lines = lines.replace(b'\xbb', b'')
    lines = lines.replace(b'\xab', b'')
    with open('pe.tmp', 'wb') as g:
        g.write(lines)
    try:
        data = np.genfromtxt('pe.tmp', delimiter='\t', skip_header=51, skip_footer=12)[:, 2:]
        return data
    except:
        print('Warning: datafile error!')
        print('Data of file ', txt_file, 'are excluded!\n')
        print('Input \"q\" to exit this procedure,')
        print('Or input any other to continue.\n')
        order = input('Your order here:')
        if order.lower == 'q':
            exit(1)


if __name__ == '__main__':
    txt_files = []
    for root, dirs, files in os.walk('./'):
        txt_files = [file for file in files if file.endswith('.txt')]
        for file in txt_files:
            pe_data = getData(file)
            if pe_data is None:
                continue
            legend = selectLegend(legend_type)
            plt.plot(pe_data[:,0]/thickness*10, pe_data[:,1]*area_test/area_actual, label=legend, linewidth=line_width)

    plt.legend(loc=legend_pos, prop={'size': legend_size})
    if output_header is None or output_header == 'auto':
        output_header = _getCommonPrefix(txt_files)
    if not output_header.endswith('_'):
        output_header += '_'
    fig_path = 'pe_' + output_header + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.' + image_type
    plt.savefig(fig_path, transparent=True)
    os.remove('pe.tmp')
