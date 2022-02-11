# -*- coding: utf-8 -*-
# @Author: ZX

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

thickness_set = 'auto'    # Total thickness of film (um)
                        # e.g. thickness_set = 0.39
                        # If use 'auto', thickness will be read from data file

area_set = 'auto'    # Actual area of electrode (cm2)
                    # e.g. area_set = 0.0005
                    # If use 'auto', area will be read from data file

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

def _getCommonPrefix(filenames: list) -> str:
    result = ''
    for i in zip(*filenames):
        if len(set(i)) == 1:
            result += i[0]
        else:
            break
    return result


class loop:
    """Data of a loop"""
    def __init__(self) -> None:
        self.pe_str = ''
        self.p_data = None
        self.e_data = None
        self.max_volt = None
        self.max_elecfield = None
        self.thickness = None
        self.area = None
        self.pmax = None
        self.pr = None
        self.wrec = None
        self.eff = None

    def compute(self, area_set: float=None, thickness_set: float=None) -> None:
        """
        To transform pe_data to array format, 
        and calculate energy storage density and efficiency.
        """
        pe_data = np.array(np.mat(self.pe_str)).reshape(-1,4)
        self.p_data = pe_data[:, 3]
        if area_set:
            correct_rate = self.area / area_set
            self.area = area_set
            self.pmax *= correct_rate
            self.pr *= correct_rate
            self.p_data *= correct_rate
        if thickness_set:
            self.thickness = thickness_set
            self.max_elecfield *= self.max_volt / thickness_set * 10    # 10 is to turn unit kV/mm to kV/cm
        self.e_data = pe_data[:, 2] / self.thickness * 10  # 10 is to turn unit kV/mm to kV/cm

    def selectLegend(self, legend_type: str) -> str:
        """To get legend of a loop data when plotting"""
        if legend_type is None:
            legend = None
        elif legend_type == 'filename':
            legend = file[:-4]
        elif legend_type == 'volt':
            legend = str(int(max(self.e_data*self.thickness/10)/5 + 0.5) * 5) + 'V'
        else:
            legend = str(int(max(self.e_data)/100 + 0.5) * 100) + 'kV/cm'
        return legend

    def processLine(self, line: str) -> None:
        """To process data of a line to loop"""
        func = self.lineProcessFunc.get(line[0])
        if func:
            func(self, line)

    def _peLine(self, line: str) -> None:
        """To process pe-data line"""
        if line[3].isdigit():
            self.pe_str = f'{self.pe_str}{line}'

    def _VLine(self, line: str) -> None:
        """To process line starting with 'V' """
        if line.startswith('Volts:'):
            self.__voltLine(line)

    def __voltLine(self, line: str) -> None:
        """To process max voltage line"""
        self.max_volt = float(line.split('\t')[1])

    def _FLine(self, line: str) -> None:
        """To process line starting with 'F' """
        if line.startswith('Field:'):
            self.__elecfieldLine(line)

    def __elecfieldLine(self, line: str) -> None:
        """To process max electric field line"""
        self.max_elecfield = float(line[7:-8])
        # 7 and -8 are respectively the length of "Field:" and "(kV/cm)"
        
    def _PLine(self, line: str) -> None:
        """To process line starting with 'P' """
        if line.startswith('PMax ('):
            self.__pmaxLine(line)
        elif line.startswith('Pr ('):
            self.__prLine(line)

    def __pmaxLine(self, line: str) -> None:
        """To process PMax line"""
        self.pmax = float(line.split('\t')[1])
    
    def __prLine(self, line: str) -> None:
        """To process Pr line"""
        self.pr = float(line.split('\t')[1])

    def _SLine(self, line: str) -> None:
        """To process line starting with 'S' """
        if line.startswith('Sample Area ('):
            self.__areaLine(line)
        elif line.startswith('Sample Thickness ('):
            self.__thicknessLine(line)

    def __areaLine(self, line: str) -> None:
        """To process area line"""
        self.area = float(line.split('\t')[1])

    def __thicknessLine(self, line: str) -> None:
        """To process thickness line"""
        self.thickness = float(line.split('\t')[1])

    lineProcessFunc = { # Functions for processing lines
        ' ': _peLine,
        'P': _PLine,
        'S': _SLine,
        'V': _VLine,
        'F': _FLine
        }


def getData(txt_file: str, area: float, thickness: float) -> loop:
    """To get data from a txt file"""
    a_loop = loop()
    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for line in lines:
        a_loop.processLine(line)
    if type(area) != float:
        area = None
    if type(thickness) != float:
        thickness = None
    a_loop.compute(area, thickness)
    return a_loop


if __name__ == '__main__':
    txt_files = []
    txt_files = [file for file in os.listdir('./') if file.endswith('.txt')]
    for file in txt_files:
        loop_data = getData(file, area=area_set, thickness=thickness_set)
        legend = loop_data.selectLegend(legend_type)
        plt.plot(loop_data.e_data, loop_data.p_data, label=legend, linewidth=line_width)

    plt.legend(loc=legend_pos, prop={'size': legend_size})
    if output_header is None or output_header == 'auto':
        output_header = _getCommonPrefix(txt_files)
    if not output_header.endswith('_'):
        output_header += '_'
    fig_path = 'pe_' + output_header + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.' + image_type
    plt.savefig(fig_path, transparent=True)