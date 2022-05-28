# -*- coding: utf-8 -*-
# @Author: ZX

"""
This script is to auto-generate PE-loops from Radiant txt data.

Operating environment (required):
    Python environment
    numpy
    matplotlib
    scipy

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
from scipy import integrate
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

energy_mode = 'on'  # To choose whether to plot P_max P_r-E, W_rec, \eta-E curves
                    # Use 'on' or 'off'

loop_to_plot = 'first'    # To select which loop to plot (only for double bipolar data)
                            # 'default': All data will be plotted
                            # 'first': First loop of the data
                            # 'last': Last loop of the data
                            # 'middle': Middle loop of the data (for double bipolar, it is point 50-150)

output_header = 'auto'  # Prefix name of output image file
                        # Any string or blank is ok
                            # e.g. output_header = 'BFO_1'
                        # If use 'auto', output file will be autonamed by common prefix of inputfiles

image_type = 'svg'  # Filetype of output image
                    # The value can be selected such as 'png', 'jpg' (convenient for watching on a phone)
                    # Or 'svg' (a vector illustration type)

line_width = 1.5   # Width of loop lines

legend_type = None   # Type of legend in PE plot
                            # If use 'volt', the legend will be the largest voltage of each loop
                            # If use 'elecfield', the legend will be the largest electric field of each loop
                            # If use 'filename', the legend will be the name of each data file
                            # If use None, there will be no legend
                                # e.g. legend_type = None

legend_pos = 'lower right'  # Position of legend in PE plot
                            # The value can be ('upper/center/lower') + ('left/right/center' or 'best')
                                # e.g. legend_pos = 'lower right'
                                # e.g. legend_pos = 'center'
                                # e.g. legend_pos = 'best'
                            # If use 'best', the legend will be autoplaced to a reasonable position

#legend_size = 10    # Size of legend

#--------------------------------------------------------------------------------------------------------------------#

graph_params={'font.family':'serif',
        'font.serif':'Times New Roman',
        "mathtext.fontset":'stix',
        'font.style':'normal',
        'font.weight':'bold',
        'font.size': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': line_width,
        'legend.loc': legend_pos,
        'legend.frameon': False,
        'legend.facecolor': 'none',
        #'legend.fontsize': legend_size
        }
rcParams.update(graph_params)

def plotPE(all_loopdata:list, suffix:str) -> None:
    """Main function of PE-loop plotting"""
    _setPELayout()
    for loop_data in all_loopdata:
        legend = loop_data.selectLegend(legend_type)
        plt.plot(loop_data.e_data, loop_data.p_data, label=legend)
    plt.legend()
    fig_path = 'pe_' + suffix + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.' + image_type
    plt.savefig(fig_path, transparent=True)
    plt.cla()

def plotPandWrec(all_loopdata:list, suffix:str) -> None:
    """Plot Pmax Pr Wrec and η and save data"""
    polarization_result = plotPmaxPr(all_loopdata, suffix)
    energy_result = plotEnergyCurve(all_loopdata, suffix)
    data_header = 'Electric Field,Polarization,Polarization,Polarization,Wrec,η\n\
        kV/cm,μC/cm2,μC/cm2,μC/cm2,J/cm3,%\n\
        ,Pmax,Pr,ΔP,,\n'
    time_temp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    np.savetxt(f'wrec_{suffix}_{time_temp}.csv', \
        np.concatenate((polarization_result, energy_result)).T, \
        delimiter=',', \
        header=data_header)

def plotPmaxPr(all_loopdata:list, suffix:str) -> np.array:
    """Main function of Pmax Pr-Electric field curves"""
    plt.xlabel('Electric Filed (kV/cm)', fontdict={'weight':'bold', 'size':16})
    plt.ylabel('Polarization (μC/cm²)', fontdict={'weight':'bold', 'size':16})

    field_data = np.array([loop.max_elecfield for loop in all_loopdata])
    pmax_data = np.array([loop.pmax for loop in all_loopdata])
    pr_data = np.array([loop.pr for loop in all_loopdata])
    delta_p = pmax_data - pr_data

    plt.xlim(0, max(field_data)*1.05)
    plt.ylim(0, max(pmax_data)*1.05)

    plt.plot(field_data, pmax_data, marker='s', color='k', label=r'$P_{max}$')
    plt.plot(field_data, pr_data, marker='o', color='r', label=r'$P_{r}$')
    plt.plot(field_data, delta_p, marker='^', color='b', label=r'$\Delta P$')
    plt.legend(loc='upper left')

    fig_path = 'pmaxpr_' + suffix + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.' + image_type
    plt.savefig(fig_path, transparent=True)
    plt.cla()
    return np.array([field_data, pmax_data, pr_data, delta_p])

def plotEnergyCurve(all_loopdata:list, suffix:str) -> np.array:
    """Main function of Wrec η-Electric filed curves"""
    field_data = np.array([loop.max_elecfield for loop in all_loopdata])
    wrec_data = np.array([loop.wrec for loop in all_loopdata])
    eff_data = np.array([loop.eff for loop in all_loopdata])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.spines['left'].set_color('r')
    ax1.tick_params(axis='y', colors='r')
    ax1.set_xlabel('Electric Filed (kV/cm)', fontdict={'weight':'bold', 'size':16})
    ax1.set_ylabel('$W_{rec}$ (J/cm$^3$)', fontdict={'weight':'bold', 'size':16, 'color':'r'})
    ax1.set_xlim(0, max(field_data)*1.05)
    ax1.set_ylim(0, max(wrec_data)*1.05)
    ax1.plot(field_data, wrec_data, marker='s', color='r')
    ax2 = ax1.twinx()
    ax2.spines['right'].set_color('b')
    ax2.spines['left'].set_color('r')
    ax2.tick_params(axis='y', colors='b')
    ax2.set_ylabel('$\eta$ (%)', fontdict={'weight':'bold', 'size':16, 'color':'b'})
    ax2.set_ylim(0, 100)
    ax2.plot(field_data, eff_data*100, marker='o', color='b')
    fig_path = 'wrec_' + suffix + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.' + image_type
    plt.savefig(fig_path, transparent=True)
    plt.cla()
    return np.array([wrec_data, eff_data])
    
def _setPELayout() -> None:
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
    def __init__(self, file_dir: str='', area: float=None, thickness: float=None) -> None:
        self.file_dir = file_dir
        self.file_name = os.path.basename(file_dir)
        self.testmode = ''
        self.fieldmode = False
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
        self.point_number = None
        self.readData(area_set, thickness_set)
        
    def processLine(self, line: str) -> None:
        """To process data of a line to loop"""
        func = self.lineProcessFunc.get(line[0])
        if func:
            func(self, line)

    def readData(self, area_set: float=None, thickness_set: float=None) -> None:
        """
        To transform pe_data to array format, 
        and calculate energy storage density and efficiency.
        """
        with open(self.file_dir, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in lines:
            self.processLine(line)
        if type(area_set) != float and type(area_set) != int:
            area_set = None
        if type(thickness_set) != float and type(thickness_set) != int:
            thickness_set = None
        self._computePE(area_set, thickness_set)
        self._selectLoop()
        self._computeEnergy()

    def selectLegend(self, legend_type: str) -> str:
        """To get legend of a loop data when plotting"""
        if legend_type is None:
            legend = None
        elif legend_type == 'filename':
            legend = self.file_name[:-4]
        elif legend_type == 'volt':
            legend = str(int(max(self.e_data*self.thickness/10)/5 + 0.5) * 5) + 'V'
        else:
            legend = str(int(max(self.e_data)/100 + 0.5) * 100) + 'kV/cm'
        return legend

    def _computePE(self, area_set: float=None, thickness_set: float=None) -> None:
        """PE data computation"""
        pe_data = np.array(np.mat(self.pe_str)).reshape(-1,4)
        self.p_data = pe_data[:, 3]
        if area_set:
            correct_rate = self.area / area_set
            self.area = area_set
            self.pmax *= correct_rate
            self.pr *= correct_rate
            self.p_data *= correct_rate
        if thickness_set:
            self.max_elecfield *= self.thickness / thickness_set
            self.thickness = thickness_set
        if self.fieldmode:
            self.e_data = pe_data[:, 2]
        else:
            self.e_data = pe_data[:, 2] / self.thickness * 10  # 10 is to turn unit kV/mm to kV/cm

    def _selectLoop(self) -> None:
        """To select which loop of data to plot"""
        if self.testmode == 'Double Bipolar':
            if loop_to_plot is None or loop_to_plot == 'default':
                return
            half_point = self.point_number//2
            if loop_to_plot == 'first':
                self.p_data = self.p_data[:half_point + 1]
                self.e_data = self.e_data[:half_point + 1]
            elif loop_to_plot == 'last':
                self.p_data = self.p_data[half_point:]
                self.e_data = self.e_data[half_point:]
            elif loop_to_plot == 'middle':
                quarter1_point = self.point_number//4
                quarter3_point = self.point_number - self.point_number//4
                self.p_data = -self.p_data[quarter1_point:quarter3_point]
                self.e_data = -self.e_data[quarter1_point:quarter3_point]

    def _computeEnergy(self) -> None:
        """Wrec and efficiency computation"""
        p_data = self.p_data
        e_data = self.e_data
        start_point = 0
        for i in range(0, self.point_number):
            if p_data[i] + p_data[i+1] > 0:
                start_point = i
                break
        max_point = 0
        for i in range(0, self.point_number):
            if p_data[i+1] < p_data[i]:
                max_point = i
                break
        back_point = 0
        for i in range(max_point+1, self.point_number):
            if e_data[i-1] + e_data[i] < 0:
                back_point = i
                break
        p_charge = p_data[start_point:max_point+1]
        e_charge = e_data[start_point:max_point+1]
        p_rec = p_data[max_point:back_point]
        e_rec = e_data[max_point:back_point]
        w_rec = -integrate.trapz(e_rec, p_rec) / 1000
        w_all = integrate.trapz(e_charge, p_charge) / 1000
        self.wrec = w_rec
        self.eff = w_rec/w_all

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
        elif line.startswith('Point'):
            self.__pointLine(line)
        elif line.startswith('Profile:'):
            self.__profileLine(line)

    def __pmaxLine(self, line: str) -> None:
        """To process PMax line"""
        self.pmax = float(line.split('\t')[1])
    
    def __prLine(self, line: str) -> None:
        """To process Pr line"""
        self.pr = float(line.split('\t')[1])
    
    def __profileLine(self, line: str) -> None:
        """To read test mode"""
        self.testmode = line.split('\t')[1].strip()

    def __pointLine(self, line:str) -> None:
        """To process lines beginning with 'point'"""
        if line[5] == 's':
            self.___pointsLine(line)
        elif line[5] == '\t':
            self.___pointModeLine(line)

    def ___pointsLine(self, line:str) -> None:
        """To read number of points"""
        self.point_number = int(line.split('\t')[1])

    def ___pointModeLine(self, line:str) -> None:
        """To detect data mode, volt or elecfield"""
        if line.split('\t')[2].startswith('Field'):
            self.fieldmode = True

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


if __name__ == '__main__':
    txt_files = [file for file in os.listdir('./') if file.endswith('.txt')]
    all_loopdata = [loop(file, area_set, thickness_set) for file in txt_files]
    all_loopdata.sort(key=lambda x: x.max_elecfield)
    if output_header is None or output_header == 'auto':
        output_header = _getCommonPrefix(txt_files)
    if not output_header.endswith('_'):
        output_header += '_'
    plotPE(all_loopdata, output_header)
    if energy_mode == 'on':
        plotPandWrec(all_loopdata, output_header)
