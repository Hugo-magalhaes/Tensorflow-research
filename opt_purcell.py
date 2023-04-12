#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 2020
@author: E. G. Melo

Nanobeam bandgap optimization.
"""

# Python libraries
from __future__ import division

import argparse
import meep as mp
import numpy as np
import csv
import os
from subprocess import call
import matplotlib
matplotlib.use('agg') # Set backend for consistency and to pull pixels quickly
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

# Physical constants.
c0 = 299792458.0                    # Light speed in vacuum.

# Number of cores in MPI process
nc = 144
N_DATABASE = 1000

# Simulation paramters
wl_cav = 930                                                    # Target wavelength.
bw_cav = 60.0                                                   # Simulation bandwidth.
run_for = 400                                                   # Harminv analyssis time.
three_d = 1                                                     # 0 = 2D | 1 = 3D.
grid_res_2d = 20                                                # Grid resolution 2D.
grid_res_3d = 20                                                # Grid resolution 3D.
grid_res = grid_res_3d if three_d else grid_res_2d              # Grid resolution.
symm =1                                                         # Symmetry condition (0  - X1 Y1 Z1
                                                                #                     1  - X1 Y-1 Z1
                                                                #                     2  - X-1 Y1 Z1
                                                                #                     3  - X-1 Y-1 Z1
                                                                #                     4  - X0 Y0 Z1)  
pad = [0.0,2.0,1.0]                                             # Extra space around nanobeam (x,y,z).
pml = [3.0,1.0,1.0]                                             # PML thickness (x,y,z).
file_par = 'sim_par.csv'
file_result = 'sim_result.csv'
#host_proc = 0                                                   # Host running the simulation (0 - Local | 1 - Cluster Aguia).

# Structural parameters
n = 20                          # Nanobeam holes/2. 
w = 620                         # Nanobeam width (nm).
h = 190                         # Nanobeam height (nm).
lc = 620                        # Minimum nanocavity length (nm).
a_min = 150
a_max = 350
r_min = 0.22
r_max = 0.35
del_min = 0.5
del_max = 3.5
del_lc_max = 250

# Optimization progress
plt.style.use(['seaborn-paper', os.path.join(os.getcwd(), 'paper.mplstyle'), os.path.join(os.getcwd(), 'paper_onecol.mplstyle')])
glob_loc = 0                    # 0 - Global optimization | 1 - Local optimization
n_eval = 0                      # Number os evaluations.
f_eval = []                     # Evaluation function values.
file_prog = 'eval_prog'

# Objective function.
def obj_function(x, solution_idx):
    global n_eval
    global f_eval
    n_eval = n_eval + 1
    
    a = a_min + x[0]*(a_max - a_min)
    rx = r_min + x[1]*(r_max - r_min)
    ry = r_min + x[2]*(r_max - r_min)
    a_m = a_min + x[3]*(a - a_min)    
    rx_m = r_min + x[4]*(r_max - r_min)
    ry_m = r_min + x[5]*(r_max - r_min)
    del_a = del_min + x[6]*(del_max - del_min)
    del_rx = del_min + x[7]*(del_max - del_min)
    del_ry = del_min + x[8]*(del_max - del_min)
    del_lc = x[9]*del_lc_max

    # Writes the parameters to a file
    par = [a,rx,ry,a_m,rx_m,ry_m,del_a,del_rx,del_ry,del_lc,n,w,h,lc]
    csvfile = open(file_par,'w', newline='')
    csvsheet = csv.writer(csvfile)  
    csvsheet.writerow(par)
    csvfile.close() 

    # Parallel MEEP processing
    exec_str = "mpirun -np {np:d} python nanobeam.py -sim_type {sim:d} -wl_cav {wl_cav:.2f} -bw_cav {bw_cav:.2f} -run_for {run_for:d} -three_d {three_d:d} -grid_res {grid_res:d} -symm {symm:d} -pad_x {pad_x:.1f} -pad_y {pad_y:.1f} -pad_z {pad_z:.1f} -pml_x {pml_x:.1f} -pml_y {pml_y:.1f} -pml_z {pml_z:.1f} -file_par {file_par} -file_res {file_res} > nanobeam.out".format(np=nc,
                               sim=1, wl_cav=wl_cav, bw_cav=bw_cav, run_for=run_for, three_d=three_d, grid_res=grid_res, symm=symm,
                               pad_x=pad[0], pad_y=pad[1], pad_z=pad[2], 
                               pml_x=pml[0], pml_y=pml[1], pml_z=pml[2], 
                               file_par=file_par, file_res=file_result)  
    
    # Gets the simulation results.
    pur_fact = 0.0
    Q = 0.0
    V = 0.0
    cav_wl = 0.0
    try:
        call(exec_str, shell="True")
        
        res = np.genfromtxt(file_result, delimiter=',')
        if res.any():
            Q = res[0]                  # Mode quality factor.
            V = res[1]                  # Mode volume (a³).
            cav_wl = res[2]             # Mode resonance wavelength (a/lamb).
            pur_fact = res[3]           # Purcell Factor.
    except:
        print("MEEP error on evaluation {:d}".format(n_eval))
        

    f_eval.append(pur_fact)
    show_progress(n_eval, f_eval)
    save_progress([n_eval,a,rx,ry,a_m,rx_m,ry_m,del_a,del_rx,del_ry,del_lc,n,w,h,lc,Q,V,cav_wl,pur_fact])

    return pur_fact

def show_progress(n, f): 
    # Shows the objective function value progress
    fig = plt.figure()
    w, h = fig.get_size_inches()
    fig.set_size_inches(0.4*w,0.30*w)
                
    ax = fig.add_axes([0.1,0.1,0.85,0.8])
    divnorm = mcolors.TwoSlopeNorm(vcenter=0.5)
    ax.scatter(range(1,n + 1), f, c=f, s=12, cmap='plasma', norm=divnorm, alpha=0.75)
    ax.set_xlabel('evaluations')
    ax.set_ylabel('F_{P}')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))                
    #plt.yscale('log')
    
    plt.savefig(file_prog+'.pdf', bbox_inches='tight')
    plt.close(fig)      
    
def save_progress(data):    
    # Open file in append mode
    with open(file_prog+'.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(data)
        

def main(args):
    #global glob_loc
    global three_d
    #global host_proc                    # Host running the simulation (0 - Local | 1 - Cluster Aguia).
    global grid_res
    global file_prog
    three_d = args.three_d              # Simulation dimensionality (True = 3D).  
    run_opt = args.run_opt              # 0 - Runs a single simulation | 1 - Runs an optimization.    
    grid_res = grid_res_3d if three_d == 1 else grid_res_2d    # Grid resolution.
    #host_proc = args.host_proc          # Host running the simulation (0 - Local | 1 - Cluster Aguia).
    
    if mp.am_master():
        if run_opt == 1: 
            header = ['Eval_ID','a','rx','ry',
                      'a_min','rx_min','ry_min',
                      'del_a','del_rx','del_ry','del_lc',
                      'n','w','h','l_c',
                      'Q','V','l_0','F_P']
            
            file_prog = 'nanocav_data'
            # Save header to file
            with open(file_prog+'.csv', 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(header)             
            
            random_data = np.random.rand(N_DATABASE, 10)
            np.savetxt('random_data.csv', random_data, delimiter=',')
            for x, eval_id in zip(random_data, range(1,N_DATABASE + 1)):
                obj_function(x, eval_id)
            
        else:
            x = {}
            x["a"] = 0.5529265875607384
            x["rx"] = 0.5974660091696494
            x["ry"] = 0.8808876184889218        
            x["a_m"] = 0.6907886184318344
            x["rx_m"] = 0.9844686097753056
            x["ry_m"] = 0.1167848056532762
            x["del_a"] = 0.4653152392144785
            x["del_rx"] = 0.331521381006703
            x["del_ry"] = 0.7361458141850779       
            x["del_lc"] = 0.8623657424901587
            
            a = a_min + x["a"]*(a_max - a_min)
            rx = r_min + x["rx"]*(r_max - r_min)
            ry = r_min + x["ry"]*(r_max - r_min)
            a_m = a_min + x["a_m"]*(a_max - a_min)    
            rx_m = r_min + x["rx_m"]*(r_max - r_min)
            ry_m = r_min + x["ry_m"]*(r_max - r_min)
            del_a = del_min + x["del_a"]*(del_max - del_min)
            del_rx = del_min + x["del_rx"]*(del_max - del_min)
            del_ry = del_min + x["del_ry"]*(del_max - del_min)
            del_lc = x["del_lc"]*del_lc_max       
            
            # Writes the parameters to a file
            par = [a,rx,ry,a_m,rx_m,ry_m,del_a,del_rx,del_ry,del_lc,n,w,h,lc]
            csvfile = open(file_par,'w', newline='')
            csvsheet = csv.writer(csvfile)  
            csvsheet.writerow(par)
            csvfile.close() 
        
            exec_str = "mpirun -np {np:d} python nanobeam.py -sim_type {sim:d} -wl_cav {wl_cav:.2f} -bw_cav {bw_cav:.2f} -run_for {run_for:d} -three_d {three_d:d} -grid_res {grid_res:d} -symm {symm:d} -pad_x {pad_x:.1f} -pad_y {pad_y:.1f} -pad_z {pad_z:.1f} -pml_x {pml_x:.1f} -pml_y {pml_y:.1f} -pml_z {pml_z:.1f} -file_par {file_par} -file_res {file_res} > nanobeam.out".format(np=nc,
                                       sim=1, wl_cav=wl_cav, bw_cav=bw_cav, run_for=run_for, three_d=three_d, grid_res=grid_res, symm=symm,
                                       pad_x=pad[0], pad_y=pad[1], pad_z=pad[2], 
                                       pml_x=pml[0], pml_y=pml[1], pml_z=pml[2], 
                                       file_par=file_par, file_res=file_result) 
            
            try:
                call(exec_str, shell="True")
            except:
                print("MEEP error on run_all")
                
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='SPS Optimization.')
    parser.add_argument('-three_d', type=int, default=1, help='Simulation dimensionality (True = 3D).')
    parser.add_argument('-run_opt', type=int, default=1, help='0 - Runs a single simulation | 1 - Runs an optimization')    
    args = parser.parse_args()
    main(args)
