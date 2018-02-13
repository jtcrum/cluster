import pandas as pd
import numpy as np
import copy
from math import *
import os
import random
from collections import defaultdict
import csv
from functools import reduce
from copy import deepcopy
import pickle
from matplotlib import pyplot as plt
from ase import Atoms
from ase.io import read
from ase.io import write
from ase.visualize import view
import itertools

def frac_to_xyz(axes_xyz, frac_coor):
    '''
    function to convert fractional coordinates to xyz coordinates
    '''
    return np.dot(frac_coor, axes_xyz)

def xyz_to_frac(axes_xyz, xyz_coor):
    '''
    function to convert xyz coordinates to fractional coordinates
    '''
    return np.dot(xyz_coor, np.linalg.inv(axes_xyz))

def frac_to_uvw(axes_abc, frac_coor):
    '''
    function to convert fractional(abc) coordinates to uvw coordinates
    '''
    return np.dot(frac_coor, np.linalg.inv(axes_abc))

def uvw_to_frac(axes_abc, uvw_coor):
    '''
    function to convert uvw coordinates to fractional(abc) coordinates
    '''
    return np.dot(uvw_coor, axes_abc)

def find_site_index_frac(frac, df):
    '''
    function to find site index from fractional coordinates
    '''
    site_index = list(df[(abs(df.a-frac[0]) < 0.001) & (abs(df.b-frac[1]) < 0.001) & (abs(df.c-frac[2]) < 0.001)].site_index)
    if len(site_index) == 0:
        print("Error! Cannot find the site index in the structure.")
        return np.nan
    elif len(site_index) < 1:
        print("Error! Find multiple site indices in the structure.")
        return np.nan
    else:
        return site_index[0]

def find_site_index_xyz(xyz, df):
    '''
    function to find site index from xyz coordinates
    '''
    site_index = list(df[(abs(df.x-xyz[0]) < 0.001) & (abs(df.y-xyz[1]) < 0.001) & (abs(df.z-xyz[2]) < 0.001)].site_index)
    #print(site_index)
    if len(site_index) == 0:
        print("Error! Cannot find the site index in the structure.")
        return np.nan
    elif len(site_index) > 1:
        print("Error! Find multiple site indices in the structure.")
        return np.nan
    else:
        return site_index[0]

def find_df_index_frac(frac, df):
    '''
    function to find index in dataframe from fractional(abc) coordinates
    '''
    df_index = list(df[(abs(df.a-frac[0]) < 0.001) & (abs(df.b-frac[1]) < 0.001) & (abs(df.c-frac[2]) < 0.001)].index)
    if len(df_index) == 0:
        print("Error! Cannot find the site index in the structure.")
        return np.nan
    elif len(df_index) > 1:
        print("Error! Find multiple site indices in the structure.")
        return np.nan
    else:
        return int(df_index[0])

def find_df_index_xyz(xyz, df):
    '''
    function to find index in dataframe from xyz coordinates
    '''
    df_index = list(df[(abs(df.x-xyz[0]) < 0.001) & (abs(df.y-xyz[1]) < 0.001) & (abs(df.z-xyz[2]) < 0.001)].index)
    if len(df_index) == 0:
        print("Error! Cannot find the site index in the structure.")
        return np.nan
    elif len(df_index) > 1:
        print("Error! Find multiple site indices in the structure.")
        return np.nan
    else:
        return int(df_index[0])

def find_df_index_from_site_index(site_index, df):
    '''
    function to find the first index in dataframe from site index
    '''
    return df[df['site_index'] == site_index].index[0]

def find_max_uvw_from_cluster_frac(axes_abc, cluster):
    urange = []
    vrange = []
    wrange = []
    for site in cluster:
        u, v, w = frac_to_uvw(axes_abc, site)
        urange.append(u)
        vrange.append(v)
        wrange.append(w)
    return [int(np.max(urange)//1)+1, int(np.max(vrange)//1)+1, int(np.max(wrange)//1)+1]

def add_site_index(df):
    '''
    function to add a column of site indices for a dataframe of sites
    should only be applied to lattice['sites'] and structure['sites']
    '''
    df['site_index'] = df.groupby('atom').cumcount()
    df['new_site_index'] = df.atom + '-' + df.site_index.astype('str')
    df['site_index'] = df['new_site_index']
    df.drop(['new_site_index'], axis = 1, inplace = True)

    return

def add_uvw(df, axes_abc):
    '''
    function to add columns of x, y and z for a dataframe with fractional(abc) coordinates
    '''
    df['u'] = df.apply(lambda row: frac_to_uvw(axes_abc, [row.a, row.b, row.c])[0], axis=1)
    df['v'] = df.apply(lambda row: frac_to_uvw(axes_abc, [row.a, row.b, row.c])[1], axis=1)
    df['w'] = df.apply(lambda row: frac_to_uvw(axes_abc, [row.a, row.b, row.c])[2], axis=1)
    return

def add_xyz(df, axes_xyz):
    '''
    function to add columns of x, y and z for a dataframe with fractional(abc) coordinates
    '''
    df['x'] = df.apply(lambda row: frac_to_xyz(axes_xyz, [row.a, row.b, row.c])[0], axis=1)
    df['y'] = df.apply(lambda row: frac_to_xyz(axes_xyz, [row.a, row.b, row.c])[1], axis=1)
    df['z'] = df.apply(lambda row: frac_to_xyz(axes_xyz, [row.a, row.b, row.c])[2], axis=1)
    return

def translate_cluster_to_cell_frac(cluster, axes_abc):
    '''
    translate a cluster in space
    make the min(u), min(v) and min(w) in the cluster within the lattice unit cell
    '''
    if not cluster:
        return [], [], [], []
    utemp = []
    vtemp = []
    wtemp = []
    for site in cluster:
        u, v, w = frac_to_uvw(axes_abc, site)
        utemp.append(u)
        vtemp.append(v)
        wtemp.append(w)
    umin = int(np.min(utemp)//1)
    vmin = int(np.min(vtemp)//1)
    wmin = int(np.min(wtemp)//1)
    utemp = [u-umin for u in utemp]
    vtemp = [v-vmin for v in vtemp]
    wtemp = [w-wmin for w in wtemp]

    #print(utemp, vtemp, wtemp)
    cluster_new = []
    for i in range(len(cluster)):
        site = uvw_to_frac(axes_abc, [utemp[i],vtemp[i],wtemp[i]])
        cluster_new.append(site)
    return utemp, vtemp, wtemp, cluster_new

def extend_sites(orig_sites, orig_vec, ext_ranges):
    '''
    function to extend sites from the sites in the original structure
    orig_sites: dataframe includes original sites
    orig_vec: the lattice vector for the original structure (u, v, w)
    ext_ranges: include the range to extend on each dimension
    '''
    ext_sites = pd.DataFrame(columns=orig_sites.columns)
    prev_sites = deepcopy(orig_sites)
    #print(orig_sites)
    for i in range(3):
        #print(ext_ranges[i])
        for n in ext_ranges[i]:
            #print(n)
            #print(orig_vec[i])
            temp = deepcopy(prev_sites)
            temp.a = temp.a+n*orig_vec[i][0]
            temp.b = temp.b+n*orig_vec[i][1]
            temp.c = temp.c+n*orig_vec[i][2]
            #print(temp)
            ext_sites = ext_sites.append(temp)
        prev_sites = deepcopy(ext_sites)
        #print(len(prev_sites.index))
    ext_sites.drop_duplicates(inplace=True)
    ext_sites.reset_index(drop=True, inplace=True)

    return ext_sites

def visualize_str_no_rep(sites_df, str_vec, xyz_file_path, png_file_path='', uvwmax=[100,100,100]):
    '''
    function to create xyz-file and pnd-file to visualize a specific cluster
    the structure configuration is not repeated in the space
    '''
    filepath = xyz_file_path
    if os.path.isfile(filepath):
        os.remove(filepath)
    with open(filepath, 'a') as file:
        df = sites_df[(sites_df.u < uvwmax[0]) & (sites_df.v < uvwmax[1]) & (sites_df.w < uvwmax[2])]
        file.write('{}\n'.format(len(df)))
        file.write('\n')
        for index, row in df.iterrows():
            file.write('{} {} {} {}\n'.format(row.atom.split(',')[int(str_vec[index])],row.x, row.y, row.z))

    if png_file_path:
        c= read(filepath)
        write(png_file_path, c, format=None, parallel=True)
    return

def visualize_str_rep(site_index_df, sites_df, str_vec, xyz_file_path, png_file_path='',uvwmax=[100,100,100]):
    '''
    function to create xyz-file and pnd-file to visualize a specific cluster
    the structure configuration is repeated in the space
    '''
    filepath = xyz_file_path
    if os.path.isfile(filepath):
        os.remove(filepath)
    with open(filepath, 'a') as file:
        df = sites_df[(sites_df.u < uvwmax[0]) & (sites_df.v < uvwmax[1]) & (sites_df.w < uvwmax[1])]
        file.write('{}\n'.format(len(df)))
        file.write('\n')
        for index, row in df.iterrows():
            #print(row.site_index)
            df_index = find_df_index_from_site_index(row.site_index, site_index_df)
            #print(df_index)
            file.write('{} {} {} {}\n'.format(row.atom.split(',')[int(str_vec[df_index])],row.x, row.y, row.z))
    if png_file_path:
        c= read(filepath)
        write(png_file_path, c, format=None, parallel=True)
    return
