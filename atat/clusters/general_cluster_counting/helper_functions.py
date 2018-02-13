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

def lattice_axes_xyz(lattice):
    '''
    function to prepare the xyz axes, which is a matrix as following:
                [ax] [ay] [az]
    axes_xyz =  [bx] [by] [bz]
                [cx] [cy] [cz]
    '''
    a = np.array([1,0,0]) * lattice['a']
    b = np.array([cos(lattice['gamma']/180*pi), sin(lattice['gamma']/180*pi), 0]) * lattice['b']
    c = np.cross(a, b)/np.linalg.norm(np.cross(a, b)) * lattice['c']
    return np.array((a,b,c))

def lattice_axes_abc(lattice):
    '''
    function to prepare the abc axes, which is a matrix as following:
                [ua] [ub] [uc]
    axes_abc =  [va] [vb] [vc]
                [wa] [wb] [wc]
    '''
    return np.array((lattice['u'], lattice['v'], lattice['w']))

def structure_axes_xyz(structure):
    '''
    function to prepare the xyz axes, which is a matrix as following:
                [ax] [ay] [az]
    axes_xyz =  [bx] [by] [bz]
                [cx] [cy] [cz]
    '''
    a = np.array([1,0,0]) * structure['lattice_a']
    b = np.array([cos(structure['lattice_gamma']/180*pi), sin(structure['lattice_gamma']/180*pi), 0]) * structure['lattice_b']
    c = np.cross(a, b)/np.linalg.norm(np.cross(a, b)) * structure['lattice_c']
    return np.array((a,b,c))

def structure_axes_abc(structure):
    '''
    function to prepare the abc axes, which is a matrix as following:
                [ua] [ub] [uc]
    axes_abc =  [va] [vb] [vc]
                [wa] [wb] [wc]
    '''
    return np.array((structure['u'], structure['v'], structure['w']))

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

def find_df_index_from_site_index(df, site_index):
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

def translate_frac_into_structure(frac_coor, structure):
    '''
    translate the point into the structure by subtracting multiple structure vector on each dimension
    but may failed due to errors generated in coordinate tranformations
    '''
    nu, nv, nw = len(structure['urange']),len(structure['vrange']),len(structure['wrange'])

    fu, fv, fw = np.dot(frac_coor, np.linalg.inv(structure_axes_abc(structure)))
    #print(fu, fv,fw)

    #translate the site into the structure by subtracting multiple structure vector on each dimension
    fu -= (fu//nu)*nu
    fv -= (fv//nv)*nv
    fw -= (fw//nw)*nw
    #print(fu, fv, fw)

    return np.dot(np.array((fu, fv, fw)), axes_abc)

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

def add_xyz(df, axes_xyz):
    '''
    function to add columns of x, y and z for a dataframe with fractional(abc) coordinates
    '''
    df['x'] = df.apply(lambda row: frac_to_xyz(axes_xyz, [row.a, row.b, row.c])[0], axis=1)
    df['y'] = df.apply(lambda row: frac_to_xyz(axes_xyz, [row.a, row.b, row.c])[1], axis=1)
    df['z'] = df.apply(lambda row: frac_to_xyz(axes_xyz, [row.a, row.b, row.c])[2], axis=1)
    return

def add_uvw(df, axes_abc):
    '''
    function to add columns of x, y and z for a dataframe with fractional(abc) coordinates
    '''
    df['u'] = df.apply(lambda row: frac_to_uvw(axes_abc, [row.a, row.b, row.c])[0], axis=1)
    df['v'] = df.apply(lambda row: frac_to_uvw(axes_abc, [row.a, row.b, row.c])[1], axis=1)
    df['w'] = df.apply(lambda row: frac_to_uvw(axes_abc, [row.a, row.b, row.c])[2], axis=1)
    return

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

def visualize_str_no_rep(vis_sites_df, str_vec, xyz_file_path, png_file_path='', uvwmax=[100,100,100]):
    '''
    function to create xyz-file and pnd-file to visualize a specific cluster
    the structure configuration is not repeated in the space
    '''
    filepath = xyz_file_path
    if os.path.isfile(filepath):
        os.remove(filepath)
    with open(filepath, 'a') as file:
        df = vis_sites_df[(vis_sites_df.u < uvwmax[0]) & (vis_sites_df.v < uvwmax[1]) & (vis_sites_df.w < uvwmax[2])]
        file.write('{}\n'.format(len(df)))
        file.write('\n')
        for index, row in df.iterrows():
            file.write('{} {} {} {}\n'.format(row.atom.split(',')[int(str_vec[index])],row.x, row.y, row.z))

    if png_file_path:
        c= read(filepath)
        write(png_file_path, c, format=None, parallel=True)
    return

def visualize_str_rep(structure, str_sites_df, str_vec, xyz_file_path, png_file_path='',uvwmax=[100,100,100]):
    '''
    function to create xyz-file and pnd-file to visualize a specific cluster
    the structure configuration is repeated in the space
    '''
    filepath = xyz_file_path
    if os.path.isfile(filepath):
        os.remove(filepath)
    with open(filepath, 'a') as file:
        df = str_sites_df[(str_sites_df.u < uvwmax[0]) & (str_sites_df.v < uvwmax[1]) & (str_sites_df.w < uvwmax[1])]
        file.write('{}\n'.format(len(df)))
        file.write('\n')
        for index, row in df.iterrows():
            #print(row.site_index)
            df_index = find_df_index_from_site_index(structure['str_sites'], row.site_index)
            #print(df_index)
            file.write('{} {} {} {}\n'.format(row.atom.split(',')[int(str_vec[df_index])],row.x, row.y, row.z))
    if png_file_path:
        c= read(filepath)
        write(png_file_path, c, format=None, parallel=True)
    return

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

def read_lat_in(filepath):
    '''
    function to read lat.in file and create a dictionary to store the following lattice information:
    1)lattice paramters
    2)dataframe of sites(site coordinates, atom type)
    '''
    file = open(filepath, 'r')
    lat = file.readlines()

    lattice = {}
    lattice['a'], lattice['b'], lattice['c'], lattice['alpha'], lattice['beta'], lattice['gamma'] = [float(number) for number in lat[0].split()]
    lattice['u'] = [int(number) for number in lat[1].split()]
    lattice['v'] = [int(number) for number in lat[2].split()]
    lattice['w'] = [int(number) for number in lat[3].split()]

    lattice_sites = pd.DataFrame(columns=['a', 'b', 'c', 'atom'])
    for line in lat[4:]:
        site = line.split()[:3]
        atom_type = ''
        for atom in line.split()[3:]:
            atom_type += str(atom)
        site.append(atom_type)
        lattice_sites = lattice_sites.append(pd.DataFrame([site], columns=['a', 'b', 'c', 'atom']))
    lattice_sites = lattice_sites.apply(pd.to_numeric, errors = 'ignore')
    lattice_sites= lattice_sites[~((lattice_sites.a >= 1) | (lattice_sites.b >= 1) | (lattice_sites.c >= 1) | (lattice_sites.a < 0) | (lattice_sites.b < 0) | (lattice_sites.c <0))]
    lattice_sites['multi_atoms']=lattice_sites['atom'].str.contains(',')
    lattice_sites.reset_index(drop=True, inplace=True)
    add_site_index(lattice_sites)
    lattice['sites'] = lattice_sites

    return lattice

def read_clusters_out(filepath):
    '''
    function to read clusters.out file and create a list of non-empty lines in clusters.out
    '''
    clusters_lines = []

    with open(filepath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        for row in readCSV:
            if(len(row) == 0):
                continue
            elif(len(row) == 1):
                row[0] = row[0].replace('\t', '')
                if '.' in row[0]:
                    clusters_lines.append(float(row[0]))
                else:
                    clusters_lines.append(int(float(row[0])))
            else:
                temp = []
                for element in row:
                    temp.append(float(element))
                clusters_lines.append(temp)

    return clusters_lines

def break_down_cluster_out_lines(cluster_out_lines, lattice):
    '''
    function to read cluster_out_lines and create a dictionary for clusters in lattice
    '''
    axes_abc = lattice_axes_abc(lattice)
    lat_clusters = {}
    lat_clusters['type_number'] = [int(0) for i in range(10)]
    i = 0
    nsite = 0
    cluster_index = 0
    urange = []
    vrange = []
    wrange = []

    while(i < len(cluster_out_lines)):
        #print('i is {}'.format(i))
        m = cluster_out_lines[i]
        #print('m is {}'.format(m))
        d = cluster_out_lines[i+1]
        #print('d is {}'.format(d))
        new_nsite = cluster_out_lines[i+2]
        #print('new_site is {}'.format(new_nsite))
        if new_nsite != nsite:
            lat_clusters['num_max_site'] = new_nsite
            nsite = new_nsite
            cluster_index = 1
        else:
            cluster_index +=1

        cluster_type = '{}-{}'.format(nsite, cluster_index)
        lat_clusters[cluster_type] = {}
        lat_clusters[cluster_type]['m'] = m
        lat_clusters[cluster_type]['max_d'] = d

        i = i+3

        #print(nsite)
        cluster = []
        for j in range(nsite):
            #print(i)
            site = cluster_out_lines[i][:3]
            #print(site)
            cluster.append(site)
            i += 1
        #print(cluster)

        utemp, vtemp, wtemp, cluster_temp = translate_cluster_to_cell_frac(cluster, lattice_axes_abc(lattice))
        urange.extend(utemp)
        vrange.extend(vtemp)
        wrange.extend(wtemp)
        #print(cluster_type, utemp, vtemp, wtemp)
        lat_clusters[cluster_type]['eg_pair_frac'] = cluster_temp

        lat_clusters['type_number'][nsite] = cluster_index
    #print(urange, vrange, wrange)
    u_vis_range = list(range(int(min(urange)//1), int(max(urange)//1)+1))
    v_vis_range = list(range(int(min(vrange)//1), int(max(vrange)//1)+1))
    w_vis_range = list(range(int(min(wrange)//1), int(max(wrange)//1)+1))
    lat_clusters['vis_range'] = [u_vis_range, v_vis_range, w_vis_range]

    lat_vis_sites = extend_sites(lattice['sites'], axes_abc, lat_clusters['vis_range'])
    add_xyz(lat_vis_sites,lattice_axes_xyz(lattice))
    add_uvw(lat_vis_sites,lattice_axes_abc(lattice))
    lat_clusters['lat_vis_sites'] = lat_vis_sites

    lat_clusters['cluster_types'] = []
    for nsite, ntypes in enumerate(lat_clusters['type_number']):
        if nsite==0:
            continue
        for k in range(1, ntypes+1):
            cluster_type = '{}-{}'.format(nsite, k)
            lat_clusters['cluster_types'].append(cluster_type)

    return lat_clusters

def read_str_dim(filepath, lattice):
    '''
    function to read str_dim.txt file and create a dictionary to store the infomation for the structure_axes_abc
    '''
    file = open(filepath, 'r')
    str_req = file.readlines()

    u = [int(number) for number in str_req[0].split()]
    v = [int(number) for number in str_req[1].split()]
    w = [int(number) for number in str_req[2].split()]
    u_mult = int(np.mean(u)/np.mean(lattice['u']))
    v_mult = int(np.mean(v)/np.mean(lattice['v']))
    w_mult = int(np.mean(w)/np.mean(lattice['w']))
    ncell = u_mult*v_mult*w_mult

    str_ranges=[list(range(u_mult)),list(range(v_mult)),list(range(w_mult))]
    str_sites = extend_sites(lattice['sites'], lattice_axes_abc(lattice), str_ranges)
    add_site_index(str_sites)
    add_xyz(str_sites, lattice_axes_xyz(lattice))
    add_uvw(str_sites, structure_axes_abc(lattice))

    structure = {}
    structure['lattice_a'] = lattice['a']
    structure['lattice_b'] = lattice['b']
    structure['lattice_c'] = lattice['c']
    structure['lattice_alpha'] = lattice['alpha']
    structure['lattice_beta'] = lattice['beta']
    structure['lattice_gamma'] = lattice['gamma']
    structure['lattice_u'] = lattice['u']
    structure['lattice_v'] = lattice['v']
    structure['lattice_w'] = lattice['w']
    structure['u'] = u
    structure['v'] = v
    structure['w'] = w
    structure['u_mult'] = u_mult
    structure['v_mult'] = v_mult
    structure['w_mult'] = w_mult
    structure['ncell'] = ncell
    structure['str_sites'] = str_sites

    return structure

def prepare_str_out(structure, filepath):
    '''
    function to prepare str.out file for a specific structure
    '''
    if os.path.isfile(filepath):
        os.remove(filepath)
    with open(filepath, 'a') as str_file:
        str_file.write('{} {} {} {} {} {}\n'.format(structure['lattice_a'], structure['lattice_b'], structure['lattice_c'], float(structure['lattice_alpha']), float(structure['lattice_beta']), float(structure['lattice_gamma'])))
        str_file.write('{} {} {}\n'.format(float(structure['u'][0]),float(structure['u'][1]),float(structure['u'][2])))
        str_file.write('{} {} {}\n'.format(float(structure['v'][0]),float(structure['v'][1]),float(structure['v'][2])))
        str_file.write('{} {} {}\n'.format(float(structure['w'][0]),float(structure['w'][1]),float(structure['w'][2])))
        for index, row in structure['str_sites'].iterrows():
            str_file.write('{} {} {} {}\n'.format(row.a, row.b, row.c, row.atom.split(',')[0]))

    return

def read_cluster_list(filepath, structure, lat_clusters):
    '''
    function to add the lines of cluster_list.csv to structure dictionary
    '''

    str_clusters = {}
    clulist = []

    with open(filepath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        for row in readCSV:
            if(len(row) == 0):
                continue
            elif(len(row) == 1):
                row[0] = row[0].replace('\t', '')
                clulist.append(int(float(row[0])))
            else:
                temp = []
                for element in row:
                    temp.append(float(element))
                clulist.append(temp)

    str_clusters['cluster_orig_list'] = clulist
    str_clusters['type_number'] = lat_clusters['type_number']
    str_clusters['cluster_types'] = lat_clusters['cluster_types']

    return str_clusters

def break_down_cluster_list(str_clusters, structure):
    '''
    function to break down the cluster list
    '''
    i = 0
    nsite = 0
    cluster_index = 1
    temp = []

    urange = []
    vrange = []
    wrange = []

    while(i < len(str_clusters['cluster_orig_list'])):
        #read n_site
        new_nsite = str_clusters['cluster_orig_list'][i]
        i += 1
        #pass first three lines where the cluster contains 0 site
        if (new_nsite == 0):
            i+=2
            continue
        #if the cluster has more sites than the previous cluster, reset nsite and cluster index
        if (new_nsite != nsite):
            nsite = new_nsite
            cluster_index = 1
        #set the cluster type = [nsite]-[order of the specific cluster in all nsite-clusters].
        cluster_type = str(nsite)+'-'+str(cluster_index)

        #initialize a empty set called temp
        temp = []

        #read multiplicity
        multiplicity = str_clusters['cluster_orig_list'][i]
        i += 1

        #go through all the clusters
        n_cluster = int(multiplicity *structure['ncell'])

        for j in range(n_cluster):
            cluster_xyz = []
            cluster_frac = []
            for k in range(nsite):
                point = []
                for coor in str_clusters['cluster_orig_list'][i]:
                    point.append(float(coor))
                #print(point)
                point_frac = xyz_to_frac(structure_axes_xyz(structure),point)
                cluster_frac.append(point_frac)
                i+=1
            #print(cluster_frac)
            utemp, vtemp, wtemp, cluster_frac_temp = translate_cluster_to_cell_frac(cluster_frac, structure_axes_abc(structure))
            urange.extend(utemp)
            vrange.extend(vtemp)
            wrange.extend(wtemp)
            #print(cluster_frac_temp, utemp, vtemp, wtemp)
            for point in cluster_frac_temp:
                cluster_xyz.append(frac_to_xyz(structure_axes_xyz(structure), point))
            temp.append(cluster_xyz)
            #print(cluster_xyz)
        #pass the line with correlation
        i += 1

        #put the set of clusters in dictionary of clusters
        str_clusters[cluster_type] = temp

        #count cluster types
        cluster_index += 1
    u_vis_range = list(range(int(min(urange)//1), int(max(urange)//1)+1))
    v_vis_range = list(range(int(min(vrange)//1), int(max(vrange)//1)+1))
    w_vis_range = list(range(int(min(wrange)//1), int(max(wrange)//1)+1))
    str_clusters['mult_vis_range'] = [u_vis_range, v_vis_range, w_vis_range]

    str_clusters['str_vis_sites'] = extend_sites(structure['str_sites'],structure_axes_abc(structure),str_clusters['mult_vis_range'])
    add_xyz(str_clusters['str_vis_sites'],structure_axes_xyz(structure))
    add_uvw(str_clusters['str_vis_sites'],structure_axes_abc(structure))

    return

def visualize_structure_one_cluster_type_one_example(str_clusters, structure, folder_path, cluster_type, example_num, rep=''):
    if cluster_type.split('-')[0] == '0':
        return
    if cluster_type in str_clusters.keys():
        if example_num > len(str_clusters[cluster_type]):
            print('Error! Example number out of range.')
            return

        str_vec_no_rep = [0 for i in range(len(str_clusters['str_vis_sites']))]
        str_vec_rep = [0 for i in range(len(structure['str_sites']))]

        sites = str_clusters[cluster_type][example_num-1]
        sites_frac = []
        for site in sites:
            site_index = find_site_index_xyz(site,str_clusters['str_vis_sites'])
            #print(site_index)
            df_index = find_df_index_from_site_index(structure['str_sites'], site_index)
            #print(df_index)
            str_vec_rep[df_index] = 1

            df_index = find_df_index_xyz(site,str_clusters['str_vis_sites'])
            str_vec_no_rep[df_index] = 1
            sites_frac.append(xyz_to_frac(structure_axes_xyz(structure), site))
        #print(sites_frac)
        uvwmax = find_max_uvw_from_cluster_frac(structure_axes_abc(structure), sites_frac)
        #print(uvwmax)
        if rep == '' or rep =='n':
            xyz_path = folder_path+'/structure_clusters_no_rep/xyzs/cluster-{}-{}.xyz'.format(cluster_type,example_num, example_num)
            png_path = folder_path+'/structure_clusters_no_rep/images/cluster-{}-{}.png'.format(cluster_type,example_num, example_num)
            visualize_str_no_rep(str_clusters['str_vis_sites'], str_vec_no_rep, xyz_path, png_path, uvwmax)

        if rep == '' or rep =='y':
            xyz_path = folder_path+'/structure_clusters_rep/xyzs/cluster-{}-{}.xyz'.format(cluster_type,example_num, example_num)
            png_path = folder_path+'/structure_clusters_rep/images/cluster-{}-{}.png'.format(cluster_type,example_num, example_num)
            visualize_str_rep(structure, str_clusters['str_vis_sites'],str_vec_rep, xyz_path, png_path, uvwmax)

    else:
        print('Error! No cluster type meets requirements')
    return

def visualize_structure_one_cluster_type_all_examples(str_clusters, structure, folder_path, cluster_type, rep=''):
    if cluster_type.split('-')[0] == '0':
        return

    if cluster_type in str_clusters.keys():
        for example_num in range(1,len(str_clusters[cluster_type])+1):
            visualize_structure_one_cluster_type_one_example(str_clusters, structure, folder_path, cluster_type, example_num, rep)
    else:
        print('Error! No cluster type meets requirements')

    return

def visualize_structure_all_cluster_type_nsite(str_clusters, structure, folder_path, nsite, rep=''):
    if nsite == 0:
        return
    for k in range(1, str_clusters['type_number'][nsite]+1):
        cluster_type = '{}-{}'.format(nsite,k)
        visualize_structure_one_cluster_type_all_examples(str_clusters, structure, folder_path, cluster_type, rep)
    return

def visualize_structure_all_sites(str_clusters, structure, folder_path, rep=''):
    if rep=='' or rep=='y':
        xyz_path = folder_path+'/structure_clusters_rep/xyzs/all_sites.xyz'
        png_path = folder_path+'/structure_clusters_rep/images/all_sites.png'
        str_vec = [0 for i in range(len(structure['str_sites'].index))]
        visualize_str_rep(structure, str_clusters['str_vis_sites'], str_vec, xyz_path, png_path)

    if rep=='' or rep=='n':
        xyz_path = folder_path+'/structure_clusters_no_rep/xyzs/all_sites.xyz'
        png_path = folder_path+'/structure_clusters_no_rep/images/all_sites.png'
        str_vec = [0 for i in range(len(str_clusters['str_vis_sites'].index)) ]
        visualize_str_no_rep(str_clusters['str_vis_sites'], str_vec, xyz_path, png_path)

    return

def visualize_structure_all_clusters(str_clusters, structure, folder_path, rep=''):
    '''
    function to visualize every cluster in a structure for all the cluster types
    '''
    visualize_structure_all_sites(str_clusters, structure, folder_path, rep)

    for nsite, value in enumerate(str_clusters['type_number']):
        visualize_structure_all_cluster_type_nsite(str_clusters, structure, folder_path, nsite, rep)

    return

def add_cluster_list_site_index(str_clusters,structure):
    '''
    function to convert cluster list from xyz coordinates to site indicies
    '''
    for nsite, value in enumerate(str_clusters['type_number']):
        if nsite == 0:
            continue
        for k in range(1, str_clusters['type_number'][nsite]+1):
            cluster_type = '{}-{}'.format(nsite,k)
            temp = []
            for cluster in str_clusters[cluster_type]:
                #print(cluster)
                site_indices = []
                for site in cluster:
                    #print(site)
                    site_index = find_site_index_xyz(site, str_clusters['str_vis_sites'])
                    site_indices.append(site_index)
                temp.append(sorted(site_indices))
            str_clusters[cluster_type+'-site_index'] = temp

def add_pair_sites(structure, str_clusters):
    '''
    function to add pair sites for each site and each cluster type in the structure sites dataframe
    '''
    for nsite, ncluster in enumerate(str_clusters['type_number']):
        if nsite == 0:
            continue
        for k in range(1, ncluster+1):
            cluster_type = '{}-{}'.format(nsite,k)
            structure['str_sites'][cluster_type] = [defaultdict(int) for _ in range(len(structure['str_sites']))]
            for cluster in str_clusters[cluster_type+'-site_index']:
                for i, site_index in enumerate(cluster):
                    #print(cluster)
                    df_index = find_df_index_from_site_index(structure['str_sites'], site_index)
                    sites_res = deepcopy(cluster[:i])
                    sites_res.extend(deepcopy(cluster[i+1:]))
                    #print(sites_res)
                    structure['str_sites'].iloc[df_index][cluster_type][frozenset(sites_res)] += 1
    return

def count_clusters_str_config(str_vec, structure, str_clusters, counting_types=[], excluding_types=[]):
    '''
    function to count the clusters for types for a structure configuration
    '''
    if counting_types==[]:
        counting_types=str_clusters['cluster_types']
    excluding_clusters = set()
    for clutype in excluding_types:
        for cluster in str_clusters[clutype+'-site_index']:
            exist = 1
            for site in cluster:
                df_index = find_df_index_from_site_index(structure['str_sites'], site)
                if str_vec[df_index] == 0:
                    exist = 0
                    break
            if exist == 1:
                for i in range(len(cluster)):
                    temp = list(itertools.combinations(cluster, i+1))
                    for c in temp:
                        excluding_clusters.update([frozenset(c)])

    counting_results = defaultdict(int)
    for clutype in counting_types:
        count = 0
        for cluster in str_clusters[clutype+'-site_index']:
            #print(cluster)
            if frozenset(cluster) in excluding_clusters:
                continue
            exist = 1
            for site in cluster:
                df_index = find_df_index_from_site_index(structure['str_sites'], site)
                if str_vec[df_index] == 0:
                    exist = 0
                    break
            if exist == 1:
                count+=1
        counting_results[clutype] = count
    return counting_results

def cal_penalty_str_config(str_vec, structure, str_clusters, penalty):
    '''
    function to calculate the penalty for a structure configuration
    '''
    count_results = count_clusters_str_config(str_vec, structure, str_clusters, list(penalty.keys()))
    p = 0
    for key, value in penalty.items():
        #print(count_results[key])
        p += count_results[key]*value
    return p

def count_clusters_one_site(site_df_index, str_vec, structure, str_clusters, counting_types=[]):
    '''
    function to count clusters for one site based on the structure configuration
    '''
    if counting_types==[]:
        counting_types=str_clusters['cluster_types']
    counting_results = defaultdict(int)
    for clutype in counting_types:
        count = 0
        for cluster in structure['str_sites'].iloc[site_df_index][clutype].keys():
            exist = 1
            for site in list(cluster):
                df_index = find_df_index_from_site_index(structure['str_sites'], site)
                if str_vec[df_index] == 0:
                    exist = 0
                    break
            if exist == 1:
                count+=structure['str_sites'].iloc[site_df_index][clutype][cluster]
        counting_results[clutype] = count
    return counting_results

def cal_penalty_difference(str_vec, Al_df_index, Si_df_index, structure, str_clusters, penalty):
    '''
    function to calculate the penalty difference for one swap
    '''
    p_Al = 0
    count_results = count_clusters_one_site(Al_df_index, str_vec, structure, str_clusters, list(penalty.keys()))
    for key, value in penalty.items():
        #print(count_results[key])
        p_Al += count_results[key]*value
    p_Si = 0
    str_vec_temp = deepcopy(str_vec)
    str_vec_temp[Al_df_index] = 0
    str_vec_temp[Si_df_index] = 1
    count_results = count_clusters_one_site(Si_df_index, str_vec_temp, structure, str_clusters, list(penalty.keys()))
    for key, value in penalty.items():
        #print(count_results[key])
        p_Si += count_results[key]*value
    return p_Si - p_Al

def random_config_swap(structure, str_clusters, atom_num, penalty, vis=0, folder_path='', max_try = 100):
    '''
    function to generate random structure configuration with Monte Carlo swap sites algorithm
    '''
    if vis:
        i = 1
        while os.path.isfile(folder_path+'/random_config_process/xyzs/select-{}.xyz'.format(i)):
            os.remove(folder_path+'/random_config_process/xyzs/select-{}.xyz'.format(i))
            os.remove(folder_path+'/random_config_process/images/select-{}.png'.format(i))
            i+=1
        i = 0
        while os.path.isfile(folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(i)):
            os.remove(folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(i))
            os.remove(folder_path+'/random_config_process/images/swap-{}.png'.format(i))
            i+=1
    t = 0
    available_sites = list(structure['str_sites'][structure['str_sites']['multi_atoms']==True].index)

    selected_sites = set(random.sample(available_sites, atom_num))
    unselected_sites = set(available_sites) - set(selected_sites)

    str_vec = [1 if i in selected_sites else 0 for i, row in structure['str_sites'].iterrows()]
    #print(len(str_vec))

    if vis:
        xyz_path = folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(t)
        png_path = folder_path+'/random_config_process/images/swap-{}.png'.format(t)
        visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)

    while (t < max_try):
        #print(t)
        t += 1
        Al_temp = random.sample(selected_sites, 1)[0]
        Si_temp = random.sample(unselected_sites, 1)[0]
        #print(selected_sites)
        #print(s_temp)
        str_vec_temp = deepcopy(str_vec)
        str_vec_temp[Al_temp] = 0
        str_vec_temp[Si_temp] = 1
        delta_p = cal_penalty_difference(str_vec, Al_temp, Si_temp, structure, str_clusters, penalty)
        print(delta_p)
        if delta_p <= 0:
            #print('swap')
            #print(p_temp)
            selected_sites.remove(Al_temp)
            selected_sites.add(Si_temp)
            unselected_sites.remove(Si_temp)
            unselected_sites.add(Al_temp)
            str_vec=deepcopy(str_vec_temp)
            #print(str_vec)
            if vis:
                xyz_path = folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(t)
                png_path = folder_path+'/random_config_process/images/swap-{}.png'.format(t)
                visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)
        else:
            #print(selected_sites)
            #print(s_temp)
            if random.uniform(0, 1) < exp(-delta_p*2):
                #print('swap')
                #print(delta_p)
                selected_sites.remove(Al_temp)
                selected_sites.add(Si_temp)
                unselected_sites.remove(Si_temp)
                unselected_sites.add(Al_temp)
                str_vec=deepcopy(str_vec_temp)
                #print(str_vec)
            #else:
                #print('not swap')
            if vis:
                xyz_path = folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(t)
                png_path = folder_path+'/random_config_process/images/swap-{}.png'.format(t)
                visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)
    return str_vec

def random_config_hybrid(structure, str_clusters, atom_num, penalty, vis=0, folder_path='', max_try = 100):
    '''
    function to generate random structure random_configuration with a hybrid algorithm
    '''
    if vis:
        i = 1
        while os.path.isfile(folder_path+'/random_config_process/xyzs/select-{}.xyz'.format(i)):
            os.remove(folder_path+'/random_config_process/xyzs/select-{}.xyz'.format(i))
            os.remove(folder_path+'/random_config_process/images/select-{}.png'.format(i))
            i+=1
        i = 1
        while os.path.isfile(folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(i)):
            os.remove(folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(i))
            os.remove(folder_path+'/random_config_process/images/swap-{}.png'.format(i))
            i+=1

    available_site_df_indices = set(structure['str_sites'][structure['str_sites']['multi_atoms']==True].index)
    str_vec = [0 for i in range(len(structure['str_sites'].index))]
    selected_sites = set()
    atom_count = 0

    while((atom_count < atom_num) & (len(available_site_df_indices) > 0)):
        #random select one site from all the available sites
        site_df_index = random.sample(available_site_df_indices, 1)[0]

        #add selected site to the list
        selected_sites.add(site_df_index)
        str_vec[site_df_index] = 1

        #find out the  sites for the selected site
        #print(site_df_index)
        no_coexist_sites = set()
        temp = set()
        for t in penalty.keys():
            groups = list(structure['str_sites'].iloc[site_df_index][t].keys())
            for group in groups:
                if len(group) == 1:
                    for s in group:
                        no_coexist_sites.update(structure['str_sites'][structure['str_sites']['site_index'] ==s].index)
                else:
                    for s in group:
                        temp.update(structure['str_sites'][structure['str_sites']['site_index'] == s].index)
        for s in temp:
            count_result = count_clusters_one_site(s, str_vec, structure, str_clusters, counting_types=penalty.keys())
            for t in count_result.keys():
                if count_result[t] > 0:
                    no_coexist_sites.add(s)
                    continue
        #print(no_coexist_sites)

        #remove the selected site and the coresponding bad sites from available sites
        available_site_df_indices.remove(site_df_index)
        available_site_df_indices = available_site_df_indices.difference(no_coexist_sites)
        #print(available_sites)

        atom_count += 1

        if vis:
            xyz_path = folder_path+'/random_config_process/xyzs/select-{}.xyz'.format(atom_count)
            png_path = folder_path+'/random_config_process/images/select-{}.png'.format(atom_count)
            visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)

    if(len(selected_sites) == atom_num):
        return str_vec

    t = 0
    n_swap = 0
    available_site_df_indices = list(structure['str_sites'][structure['str_sites']['multi_atoms']==True].index)
    temp = random.sample(list(set(available_site_df_indices).difference(selected_sites)), atom_num-len(selected_sites))
    selected_sites.update(temp)
    unselected_sites = set(available_site_df_indices) - set(selected_sites)

    str_vec = [1 if i in selected_sites else 0 for i, row in structure['str_sites'].iterrows()]
    #print(len(str_vec))
    p = cal_penalty(str_vec, structure, str_clusters, penalty)
    #print(p)

    if vis:
        xyz_path = folder_path+'/random_config_process/xyzs/swap-{}.xyz'.format(n_swap)
        png_path = folder_path+'/random_config_process/images/swap-{}.png'.format(n_swap)
        visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)

    while (t < max_try):
        #print(t)
        t += 1
        s_temp = random.sample(selected_sites, 1)[0]
        u_temp = random.sample(unselected_sites, 1)[0]
        #print(selected_sites)
        #print(s_temp)
        str_vec_temp = deepcopy(str_vec)
        #print(str_vec_temp)
        str_vec_temp[s_temp] = 0
        str_vec_temp[u_temp] = 1
        p_temp = cal_penalty(str_vec_temp, structure, str_clusters, penalty)

        if p_temp < p:
            #print(n_swap)
            n_swap += 1
            t = 0
            #print(p_temp)
            selected_sites.remove(s_temp)
            selected_sites.add(u_temp)
            unselected_sites.remove(u_temp)
            unselected_sites.add(s_temp)
            p = p_temp
            str_vec=deepcopy(str_vec_temp)
            #print(str_vec)
            if vis:
                xyz_path = folder_path+'/random_config_process/xyzs/{}.xyz'.format(n_swap)
                png_path = folder_path+'/random_config_process/images/{}.png'.format(n_swap)
                visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)

        elif p_temp == p:
            #print(selected_sites)
            #print(s_temp)
            if random.uniform(0, 1) < 0.5:
                #print(n_swap)
                n_swap += 1
                t = 0
                #print(p_temp)
                selected_sites.remove(s_temp)
                selected_sites.add(u_temp)
                unselected_sites.remove(u_temp)
                unselected_sites.add(s_temp)
                p = p_temp
                str_vec=deepcopy(str_vec_temp)
                #print(str_vec)
                if vis:
                    xyz_path = folder_path+'/random_config_process/xyzs/{}.xyz'.format(n_swap)
                    png_path = folder_path+'/random_config_process/images/{}.png'.format(n_swap)
                    visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)
    return str_vec

def random_config_mult(structure, str_clusters, atom_num, penalty, config_num, vis=0, folder_path='', max_try = 100, method='hybrid'):
    '''
    function to generate multiple structure configurations for one composition
    '''
    str_vecs = []
    for i in range(config_num):
        if method =='swap':
            str_vec = random_config_swap(structure, str_clusters, atom_num, penalty, max_try=max_try)
        else:
            str_vec = random_config_hybrid(structure, str_clusters, atom_num, penalty, max_try=max_try)
        if vis:
            xyz_path = folder_path+'/random_configs/xyzs/config-{}.xyz'.format(i+1)
            png_path = folder_path+'/random_configs/images/config-{}.png'.format(i+1)
            visualize_str_rep(structure, structure['str_sites'], str_vec, xyz_path, png_path)
        str_vecs.append(str_vec)
    return str_vecs

def prepare_cluster_type_dict(str_clusters):
    '''
    function to prepare a dictionary, key = site indices for a cluster, value = cluster type
    '''
    cluster_type_dict = {}
    for nsite, ntypes in enumerate(str_clusters['type_number']):
        if nsite==0:
            continue
        for k in range(1, ntypes+1):
            cluster_type = '{}-{}'.format(nsite, k)
            for cluster in str_clusters[cluster_type+'-site_index']:
                #print(cluster)
                cluster_type_dict[frozenset(cluster)] = cluster_type
    return cluster_type_dict

def titrate_config_one_group(str_vec, structure, str_clusters, cluster_type_dict, counting_types=[], excluding_types=[]):
    '''
    function to titrate one group of clusters for one structure configuration
    '''

    if counting_types == []:
        for nsite, ntypes in enumerate(str_clusters['type_number']):
            if nsite==0:
                continue
            for k in range(1, ntypes+1):
                cluster_type = '{}-{}'.format(nsite, k)
                counting_types.append(cluster_type)

    excluding_clusters = set()

    for clutype in excluding_types:
        for cluster in str_clusters[clutype+'-site_index']:
            exist = 1
            for site in cluster:
                df_index = find_df_index_from_site_index(structure['str_sites'], site)
                if str_vec[df_index] == 0:
                    exist = 0
                    break
            if exist == 1:
                for i in range(len(cluster)):
                    temp = list(itertools.combinations(cluster, i+1))
                    for c in temp:
                        excluding_clusters.update([frozenset(c)])

    str_vec = deepcopy(str_vec)
    exist_clusters = set()
    selected_clusters = set()

    for cluster_type in counting_types:
        for cluster in str_clusters[cluster_type+'-site_index']:
            if frozenset(cluster) in excluding_clusters:
                continue
            exist = 1
            for site in cluster:
                df_index = find_df_index_from_site_index(structure['str_sites'], site)
                if str_vec[df_index] == 0:
                    exist = 0
                    break
            if exist == 1:
                exist_clusters.add(frozenset(cluster))
    #print(exist_clusters)

    while(exist_clusters):
        cluster = random.sample(exist_clusters, 1)[0]
        #print(cluster)

        #add selected clusters to list
        selected_clusters.add(cluster)
        #print(cluster)
        #remove selected clusters from the clusters of interest
        exist_clusters.remove(cluster)

        #remove other clusters that share atoms with the selected cluster
        for site in cluster:
            #print(site)
            df_index = find_df_index_from_site_index(structure['str_sites'], site)
            str_vec[df_index] = 0

            for cluster_type in counting_types:
                for pair_sites in list(structure['str_sites'].iloc[df_index][cluster_type]):
                    relevant_cluster = list(pair_sites)
                    relevant_cluster.append(site)
                    relevant_cluster = frozenset(relevant_cluster)
                    if relevant_cluster in exist_clusters:
                        exist_clusters.remove(relevant_cluster)
        #print(exist_clusters)

    result = defaultdict(lambda: 0)
    for cluster in selected_clusters:
        cluster_type = cluster_type_dict[frozenset(cluster)]
        result[cluster_type] += 1

    return result, str_vec

def titrate_config_multi_groups(str_vec, structure, str_clusters, cluster_type_dict, counting_groups=[[]], excluding_types=[], repeat_num=1):
    '''
    function to titrate multiple group of clusters for one structure configuration
    '''
    titration_results = defaultdict(lambda: [])

    for i in range(repeat_num):
        result = defaultdict(lambda: 0)
        str_vec_temp = deepcopy(str_vec)
        for counting_group in counting_groups:
            result_temp, str_vec_temp = titrate_config_one_group(str_vec_temp, structure, str_clusters, cluster_type_dict, counting_group, excluding_types)
            for key in set(result.keys()).union(set(result_temp.keys())):
                result[key] += result_temp[key]

        for cluster_type in str_clusters['cluster_types']:
            titration_results[cluster_type].append(result[cluster_type])

    return titration_results

def count_clusters_multi_configs(str_vecs, structure, str_clusters, counting_types=[], excluding_types=[]):
    '''
    function to count clusters for a specific composition
    '''
    counting_results = defaultdict(lambda: [])

    for str_vec in str_vecs:
        #print(str_vec)
        result = count_clusters_str_config(str_vec, structure, str_clusters, counting_types, excluding_types)
        #print(d1)
        for cluster_type in str_clusters['cluster_types']:
            counting_results[cluster_type].append(result[cluster_type])

    return counting_results

def titrate_clusters_multi_configs(str_vecs, structure, str_clusters, cluster_type_dict, counting_groups=[[]], excluding_types=[], titrate_num=1):
    '''
    function to titrate clusters for a specific composition
    '''
    titration_results = defaultdict(lambda: [])

    for str_vec in str_vecs:
        #print(str_vec)
        result = titrate_config_multi_groups(str_vec, structure, str_clusters, cluster_type_dict, counting_groups=counting_groups, excluding_types=excluding_types, repeat_num=titrate_num)
        for cluster_type in str_clusters['cluster_types']:
            titration_results[cluster_type].append(np.mean(result[cluster_type]))

    return titration_results

def plot_one_bar_chart(filename, counting_types, values, errors=[], ylabel='count/site', label='', color='#1B62A5'):
    '''
    function to plot one bar plot
    '''
    fig_w = 0.5*len(counting_types)
    value_series = pd.Series.from_array(values)
    x_labels = counting_types

    # now to plot the figure...
    plt.figure(figsize=(fig_w, 5))
    ax = value_series.plot(kind='bar', color = color)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(x_labels)

    if label:
        ax.legend([label])

    rects = ax.patches

    # Now make some labels
    labels = ['{:.2f}'.format(x) for x in values]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom')

    plt.savefig(filename, bbox_inches='tight')
    return

def plot_two_bar_charts(filename, counting_types, values1, values2, errors1=[], errors2=[], xlabel='', ylabel='count/site', label1='1', label2='2', color1='#1B62A5', color2='#FD690F'):
    '''
    function to plot two bar charts for comparison
    '''
    N = len(counting_types)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind-0.5*width, values1, width, yerr=errors1)
    rects2 = ax.bar(ind+0.5*width, values2, width, yerr=errors2)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind)
    ax.set_xticklabels( tuple(counting_types) )

    ax.legend( (rects1[0], rects2[0]), (label1, label2) )
    plt.savefig(filename, bbox_inches='tight')
    return
