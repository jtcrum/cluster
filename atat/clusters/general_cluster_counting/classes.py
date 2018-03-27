import pandas as pd
import numpy as np
import copy
from math import *
import os
import sys
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
from utilities import *
import time

class Lattice:

    def __init__(self, folder_path):
        '''
        function to intialize a Lattice object by reading the lat.in file.
        folder_path: the folder that includes lat.in file
        '''
        #find lat.in file
        self.folder_path = folder_path
        filepath=folder_path+'/lat.in'
        file = open(filepath, 'r')
        lat = file.readlines()
        #the first line includes a, b, c, alpha, beta, gamma
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = [float(number) for number in lat[0].split()]
        #the next three lines includes u, v and w expressed as [ua, ub, uc], [va, vb, vc] and [wa, wb, wc] separatedly
        self.u = [int(number) for number in lat[1].split()]
        self.v = [int(number) for number in lat[2].split()]
        self.w = [int(number) for number in lat[3].split()]
        #the remaining lines represent the position (fractional coordinates) and possible atom types at each site.
        self.sites = pd.DataFrame(columns=['a', 'b', 'c', 'atom'])
        for line in lat[4:]:
            site = line.split()[:3]
            atom_type = ''
            for atom in line.split()[3:]:
                atom_type += str(atom)
            site.append(atom_type)
            self.sites = self.sites.append(pd.DataFrame([site], columns=['a', 'b', 'c', 'atom']))
        #convert all the numbers in lattice sites dataframe from string to float
        self.sites = self.sites.apply(pd.to_numeric, errors = 'ignore')
        #compute the axes abc for the lattice which can be expressed as:
        #            [ua] [ub] [uc]
        #axes_abc =  [va] [vb] [vc]
        #            [wa] [wb] [wc]
        self.axes_abc = np.array((self.u, self.v, self.w))
        #compute the axes xyz for the lattice which can be expressed as:
        #             [ax] [ay] [az]
        # axes_xyz =  [bx] [by] [bz]
        #             [cx] [cy] [cz]
        avec = np.array([1,0,0]) * self.a
        bvec = np.array([cos(self.gamma/180*pi), sin(self.gamma/180*pi), 0]) * self.b
        cvec = np.cross(avec, bvec)/np.linalg.norm(np.cross(avec, bvec)) * self.c
        self.axes_xyz = np.array((avec,bvec,cvec))
        #add uvw coordinates and xyz coordiantes for all sites
        add_uvw(self.sites, self.axes_abc)
        #only keep the sites within one unit cell defined by u, v, w vectors.
        #self.sites= self.sites[~((self.sites.u >= 1) | (self.sites.v >= 1) | (self.sites.w >= 1) | (self.sites.u < 0) | (self.sites.v < 0) | (self.sites.w <0))]
        # self.sites['utemp']=self.sites['u']//1
        # self.sites['vtemp']=self.sites['v']//1
        # self.sites['wtemp']=self.sites['w']//1
        # self.sites['u'] = self.sites['u'] - self.sites['utemp']
        # self.sites['v'] = self.sites['v'] - self.sites['vtemp']
        # self.sites['w'] = self.sites['w'] - self.sites['wtemp']
        # self.sites['a'] = self.sites['a'] - self.sites['utemp']*self.u[0] - self.sites['vtemp']*self.v[0] - self.sites['wtemp']*self.w[0]
        # self.sites['b'] = self.sites['b'] - self.sites['utemp']*self.u[1] - self.sites['vtemp']*self.v[1] - self.sites['wtemp']*self.w[1]
        # self.sites['c'] = self.sites['c'] - self.sites['utemp']*self.u[2] - self.sites['vtemp']*self.v[2] - self.sites['wtemp']*self.w[2]
        # self.sites.drop(['utemp','vtemp','wtemp'], axis=1, inplace=True)
        # self.sites = self.sites.round(6)
        # self.sites.drop_duplicates(inplace=True)
        add_xyz(self.sites, self.axes_xyz)
        #add a column records whether each site can have multiple atoms or not in the dataframe
        self.sites['multi_atoms']=self.sites['atom'].str.contains(',')
        self.sites.reset_index(drop=True, inplace=True)
        #add site indices for each site based on different atom types
        add_site_index(self.sites)

    def read_clusters_out(self):
        '''
        function to add the cluster infomation from cluster.out file
        cluster.out file should be saved in the same folder with lat.in file
        '''
        cluster_out_lines = []
        #read clusters.out line by line
        filepath = self.folder_path+'/clusters.out'
        with open(filepath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=' ')
            for row in readCSV:
                if(len(row) == 0):
                    continue
                elif(len(row) == 1):
                    row[0] = row[0].replace('\t', '')
                    cluster_out_lines.append(float(row[0])) if '.' in row[0] else cluster_out_lines.append(int(float(row[0])))
                else:
                    cluster_out_lines.append([float(number) for number in row])
        #break down clusters.out by cluster types
        self.clusters = {}
        self.cluster_type_numbers = [int(0) for i in range(10)]

        i = 0
        nsite = 0
        cluster_index = 0
        urange = []
        vrange = []
        wrange = []
        while(i < len(cluster_out_lines)):
            #each cluster type starts with its multiplicity, maximum distance between two sites and number of sites
            m = cluster_out_lines[i]
            d = cluster_out_lines[i+1]
            new_nsite = cluster_out_lines[i+2]
            #if number of sites changed, restart the cluster type index from 1
            if new_nsite != nsite:
                nsite = new_nsite
                cluster_index = 1
            #else increase the cluster index by 1
            else:
                cluster_index +=1

            cluster_type = '{}-{}'.format(nsite, cluster_index)
            self.clusters[cluster_type] = {}
            self.clusters[cluster_type]['m'] = m
            self.clusters[cluster_type]['max_d'] = d

            i = i+3
            #each cluster type has an example
            cluster = []
            for j in range(nsite):
                site = cluster_out_lines[i][:3]
                cluster.append(site)
                i += 1
            #try to translate smallest u, vand w for the coordinates in the example cluster to the unit cell
            utemp, vtemp, wtemp, cluster_temp = translate_cluster_to_cell_frac(cluster, self.axes_abc)
            #record the fractional coordinates after translation as the example cluster
            self.clusters[cluster_type]['eg_frac'] = cluster_temp
            #record the maximum u, v, and w for the coordinates in the example cluster after translation
            urange.extend(utemp)
            vrange.extend(vtemp)
            wrange.extend(wtemp)
            #record the cluster index as the maxmimum number of nsite cluster types
            self.cluster_type_numbers[nsite] = cluster_index
        #prepare the dataframe of sites for visualization to includes all the points in the example clusters
        u_vis_range = list(range(int(min(urange)//1), int(max(urange)//1)+1))
        v_vis_range = list(range(int(min(vrange)//1), int(max(vrange)//1)+1))
        w_vis_range = list(range(int(min(wrange)//1), int(max(wrange)//1)+1))
        self.vis_range = [u_vis_range, v_vis_range, w_vis_range]
        self.vis_sites = extend_sites(self.sites, self.axes_abc, self.vis_range)
        add_xyz(self.vis_sites,self.axes_xyz)
        add_uvw(self.vis_sites,self.axes_abc)
        #prepare a list of all the cluster types
        self.cluster_types = []
        for nsite, ntypes in enumerate(self.cluster_type_numbers):
            if nsite==0:
                continue
            for k in range(1, ntypes+1):
                cluster_type = '{}-{}'.format(nsite, k)
                self.cluster_types.append(cluster_type)
        return

    def visualize_cluster(self, cluster_type=['no_cluster']):
        str_vec = [0 for i in range(len(self.vis_sites.index))]
        if not os.path.exists(self.folder_path+'/lattice_clusters/'):
            os.makedirs(self.folder_path+'/lattice_clusters/')
            os.makedirs(self.folder_path+'/lattice_clusters/xyzs/')
            os.makedirs(self.folder_path+'/lattice_clusters/images/')
        if cluster_type == ['no_cluster']:
            xyz_filepath = self.folder_path+'/lattice_clusters/xyzs/no_cluster.xyz'
            png_filepath = self.folder_path+'/lattice_clusters/images/no_cluster.png'
            visualize_str_no_rep(self.vis_sites, str_vec, xyz_filepath, png_filepath)
        else:
            sites = self.clusters[cluster_type]['eg_frac']
            for site in sites:
                index = find_df_index_frac(site, self.vis_sites)
                str_vec[index] = 1
            uvwmax = find_max_uvw_from_cluster_frac(self.axes_abc, sites)
            xyz_filepath = self.folder_path+'/lattice_clusters/xyzs/cluster-{}.xyz'.format(cluster_type)
            png_filepath = self.folder_path+'/lattice_clusters/images/cluster-{}.png'.format(cluster_type)
            visualize_str_no_rep(self.vis_sites, str_vec, xyz_filepath, png_filepath, uvwmax)
        return

class Structure:

    def __init__(self, lattice, folder_path):
        '''
        function to intialize a Structure object by reading the str_dim.txt file.
        '''
        self.folder_path = folder_path
        filepath = self.folder_path+'/str_dim.txt'
        file = open(filepath, 'r')
        str_req = file.readlines()

        self.u = [int(number) for number in str_req[0].split()]
        self.v = [int(number) for number in str_req[1].split()]
        self.w = [int(number) for number in str_req[2].split()]
        self.u_mult = int(np.mean(self.u)/np.mean(lattice.u))
        self.v_mult = int(np.mean(self.v)/np.mean(lattice.v))
        self.w_mult = int(np.mean(self.w)/np.mean(lattice.w))
        self.ncell = self.u_mult*self.v_mult*self.w_mult

        self.axes_abc = np.array((self.u, self.v, self.w))
        self.axes_xyz = lattice.axes_xyz

        str_ranges=[list(range(self.u_mult)),list(range(self.v_mult)),list(range(self.w_mult))]
        self.sites = extend_sites(lattice.sites, lattice.axes_abc, str_ranges)
        add_site_index(self.sites)
        add_xyz(self.sites, self.axes_xyz)
        add_uvw(self.sites, self.axes_abc)

        self.lattice_a = lattice.a
        self.lattice_b = lattice.b
        self.lattice_c = lattice.c
        self.lattice_alpha = float(lattice.alpha)
        self.lattice_beta = float(lattice.beta)
        self.lattice_gamma = float(lattice.gamma)
        self.lattice_u = lattice.u
        self.lattice_v = lattice.v
        self.lattice_w = lattice.w

        return

    def prepare_str_out(self):
        '''
        function to prepare str.out file for the structure
        '''
        filepath = self.folder_path+'/str.out'
        if os.path.isfile(filepath):
            os.remove(filepath)
        with open(filepath, 'a') as str_file:
            str_file.write('{} {} {} {} {} {}\n'.format(self.lattice_a, self.lattice_b, self.lattice_c, self.lattice_alpha, self.lattice_beta, self.lattice_gamma))
            str_file.write('{} {} {}\n'.format(float(self.u[0]),float(self.u[1]),float(self.u[2])))
            str_file.write('{} {} {}\n'.format(float(self.v[0]),float(self.v[1]),float(self.v[2])))
            str_file.write('{} {} {}\n'.format(float(self.w[0]),float(self.w[1]),float(self.w[2])))
            for index, row in self.sites.iterrows():
                str_file.write('{} {} {} {}\n'.format(row.a, row.b, row.c, row.atom.split(',')[0]))
        return

    def read_cluster_list(self):
        #read cluster list line by line
        self.clulist = []
        filepath = self.folder_path+'/cluster_list.csv'
        with open(filepath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=' ')
            for row in readCSV:
                if(len(row) == 0):
                    continue
                elif(len(row) == 1):
                    row[0] = row[0].replace('\t', '')
                    self.clulist.append(int(float(row[0])))
                else:
                    temp = []
                    for element in row:
                        temp.append(float(element))
                    self.clulist.append(temp)
        #break down the cluster list
        self.clusters_xyz = {}
        self.clusters_frac = {}
        self.cluster_type_numbers = [int(0) for i in range(10)]
        self.cluster_types = []
        i = 0
        nsite = 0
        cluster_index = 1
        urange = []
        vrange = []
        wrange = []
        while(i < len(self.clulist)):
            #read n_site
            new_nsite = self.clulist[i]
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
            self.cluster_types.append(cluster_type)
            #initialize a empty set called temp
            temp_xyz = []
            temp_frac = []
            #read multiplicity
            multiplicity = self.clulist[i]
            i += 1
            #go through all the clusters
            n_cluster = int(multiplicity *self.ncell)
            for j in range(n_cluster):
                cluster_xyz = []
                cluster_frac = []
                for k in range(nsite):
                    point = []
                    for coor in self.clulist[i]:
                        point.append(float(coor))
                    point_frac = xyz_to_frac(self.axes_xyz,point)
                    cluster_frac.append(point_frac)
                    i+=1
                utemp, vtemp, wtemp, cluster_frac_temp = translate_cluster_to_cell_frac(cluster_frac, self.axes_abc)
                urange.extend(utemp)
                vrange.extend(vtemp)
                wrange.extend(wtemp)
                for point in cluster_frac_temp:
                    cluster_xyz.append(frac_to_xyz(self.axes_xyz, point))
                temp_xyz.append(cluster_xyz)
                temp_frac.append(cluster_frac_temp)
            #pass the line with correlation
            i += 1
            #put the set of clusters in dictionary of clusters
            self.clusters_xyz[cluster_type] = temp_xyz
            self.clusters_frac[cluster_type] = temp_frac
            #count cluster types
            self.cluster_type_numbers[nsite] = cluster_index
            cluster_index += 1
        u_vis_range = list(range(int(min(urange)//1), int(max(urange)//1)+1))
        v_vis_range = list(range(int(min(vrange)//1), int(max(vrange)//1)+1))
        w_vis_range = list(range(int(min(wrange)//1), int(max(wrange)//1)+1))
        self.vis_range = [u_vis_range, v_vis_range, w_vis_range]
        #prepare dataframe for visualization
        self.vis_sites = extend_sites(self.sites,self.axes_abc,self.vis_range)
        add_xyz(self.vis_sites,self.axes_xyz)
        add_uvw(self.vis_sites,self.axes_abc)
        #create a dictionary for the site indices of each cluster for all cluster types and add pair sites for each site for each cluster type
        self.clusters_indices = {}
        self.sites['1'] = [0 for _ in range(len(self.sites))]
        for cluster_type in self.cluster_types:
            cluster_indices = []
            if cluster_type.split('-')[0] == '1':
                for cluster in self.clusters_xyz[cluster_type]:
                    site_index = find_site_index_xyz(cluster[0], self.vis_sites)
                    df_index = find_df_index_from_site_index(site_index, self.sites)
                    self.sites.loc[self.sites.index == df_index, '1'] = cluster_type.split('-')[1]
                    cluster_indices.append([site_index])
            else:
                self.sites[cluster_type] = [defaultdict(int) for _ in range(len(self.sites))]
                for cluster in self.clusters_xyz[cluster_type]:
                    site_indices = []
                    for site in cluster:
                        site_index = find_site_index_xyz(site, self.vis_sites)
                        site_indices.append(site_index)
                    for i, site_index in enumerate(site_indices):
                        df_index = find_df_index_from_site_index(site_index, self.sites)
                        sites_res = deepcopy(site_indices[:i])
                        sites_res.extend(deepcopy(site_indices[i+1:]))
                        self.sites.iloc[df_index][cluster_type][frozenset(sites_res)] += 1
                    cluster_indices.append(site_indices)
            self.clusters_indices[cluster_type] = cluster_indices
        return

    def visualize_one_cluster_type_one_example(self, cluster_type, example_num, rep=''):
        if cluster_type.split('-')[0] == '0':
            return
        if cluster_type in self.cluster_types:
            if example_num > len(self.clusters_xyz[cluster_type]):
                print('Error! Example number out of range.')
                return
            str_vec_no_rep = [0 for i in range(len(self.vis_sites))]
            str_vec_rep = [0 for i in range(len(self.sites))]

            sites_frac = self.clusters_frac[cluster_type][example_num-1]
            sites_indices = self.clusters_indices[cluster_type][example_num-1]
            for i, site_index in enumerate(sites_indices):
                sites_df_index = find_df_index_from_site_index(site_index, self.sites)
                str_vec_rep[sites_df_index] = 1

                vis_sites_df_index = find_df_index_frac(sites_frac[i],self.vis_sites)
                str_vec_no_rep[vis_sites_df_index] = 1
            uvwmax = find_max_uvw_from_cluster_frac(self.axes_abc, sites_frac)
            if rep == '' or rep =='n':
                if not os.path.exists(self.folder_path+'/structure_clusters_no_rep/'):
                    os.makedirs(self.folder_path+'/structure_clusters_no_rep/')
                    os.makedirs(self.folder_path+'/structure_clusters_no_rep/images/')
                    os.makedirs(self.folder_path+'/structure_clusters_no_rep/xyzs/')
                xyz_path = self.folder_path+'/structure_clusters_no_rep/xyzs/cluster-{}-{}.xyz'.format(cluster_type,example_num)
                png_path = self.folder_path+'/structure_clusters_no_rep/images/cluster-{}-{}.png'.format(cluster_type,example_num)
                visualize_str_no_rep(self.vis_sites, str_vec_no_rep, xyz_path, png_path, uvwmax)

            if rep == '' or rep =='y':
                if not os.path.exists(self.folder_path+'/structure_clusters_rep/'):
                    os.makedirs(self.folder_path+'/structure_clusters_rep/')
                    os.makedirs(self.folder_path+'/structure_clusters_rep/xyzs/')
                    os.makedirs(self.folder_path+'/structure_clusters_rep/images/')
                xyz_path = self.folder_path+'/structure_clusters_rep/xyzs/cluster-{}-{}.xyz'.format(cluster_type,example_num)
                png_path = self.folder_path+'/structure_clusters_rep/images/cluster-{}-{}.png'.format(cluster_type,example_num)
                visualize_str_rep(self.sites, self.vis_sites, str_vec_rep, xyz_path, png_path, uvwmax)
        else:
            print('Error! No cluster type meets requirements')
        return

    def visualize_one_cluster_type_all_examples(self, cluster_type, rep=''):
        if cluster_type.split('-')[0] == '0':
            return
        if cluster_type in self.cluster_types:
            for example_num in range(1,len(self.clusters_xyz[cluster_type])+1):
                self.visualize_one_cluster_type_one_example(cluster_type, example_num, rep)
        else:
            print('Error! No cluster type meets requirements')
        return

    def visualize_all_cluster_types_nsite(self, nsite, rep=''):
        if nsite == 0:
            return
        for k in range(1, self.cluster_type_numbers[nsite]+1):
            cluster_type = '{}-{}'.format(nsite,k)
            self.visualize_one_cluster_type_all_examples(cluster_type, rep)
        return

    def visualize_all_sites(self, rep=''):
        if rep=='' or rep=='y':
            if not os.path.exists(self.folder_path+'/structure_clusters_rep/'):
                os.makedirs(self.folder_path+'/structure_clusters_rep/')
                os.makedirs(self.folder_path+'/structure_clusters_rep/xyzs/')
                os.makedirs(self.folder_path+'/structure_clusters_rep/images/')
            xyz_path = self.folder_path+'/structure_clusters_rep/xyzs/all_sites.xyz'
            png_path = self.folder_path+'/structure_clusters_rep/images/all_sites.png'
            str_vec = [0 for i in range(len(self.sites.index))]
            visualize_str_rep(self.sites, self.vis_sites, str_vec, xyz_path, png_path)

        if rep=='' or rep=='n':
            if not os.path.exists(self.folder_path+'/structure_clusters_no_rep/'):
                os.makedirs(self.folder_path+'/structure_clusters_no_rep/')
                os.makedirs(self.folder_path+'/structure_clusters_no_rep/xyzs')
                os.makedirs(self.folder_path+'/structure_clusters_no_rep/images')
            xyz_path = self.folder_path+'/structure_clusters_no_rep/xyzs/all_sites.xyz'
            png_path = self.folder_path+'/structure_clusters_no_rep/images/all_sites.png'
            str_vec = [0 for i in range(len(self.vis_sites.index)) ]
            visualize_str_no_rep(self.vis_sites, str_vec, xyz_path, png_path)

        return

    def visualize_all_clusters(self, rep=''):
        '''
        function to visualize every cluster in a structure for all the cluster types
        '''
        self.visualize_all_sites(rep)

        for cluster_type in self.cluster_types:
            self.visualize_one_cluster_type_all_examples(cluster_type, rep)

        return

    def count_clusters_str_config(self, str_vec, counting_types=[], excluding_types=[]):
        '''
        function to count the clusters for types for a structure configuration
        '''
        #if counting types are not given, then count for every cluster type
        if counting_types==[]:
            counting_types=self.cluster_types
        #create a set for all the excluding clusters
        excluding_clusters = set()
        for clutype in excluding_types:
            for cluster in self.clusters_indices[clutype]:
                exist = 1
                for site in cluster:
                    df_index = find_df_index_from_site_index(site, self.sites)
                    if str_vec[df_index] == 0:
                        exist = 0
                        break
                if exist == 1:
                    for i in range(len(cluster)):
                        temp = list(itertools.combinations(cluster, i+1))
                        for c in temp:
                            excluding_clusters.update([frozenset(c)])
        #start counting
        counting_results = defaultdict(int)
        for clutype in counting_types:
            count = 0
            for cluster in self.clusters_indices[clutype]:
                if frozenset(cluster) in excluding_clusters:
                    continue
                exist = 1
                for site in cluster:
                    df_index = find_df_index_from_site_index(site, self.sites)
                    if str_vec[df_index] == 0:
                        exist = 0
                        break
                if exist == 1:
                    count+=1
            counting_results[clutype] = count
        return counting_results

    def count_clusters_one_site(self, site_df_index, str_vec, counting_types=[]):
        '''
        function to count clusters for one site based on the structure configuration
        '''
        #if counting types are not given, then count for every cluster type
        if counting_types==[]:
            counting_types=self.cluster_types
        counting_results = defaultdict(int)
        for clutype in counting_types:
            count = 0
            for cluster in self.sites.iloc[site_df_index][clutype].keys():
                exist = 1
                for site in list(cluster):
                    df_index = find_df_index_from_site_index(site, self.sites)
                    if str_vec[df_index] == 0:
                        exist = 0
                        break
                if exist == 1:
                    count+=self.sites.iloc[site_df_index][clutype][cluster]
            counting_results[clutype] = count
        return counting_results

    def count_clusters_multi_configs(self, str_vecs, counting_types=[], excluding_types=[]):
        '''
        function to count clusters for a specific composition
        '''
        counting_results = defaultdict(lambda: [])
        for str_vec in str_vecs:
            result = self.count_clusters_str_config(str_vec,counting_types=counting_types, excluding_types=excluding_types)
            for cluster_type in self.cluster_types:
                counting_results[cluster_type].append(result[cluster_type])
        return counting_results

    def cal_penalty_str_config(self, str_vec, penalty):
        '''
        function to calculate the penalty for a structure configuration
        '''
        if len(penalty.keys())==0:
            return 0
        count_results = self.count_clusters_str_config(str_vec, counting_types = list(penalty.keys()))
        total_p = 0
        for cluster_type, p in penalty.items():
            total_p += count_results[cluster_type]*p
        return total_p

    def cal_penalty_difference(self, str_vec, selected_df_index, unselected_df_index, penalty):
        '''
        function to calculate the penalty difference for the swap between one selected site and one unselected site
        '''
        if len(penalty.keys())==0:
            return 0
        total_delta_p = 0
        count_results_selected = self.count_clusters_one_site(selected_df_index, str_vec, counting_types = list(penalty.keys()))
        #create a temporary structure vector represents the configuration after swap
        str_vec_temp = deepcopy(str_vec)
        str_vec_temp[selected_df_index] = 0
        str_vec_temp[unselected_df_index] = 1
        count_results_unselected = self.count_clusters_one_site(unselected_df_index, str_vec_temp, counting_types = list(penalty.keys()))
        for cluster_type, p in penalty.items():
            total_delta_p += (count_results_unselected[cluster_type] - count_results_selected[cluster_type])*p
        return total_delta_p

    def random_config_swap(self, atom_num, penalty={}, prob={}, num_vecs=1, num_step=100, vis=0, process=0, ptfile=''):
        '''
        function to generate random structure configurations with Monte Carlo swap sites algorithm
        '''
        #remove previous configurations
        if vis:
            if not os.path.exists(self.folder_path+'/random_config_process/'):
                os.makedirs(self.folder_path+'/random_config_process/')
                os.makedirs(self.folder_path+'/random_config_process/xyzs')
                os.makedirs(self.folder_path+'/random_config_process/images')
            os.system("rm {}/random_config_process/xyzs/swap-*.xyz".format(self.folder_path))
            os.system("rm {}/random_config_process/images/swap-*.png".format(self.folder_path))
        str_vecs = []
        if ptfile:
            if not os.path.exists(self.folder_path+'/random_config_process/'):
                os.makedirs(self.folder_path+'/random_config_process/')
            os.system("rm {}/random_config_process/*.txt".format(self.folder_path))
        for i in range(num_vecs):
            step = 0
            #create lists of different groups of available sites
            available_sites = {}
            for t in prob.keys():
                available_sites[t] = list(self.sites[self.sites['1']==t].index)
            available_sites['others'] = list(self.sites[(~self.sites['1'].isin(prob.keys())) & (self.sites['multi_atoms']==True)].index)
            #randomly selected atom_num sites
            selected_sites = {}
            groups = []
            group_probs = []
            num_others = atom_num
            for t, p in prob.items():
                num = int(round(atom_num*p))
                num_others -= num
                selected_sites[t] = set(random.sample(available_sites[t], num))
                groups.append(t)
                group_probs.append(num/atom_num)
            selected_sites['others'] = set(random.sample(available_sites['others'], num_others))
            groups.append('others')
            group_probs.append(num_others/atom_num)
            unselected_sites = {}
            for t, p in prob.items():
                unselected_sites[t] = set(available_sites[t]) - set(selected_sites[t])
            unselected_sites['others'] = set(available_sites['others']) - set(selected_sites['others'])
            #create a structure vector for the randomly selected configuration
            str_vec = [1 if i in set().union(*selected_sites.values()) else 0 for i, row in self.sites.iterrows()]
            p = self.cal_penalty_str_config(str_vec, penalty)
            #start swapping
            if ptfile:
                starttime = time.time()
                pfile = '{}/random_config_process/{}_ps.txt'.format(self.folder_path,i+1)
                timefile = '{}/random_config_process/{}_time.txt'.format(self.folder_path,i+1)
            while (step < num_step):
                step += 1
                group = np.random.choice(groups, 1,p=group_probs)[0]
                s_index = random.sample(selected_sites[group], 1)[0]
                us_index = random.sample(unselected_sites[group], 1)[0]
                delta_p = self.cal_penalty_difference(str_vec, s_index, us_index, penalty)
                #if penalty decreases or doesn't change, swap the atoms on the two sites
                if delta_p <= 0:
                    selected_sites[group].remove(s_index)
                    selected_sites[group].add(us_index)
                    unselected_sites[group].remove(us_index)
                    unselected_sites[group].add(s_index)
                    str_vec[s_index] = 0
                    str_vec[us_index] = 1
                    p += delta_p
                #if penalty increases, the probability to swap follows the Boltzmann distribution
                else:
                    if random.uniform(0, 1) < exp(-delta_p):
                        selected_sites[group].remove(s_index)
                        selected_sites[group].add(us_index)
                        unselected_sites[group].remove(us_index)
                        unselected_sites[group].add(s_index)
                        str_vec[s_index] = 0
                        str_vec[us_index] = 1
                        p += delta_p
                if process:
                    if vis:
                        xyz_path = self.folder_path+'/random_config_process/xyzs/swap-{}-{}.xyz'.format(i+1, step)
                        png_path = self.folder_path+'/random_config_process/images/swap-{}-{}.png'.format(i+1, step)
                        visualize_str_rep(self.sites, self.sites, str_vec, xyz_path, png_path)
                if ptfile:
                    with open(pfile, 'a') as f:
                        f.write('{}\n'.format(p))
                    with open(timefile, 'a') as f:
                        f.write('{}\n'.format(time.time()-starttime))
            str_vecs.append(str_vec)
            if vis:
                xyz_path = self.folder_path+'/random_config_process/xyzs/swap-{}-final.xyz'.format(i+1)
                png_path = self.folder_path+'/random_config_process/images/swap-{}-final.png'.format(i+1)
                visualize_str_rep(self.sites, self.sites, str_vec, xyz_path, png_path)
        return str_vecs

    def random_config_select(self, atom_num, penalty, num_vecs=1, vis=0):
        '''
        function to generate random structure configurations by selecting site one by one and crossing out the penalty sites along the process
        '''
        #remove previous configurations
        str_vecs = []
        if vis:
            if not os.path.exists(self.folder_path+'/random_config_process/'):
                os.makedirs(self.folder_path+'/random_config_process/')
                os.makedirs(self.folder_path+'/random_config_process/xyzs')
                os.makedirs(self.folder_path+'/random_config_process/images')
        for i in range(num_vecs):
            if vis:
                os.system("rm {}/random_config_process/xyzs/select-*.xyz".format(self.folder_path))
                os.system("rm {}/random_config_process/images/select-*.png".format(self.folder_path))
            #create a list of available sites
            available_site_df_indices = set(self.sites[self.sites['multi_atoms']==True].index)
            str_vec = [0 for i in range(len(self.sites.index))]
            selected_sites = set()
            atom_count = 0
            #select one site at a time and cross out its penalty sites until reaching the required atom numbers
            while((atom_count < atom_num) and (len(available_site_df_indices) > 0)):
                #random select one site from all the available sites
                site_df_index = random.sample(available_site_df_indices, 1)[0]
                #add selected site to the list
                selected_sites.add(site_df_index)
                str_vec[site_df_index] = 1
                #find out the penalty sites for the selected site
                penalty_sites = set()
                possible_penalty_sites = set()
                for clutype in penalty.keys():
                    clusters = list(self.sites.iloc[site_df_index][clutype].keys())
                    for cluster in clusters:
                        if len(cluster) == 1:
                            for s in cluster:
                                penalty_sites.update(self.sites[self.sites['site_index'] ==s].index)
                        else:
                            for s in cluster:
                                possible_penalty_sites.update(self.sites[self.sites['site_index'] == s].index)
                #check each possible penalty site, if it has any cluster in the penalty cluster types, add it to penalty sites
                for s in possible_penalty_sites:
                    for cluster_type in penalty.keys():
                        p = False
                        for cluster in self.sites.iloc[s][cluster_type].keys():
                            exist = 1
                            for site in list(cluster):
                                df_index = find_df_index_from_site_index(site, self.sites)
                                if str_vec[df_index] == 0:
                                    exist = 0
                                    break
                            if exist == 1:
                                penalty_sites.add(s)
                                p = True
                                break
                        if p:
                            break
                #remove the selected site and the coresponding penalty sites from available sites
                available_site_df_indices.remove(site_df_index)
                available_site_df_indices=available_site_df_indices.difference(penalty_sites)
                atom_count += 1
                if vis and (i==0):
                    xyz_path = self.folder_path+'/random_config_process/xyzs/select-{}.xyz'.format(atom_count)
                    png_path = self.folder_path+'/random_config_process/images/select-{}.png'.format(atom_count)
                    visualize_str_rep(self.sites, self.sites, str_vec, xyz_path, png_path)

            if(len(selected_sites) == atom_num):
                str_vecs.append(str_vec)
            else:
                print("Failed to create a structure configuration that meets the requirements.")
                return str_vecs
        return str_vecs

    def titrate_config_one_group(self, str_vec, titration_types=[], excluding_types=[]):
        '''
        function to titrate one group of clusters for one structure configuration
        '''
        if titration_types == []:
            titration_types = self.cluster_types
        #create a set which includes all the excluding clusters
        excluding_clusters = set()
        for clutype in excluding_types:
            for cluster in self.clusters_indices[clutype]:
                exist = 1
                for site in cluster:
                    df_index = find_df_index_from_site_index(site, self.sites)
                    if str_vec[df_index] == 0:
                        exist = 0
                        break
                if exist == 1:
                    for i in range(len(cluster)):
                        temp = list(itertools.combinations(cluster, i+1))
                        for c in temp:
                            excluding_clusters.update([frozenset(c)])
        #create a set which includes all the existing non-excluding clusters
        exist_clusters = []
        exist_cluster_types = []
        exist_site_cluster_dict = defaultdict(lambda:[])
        for cluster_type in titration_types:
            for cluster in self.clusters_indices[cluster_type]:
                if frozenset(cluster) in excluding_clusters:
                    continue
                exist = 1
                for site in cluster:
                    df_index = find_df_index_from_site_index(site, self.sites)
                    if str_vec[df_index] == 0:
                        exist = 0
                        break
                if exist == 1:
                    exist_clusters.append(cluster)
                    exist_cluster_types.append(cluster_type)
                    for site in cluster:
                        exist_site_cluster_dict[site].append(len(exist_clusters)-1)
        str_vec_titration = deepcopy(str_vec)
        exist_clusters_set = set(range(len(exist_clusters)))
        result = defaultdict(lambda: 0)
        while(exist_clusters_set):
            titrated = random.sample(exist_clusters_set, 1)[0]
            result[exist_cluster_types[titrated]] += 1
            #remove selected clusters from the exist cluster set
            exist_clusters_set.remove(titrated)
            #remove other clusters that share atoms with the selected cluster
            for site in exist_clusters[titrated]:
                #make the site as used in structure vector
                df_index = find_df_index_from_site_index(site, self.sites)
                str_vec_titration[df_index] = 0
                for cluster in exist_site_cluster_dict[site]:
                    if cluster in exist_clusters_set:
                        exist_clusters_set.remove(cluster)
        return result, str_vec_titration

    def titrate_config_multi_groups(self, str_vec, titration_groups=[[]], excluding_types=[], titrate_num=1, hist=0):
        '''
        function to titrate consecutively multiple group of clusters for one structure configuration
        '''
        cluster_types=reduce(lambda x,y: x+y,titration_groups)
        titration_results = defaultdict(lambda: [])
        for i in range(titrate_num):
            result = defaultdict(lambda: 0)
            str_vec_temp = deepcopy(str_vec)
            for titration_group in titration_groups:
                result_temp, str_vec_temp = self.titrate_config_one_group(str_vec_temp,titration_types=titration_group, excluding_types=excluding_types)
                for key in result_temp.keys():
                    result[key] = result_temp[key]
            for cluster_type in cluster_types:
                titration_results[cluster_type].append(result[cluster_type])
        if hist==0:
            for cluster_type in cluster_types:
                titration_results[cluster_type]=[np.mean(titration_results[cluster_type])]
        return dict(titration_results)

    def titrate_clusters_multi_configs(self, str_vecs, titration_groups=[[]], excluding_types=[], titrate_num=1, hist=0):
        '''
        function to titrate clusters for a specific composition
        '''
        cluster_types=reduce(lambda x,y: x+y,titration_groups)
        titration_results = defaultdict(lambda: [])
        for str_vec in str_vecs:
            result = self.titrate_config_multi_groups(str_vec, titration_groups=titration_groups, excluding_types=excluding_types, titrate_num=titrate_num, hist=hist)
            for cluster_type in cluster_types:
                titration_results[cluster_type].append(result[cluster_type])
        return dict(titration_results)

    def titrate_clusters_multi_configs_titrate_hist(self, str_vecs, titration_groups=[[]], excluding_types=[], titrate_num=1):
        '''
        function to titrate clusters for a specific composition
        '''
        titration_results = defaultdict(lambda: [])
        for str_vec in str_vecs:
            result = self.titrate_config_multi_groups(str_vec, titration_groups=titration_groups, excluding_types=excluding_types, titrate_num=titrate_num)
            for cluster_type in self.cluster_types:
                titration_results[cluster_type].append(result[cluster_type])
        return titration_results
