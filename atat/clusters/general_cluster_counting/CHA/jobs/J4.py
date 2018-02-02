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
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from ase import Atoms
from ase.io import read
from ase.io import write
from ase.visualize import view
import itertools

from helper_functions import *

lattice = pickle.load(open('lattice.p','rb'))
structure = pickle.load(open('structure.p','rb'))
lat_clusters = pickle.load(open('lat_clusters.p','rb'))
str_clusters = pickle.load(open('str_clusters.p','rb'))
penalty = pickle.load(open('penalty.p','rb'))
Al_num = 88
cluster_type_dict = prepare_cluster_type_dict(str_clusters)

str_vecs = random_config_mult(structure, str_clusters, Al_num, penalty, 100, max_try = 1000)
titrate_results = titrate_clusters_multi_configs(str_vecs, structure, str_clusters, cluster_type_dict, counting_groups=[['2-7', '2-11']], excluding_types=['3-6'], titrate_num=1)
pickle.dump(dict(titrate_results), open('titrate_results_4.p', 'wb'))
