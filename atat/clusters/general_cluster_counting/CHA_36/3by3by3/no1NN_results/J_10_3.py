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
import time

from utilities import *
from classes import *

lattice=pickle.load(open('lattice.p','rb'))
structure=pickle.load(open('structure.p','rb'))
penalty=pickle.load(open('penalty.p', 'rb'))

str_vecs = pickle.load(open('swap_str_vecs_10_3.p','rb'))
    
result = structure.titrate_clusters_multi_configs(str_vecs, titration_groups=[['2-7', '2-11']],titrate_num=100)
final_result = dict()
final_result['2-11'] = result['2-11']
final_result['2-7'] = result['2-7']
#pickle.dump(str_vecs, open('swap_str_vecs_10_3.p','wb'))
pickle.dump(final_result, open('swap_titration_100times_10_3.p','wb'))
