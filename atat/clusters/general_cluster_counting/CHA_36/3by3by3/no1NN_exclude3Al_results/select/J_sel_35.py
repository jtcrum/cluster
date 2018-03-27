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

Al_num = int(1/(1+35)*972)

str_vec = structure.random_config_select(atom_num=Al_num, penalty=penalty, num_vecs=1)[0]

result = structure.titrate_config_multi_groups(str_vec, titration_groups=[['2-7', '2-11']], excluding_types=['3-6'], titrate_num=1)

final_result = dict()
final_result['2-7'] = result['2-7']
final_result['2-11']= result['2-11']
    
str_vecs = pickle.load(open('select_str_vecs_35.p','rb'))
str_vecs.append(str_vec)
pickle.dump(str_vecs, open('select_str_vecs_35.p','wb'))

results = pickle.load(open('select_titration_35.p','rb'))
results.append(final_result)
pickle.dump(results, open('select_titration_35.p','wb'))

