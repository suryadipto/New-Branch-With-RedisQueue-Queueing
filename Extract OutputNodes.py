#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 07:16:36 2022

@author: surya
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mygene
import math
import seaborn as sns

##########################################################################################

# G=nx.read_graphml('/Users/surya/Desktop/robust1.00/UNIFORM result/ms.graphml')

# nx.draw(G)
# plt.show()

# NODES=G.nodes

# for i in NODES:
#     print(i)

# with open('UNIFORM_DIGESTresults.txt', 'w') as f:
#     for item in NODES:
#         f.write("%s\n" % item)

##########################################################################################

G=nx.read_graphml('/Users/surya/Desktop/robust1.00/ADDITIVE, STUDY_ATTENTION, lambda0.75 result/ms.graphml')

nx.draw(G)
plt.show()

NODES=G.nodes

for i in NODES:
    print(i)

with open('ADDITIVE, STUDY_ATTENTION, lambda0.75_DIGESTresults.txt', 'w') as f:
    for item in NODES:
        f.write("%s\n" % item)

# ##########################################################################################

# G=nx.read_graphml('/Users/surya/Desktop/robust1.00/ADDITIVE, BAIT_USAGE lambda1.0 result/ms.graphml')

# NODES=G.nodes

# for i in NODES:
#     print(i)

# with open('ADDITIVE, BAIT_USAGE lambda1.0_DIGESTresults.txt', 'w') as f:
#     for item in NODES:
#         f.write("%s\n" % item)

##########################################################################################