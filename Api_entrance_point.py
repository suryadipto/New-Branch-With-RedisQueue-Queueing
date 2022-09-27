import sys
import argparse
import json
import networkx as nx
from networkx.readwrite import json_graph
from robust_dev_branch_GitHub.robust.main import run 
import requests, zipfile, io
import pandas as pd
import numpy as np
import mygene

def api_entrance_point(input_array):
    api_output_json, node_list, api_output_df, is_seed=check_input(input_array)
    return api_output_json, node_list, api_output_df, is_seed

def check_input(input_array):
    
    # Required params:
    seeds=str(input_array["seeds"])
    seeds = seeds.split()

    network=str(input_array["path_to_graph"])
    
    # Optional params:
    namespace=input_array["namespace"]
    alpha=input_array["alpha"]
    beta=input_array["beta"]
    n=input_array["n"]
    tau=input_array["tau"]
    gamma= input_array["gamma"]
    study_bias_score=input_array["study_bias_score"]
    if study_bias_score=='No':
        study_bias_score='None'
    if study_bias_score=='CUSTOM':
        study_bias_score=input_array["study_bias_score_data"]
        study_bias_score = list(map(lambda x: x.split(' '),study_bias_score.split("\r\n")))
        study_bias_score=pd.DataFrame(study_bias_score[1:], columns=study_bias_score[0])

        study_bias_score.columns.values[0] = "gene_or_protein"
        study_bias_score.columns.values[1] = "study_bias_score"

    outfile=None
    
    in_built_network=input_array["in_built_network"]
    is_graphml=input_array["is_graphml"]

    if in_built_network=="No":
        if is_graphml==False:
            provided_network=input_array["provided_network"]
            provided_network = list(map(lambda x: x.split(' '),provided_network.split("\r\n")))
            provided_network=pd.DataFrame(provided_network[1:], columns=provided_network[0])
        elif is_graphml==True:
            provided_network=nx.parse_graphml(input_array["provided_network"])
    elif in_built_network=="Yes":
        provided_network=input_array["provided_network"]
    
    n -=1

    DF, SubGraph=run(seeds, provided_network, namespace, alpha, beta, n, tau, study_bias_score, gamma, outfile)
    

    G=SubGraph
    src=[]
    dest=[]

    for i, j in G.edges:
        src.append(i)
        dest.append(j)

    _nodes=list(G.nodes)
    _edges=list(G.edges)


    data1 = json_graph.node_link_data(G)
    data2 = json_graph.node_link_data(G, {"link": "edges", "source": "from", "target": "to"})

    NodeList=[]
    EdgeList_src=[]
    EdgeList_dest=[]
    
    node_data=[]
    is_seed=[]

    # for i in _nodes:
    #     if i in seeds:
    #         node_dict = {"id": i, "group": "important"}
    #     else:
    #         node_dict = {"id": i, "group": "gene"}
    #     node_data.append(node_dict)
    for i, data in G.nodes(data=True):
        NodeList.append(i)
        if data['isSeed']:
            node_dict = {"id": i, "group": "important"}
            is_seed.append(int(1))
        else:
            node_dict = {"id": i, "group": "gene"}
            is_seed.append(int(0))
        node_data.append(node_dict)

    
    edge_data=[]
    for i,j in _edges:
        EdgeList_src.append(i)
        EdgeList_dest.append(j)
        edge_dict = {"from": i, "to": j, "group": "default"}
        edge_data.append(edge_dict)

    outputData_dict={"nodes": node_data, "edges": edge_data}
    OutputData_json=json.dumps(outputData_dict)

    OutputData_df=pd.DataFrame()
    OutputData_df['EdgeList_src'] = pd.Series(EdgeList_src)
    OutputData_df['EdgeList_dest'] = pd.Series(EdgeList_dest)

    # ==============================================================

    return OutputData_json, NodeList, OutputData_df, is_seed