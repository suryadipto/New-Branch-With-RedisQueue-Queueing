import time
from Api_entrance_point import api_entrance_point
from StudyBiasData.UpdateStudyBiasData import update_study_bias_data
from flask import Flask, render_template, request, jsonify
from celery import Celery
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import SelectField
import json
import pandas as pd
from csv import reader
import io
import networkx as nx
from networkx.readwrite import json_graph
import atexit
from apscheduler.scheduler import Scheduler
import requests, zipfile, io
import pandas as pd
import numpy as np
import mygene

import redis
from rq import Queue

app=Flask(__name__)
r=redis.Redis()
q=Queue(connection=r)
# app.config['CELERY_BROKER_URL']     = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

cron = Scheduler(daemon=True)

# Explicitly kick off the background thread
cron.start()

@cron.interval_schedule(hours=720)
def job_function():
    update_study_bias_data()

# Shutdown your cron thread if the web process is stopped
atexit.register(lambda: cron.shutdown(wait=False))

# --------------------------------------------------------------


app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///robust.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['SECRET_KEY']='secret'

db=SQLAlchemy(app)

class Robust(db.Model):
    __tablename__ = 'robust'
    id = db.Column(db.Integer, primary_key=True)
    path_to_graph = db.Column(db.String)
    seeds = db.Column(db.String)
    namespace = db.Column(db.String)

    alpha = db.Column(db.Float)
    beta = db.Column(db.Float)
    n = db.Column(db.Integer, primary_key=False)
    tau = db.Column(db.Float)
    
    study_bias_score=db.Column(db.String)
    study_bias_score_data=db.Column(db.String)
    gamma = db.Column(db.Float)

    in_built_network = db.Column(db.String)
    provided_network = db.Column(db.String)
    is_graphml = db.Column(db.Boolean)

    # api_output=db.Column(db.JSON)
    nodeData_str=db.Column(db.String)
    edgeDataSrc_str=db.Column(db.String)
    edgeDataDest_str=db.Column(db.String)

    is_seed_str=db.Column(db.String)



    def __init__(self, path_to_graph, seeds, namespace, alpha, beta, n, tau, study_bias_score, study_bias_score_data, gamma, in_built_network, provided_network, is_graphml, nodeData_str, edgeDataSrc_str, edgeDataDest_str, is_seed_str):
        self.path_to_graph = path_to_graph
        self.seeds = seeds
        self.namespace = namespace
        self.alpha = alpha
        self.beta = beta
        self.n=n
        self.tau=tau
        self.study_bias_score=study_bias_score
        self.study_bias_score_data=study_bias_score_data
        self.gamma=gamma
        self.in_built_network=in_built_network
        self.provided_network=provided_network
        self.is_graphml=is_graphml
        # self.api_output=api_output
        self.nodeData_str=nodeData_str
        self.edgeDataSrc_str=edgeDataSrc_str
        self.edgeDataDest_str=edgeDataDest_str
        self.is_seed_str=is_seed_str


@app.route('/', methods=['GET'])
def index():
    title_='Home - ROBUST'
    return render_template('index.html', title=title_)

@app.route('/robust_about', methods=['GET'])
def robust_about():
    title_='About - ROBUST'
    return render_template('robust_about.html', title=title_)

@app.route('/robust_documentation', methods=['GET'])
def robust_documentation():
    title_='Documentation - ROBUST'
    return render_template('robust_documentation.html', title=title_)

@app.route('/run_robust', methods=['POST','GET'])
def run_robust():
    title_='Run ROBUST'
    return render_template('run_robust.html', title=title_)


@app.route('/results', methods=['POST'])
def results():

    network=['BioGRID', 'APID', 'HPRD', 'STRING']
    NAMESPACE=['GENE_SYMBOL', 'UNIPROT', 'ENTREZ']
    study_bias_score=['No','BAIT_USAGE', 'STUDY_ATTENTION', 'CUSTOM']
    #---------------------------------------------#
    study_bias_score_data='No'
    uploaded_network=''
    #---------------------------------------------#
    try:
        namespace=NAMESPACE[int(request.form.get("namespace"))] # dropdown list
    except:
        namespace='GENE_SYMBOL'
    try:
        alpha=float(request.form.get('alpha')) # number field
    except:
        alpha=0.25
    try:
        beta=float(request.form.get('beta')) # number field
    except:
        beta=0.9
    try:
        n=int(request.form.get('n')) # number field
    except:
        n=30
    try:
        tau=float(request.form.get('tau')) # number field
    except:
        tau=0.1
    try:
        study_bias_score=study_bias_score[int(request.form.get("study_bias_score"))] # dropdown list
    except:
        study_bias_score='No'
    
    try:
        gamma=float(request.form.get('gamma')) # number field
    except:
        gamma=1.0
    
    try:
        path_to_graph=network[int(request.form.get('inbuilt_network_selection'))] # number field
    except:
        path_to_graph='BioGRID'
    
    try:
        uploaded_network=str(request.form.get("uploaded_ppi_network_filename"))
    except:
        pass
    #---------------------------------------------#
    seeds=''
    try:
        seeds=request.form.get("textbox_seeds")
    except:
        pass

    is_graphml=False
    
    in_built_network=request.form.get("network_selection")
    ppi_network_contents_df=pd.DataFrame()
    
    if in_built_network=="Yes":
        try:
            provided_network=network[int(request.form.get("inbuilt_network_selection"))]
        except:
            provided_network='BioGRID'
    elif in_built_network=="No":
        if not uploaded_network.endswith('.graphml'):
            try:
                provided_network=request.form.get("network_contents")

                # # Correct-1: #
                data2 = list(map(lambda x: x.split(' '),provided_network.split("\r\n")))
                ppi_network_contents_df=pd.DataFrame(data2[1:], columns=data2[0])
                # # Correct-1 #

                # Correct-2: #
                # ppi_network_contents_df = pd.read_csv(io.BytesIO(bytearray(provided_network, encoding='utf-8')))
                # Correct-2 #
            except:
                pass
        elif uploaded_network.endswith('.graphml'):
            is_graphml=True
            try:
                provided_network=request.form.get("network_contents")

                # # Correct-1: #
                data2 = list(map(lambda x: x.split(' '),provided_network.split("\r\n")))
                ppi_network_contents_df=pd.DataFrame(data2[1:], columns=data2[0])
                # # Correct-1 #

                # Correct-2: #
                # ppi_network_contents_df = pd.read_csv(io.BytesIO(bytearray(provided_network, encoding='utf-8')))
                # Correct-2 #

            except:
                pass
    
    custom_studybiasdata_input_df=pd.DataFrame()
    if study_bias_score=='CUSTOM':
        try:
            study_bias_score_data=request.form.get("custom_studybiasdata_contents_textbox")
            data = list(map(lambda x: x.split(' '),study_bias_score_data.split("\r\n")))
            custom_studybiasdata_input_df=pd.DataFrame(data[1:], columns=data[0])
        except:
            pass
    else:
        study_bias_score_data=study_bias_score
    
    # ############################ Error flags: ################################
    if seeds=='' or seeds==None:
        error_statement='Seeds cannot be empty'
        return render_template('run_robust.html', error=error_statement)

    
    if in_built_network=="No":
        if not is_graphml==True:
            if ppi_network_contents_df.empty:
                # errorFlag1_customNetwork=1
                error_statement='Custom network has to be uploaded!'
                return render_template('run_robust.html', error=error_statement)
            else:
                numRows_df2=ppi_network_contents_df.shape[0]
                if numRows_df2==0:
                    error_statement='Custom network with zero rows uploaded. Please add atleast one row excluding the column headers.'
                    return render_template('run_robust.html', error=error_statement)
                else:
                    if ppi_network_contents_df.shape[1]<2:
                        error_statement='Custom network with less than two columns uploaded. Please add two columns.'
                        return render_template('run_robust.html', error=error_statement)
                    elif ppi_network_contents_df.shape[1]>2:
                        # Custom network with more than two columns uploaded. First two columns retained:
                        ppi_network_contents_df = ppi_network_contents_df.iloc[:,:2]



    
    if study_bias_score=='CUSTOM':
        if custom_studybiasdata_input_df.empty:
            error_statement='Custom study bias data has to be uploaded!'
            return render_template('run_robust.html', error=error_statement)
        else:
            numRows_df=custom_studybiasdata_input_df.shape[0]
            if numRows_df==0:
                error_statement='Custom study bias data with zero rows uploaded. Please add atleast one row excluding the column headers.'
                return render_template('run_robust.html', error=error_statement)
            else:
                if custom_studybiasdata_input_df.shape[1]<2:
                    error_statement='Custom study bias data with less than two columns uploaded. Please add two columns.'
                    return render_template('run_robust.html', error=error_statement)
                elif custom_studybiasdata_input_df.shape[1]>2:
                    # Custom study bias data with more than two columns uploaded. First two columns retained:
                    custom_studybiasdata_input_df = custom_studybiasdata_input_df.iloc[:,:2]
    

    input_dict={
    "path_to_graph":path_to_graph,
    "seeds": seeds,
    "namespace":namespace,
    "alpha":alpha,
    "beta":beta,
    "n":n,
    "tau":tau,
    "study_bias_score":study_bias_score,
    "study_bias_score_data": study_bias_score_data,
    "gamma":gamma,
    "in_built_network": in_built_network,
    "provided_network": provided_network,
    "is_graphml": is_graphml
    }

    input_json = json.dumps(input_dict)

    job=q.enqueue(computing_results_with_celery, input_dict)
    q_len=len(q)
    return f'Task {job.id} added to queue. {q_len} tasks in the queue.'

    # api_output_json, node_list, api_output_df, is_seed=computing_results_with_celery(input_dict)

    # nodeData=node_list
    # edgeDataSrc=api_output_df['EdgeList_src'].values.tolist()
    # edgeDataDest=api_output_df['EdgeList_dest'].values.tolist()

    # nodeData_str = ','.join(nodeData)
    # edgeDataSrc_str=','.join(edgeDataSrc)
    # edgeDataDest_str=','.join(edgeDataDest)
    
    # is_seed_strings = [str(x) for x in is_seed]
    # is_seed_str=','.join(is_seed_strings)



    # record = Robust(path_to_graph, seeds, namespace, alpha, beta, n, tau, study_bias_score, study_bias_score_data, gamma, in_built_network, provided_network, is_graphml, nodeData_str, edgeDataSrc_str, edgeDataDest_str, is_seed_str)
    # db.session.add(record)
    # db.session.commit()

    # title_="Results - ROBUST"
    # return render_template('results.html', title=title_, input_dict=input_dict, api_output_json=api_output_json, api_output_df=api_output_df, namespace=namespace, record=record, node_list=len(node_list), is_seed=len(is_seed), n=n)


@app.route('/saved_results/<int:id>', methods=['POST','GET'])
def retrieve(id):

    retrievedRecord = Robust.query.get(id)
    path_to_graph=retrievedRecord.path_to_graph
    seeds=retrievedRecord.seeds
    namespace=retrievedRecord.namespace
    alpha=retrievedRecord.alpha
    beta=retrievedRecord.beta
    n=retrievedRecord.n
    tau=retrievedRecord.tau
    study_bias_score=retrievedRecord.study_bias_score
    study_bias_score_data=retrievedRecord.study_bias_score_data
    gamma=retrievedRecord.gamma
    in_built_network=retrievedRecord.in_built_network
    provided_network=retrievedRecord.provided_network
    is_graphml=retrievedRecord.is_graphml
    nodeData_str=retrievedRecord.nodeData_str
    edgeDataSrc_str=retrievedRecord.edgeDataSrc_str
    edgeDataDest_str=retrievedRecord.edgeDataDest_str
    is_seed_str=retrievedRecord.is_seed_str

    input_dict={
    "path_to_graph":path_to_graph,
    "seeds":seeds,
    "namespace":namespace,
    "alpha":alpha,
    "beta":beta,
    "n":n,
    "tau":tau,
    "study_bias_score":study_bias_score,
    "study_bias_score_data": study_bias_score_data,
    "gamma":gamma,
    "in_built_network": in_built_network,
    "provided_network": provided_network,
    "is_graphml": is_graphml
    }

    _nodes=_convert_comma_separated_str_to_list(nodeData_str)
    src=_convert_comma_separated_str_to_list(edgeDataSrc_str)
    dest=_convert_comma_separated_str_to_list(edgeDataDest_str)
    _edges=zip(src, dest)
    _is_seed=_convert_comma_separated_str_to_list(is_seed_str)
    is_seed_int=_convert_strList_to_intList(_is_seed)
    node_data=_make_node_data(_nodes, is_seed_int)
    edge_data=_make_edge_data(_edges)
    outputData_dict=_make_dict(node_data, edge_data)
    OutputData_json=_convert_dict_to_json(outputData_dict)
    accessLink=_make_access_link(id)
    input_network=_check_input_network(provided_network)
    input_seeds=_split_data_to_list(seeds)

    
    title_="Saved results - ROBUST"
    return render_template('saved_results.html', retrievedRecord=retrievedRecord, input_dict=input_dict, OutputData_json=OutputData_json, namespace=namespace, accessLink=accessLink, input_network=input_network, input_seeds=input_seeds, n=n)


def _convert_strList_to_intList(str_list):
    intList=[]
    for i in str_list:
        intList.append(int(i))
    return intList

def _convert_comma_separated_str_to_list(str_data):
    list_data=str_data.split(",")
    return list_data

def _make_node_data(_nodes, is_seed_int):
    node_data=[]
    for i in range(len(_nodes)):
        if is_seed_int[i]==1:
            node_dict = {"id": _nodes[i], "group": "important"}
        else:
            node_dict = {"id": _nodes[i], "group": "gene"}
        node_data.append(node_dict)
    return node_data

def _make_edge_data(_edges):
    edge_data=[]
    for i,j in _edges:
        edge_dict = {"from": i, "to": j, "group": "default"}
        edge_data.append(edge_dict)
    return edge_data

def _make_dict(node_data, edge_data):
    outputData_dict={"nodes": node_data, "edges": edge_data}
    return outputData_dict

def _convert_dict_to_json(outputData_dict):
    OutputData_json=json.dumps(outputData_dict)
    return OutputData_json

def _make_access_link(id):
    accessLink='127.0.0.1:2008/saved_results/'+str(id)
    return accessLink

def _check_input_network(provided_network):
    if provided_network in ['BioGRID', 'APID', 'HPRD', 'STRING']:
        input_network=provided_network
    else:
        input_network='custom'
    return input_network

def _split_data_to_list(data):
    str_data=str(data)
    list_data = str_data.split()
    return list_data

# celery tasks
# @celery.task
def computing_results_with_celery(input_dict):

    api_output_json, node_list, api_output_df, is_seed=api_entrance_point(input_dict)
    # return api_output_json, node_list, api_output_df, is_seed
    nodeData=node_list
    edgeDataSrc=api_output_df['EdgeList_src'].values.tolist()
    edgeDataDest=api_output_df['EdgeList_dest'].values.tolist()

    nodeData_str = ','.join(nodeData)
    edgeDataSrc_str=','.join(edgeDataSrc)
    edgeDataDest_str=','.join(edgeDataDest)
    
    is_seed_strings = [str(x) for x in is_seed]
    is_seed_str=','.join(is_seed_strings)



    record = Robust(path_to_graph, seeds, namespace, alpha, beta, n, tau, study_bias_score, study_bias_score_data, gamma, in_built_network, provided_network, is_graphml, nodeData_str, edgeDataSrc_str, edgeDataDest_str, is_seed_str)
    db.session.add(record)
    db.session.commit()

    title_="Results - ROBUST"
    return render_template('results.html', title=title_, input_dict=input_dict, api_output_json=api_output_json, api_output_df=api_output_df, namespace=namespace, record=record, node_list=len(node_list), is_seed=len(is_seed), n=n)


if __name__=='__main__':
    db.create_all()
    app.run(debug=True, port=2008)