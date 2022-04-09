#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:34:10 2022

@author: yasmmin
"""

from kedro.pipeline import node, Pipeline
from .nodes import *

def prepare_features_from_data_node():
    return node(
        func=prepare_features_from_data ,
        inputs=["treat_output", 'params:voc'],
        outputs="data_parameters",
        name="generate_dl_features", tags=['model_bigru', 'model_bilstm', 'model_conv1d', 'model_conv2d', 'model_conv3layer', 'model_lstmgru', 'model_lstmconv']
    )

def run_model_node(model):
    return node(
        func=eval('run_'+model) ,
        inputs=["data_parameters"],
        outputs="results_"+model,
        name="train_"+model, tags="model_"+model
    )

def evaluate_node(model):
    return node(
        func=evaluate ,
        inputs=["results_"+model],
        #outputs="model_metrics_"+model,
        outputs=None,
        name="evaluate_metrics_"+model, tags="model_"+model
    )

def create_pipeline_prepare():
    return Pipeline(
        [ prepare_features_from_data_node() ]
        )

def create_pipeline_bigru_evaluate():
    return Pipeline ([
        run_model_node('bigru'),
        evaluate_node('bigru')
    ])

def create_pipeline_bilstm_evaluate():
    return Pipeline ([
        run_model_node('bilstm'),
        evaluate_node('bilstm')
    ])

def create_pipeline_conv1d_evaluate():
    return Pipeline ([
        run_model_node('conv1d'),
        evaluate_node('conv1d')
    ])

def create_pipeline_conv2d_evaluate():
    return Pipeline ([
        run_model_node('conv2d'),
        evaluate_node('conv2d')
    ])

def create_pipeline_conv3layer_evaluate():
    return Pipeline ([
        run_model_node('conv3layer'),
        evaluate_node('conv3layer')
    ])

def create_pipeline_lstmgru_evaluate():
    return Pipeline ([
        run_model_node('lstmgru'),
        evaluate_node('lstmgru')
    ])

def create_pipeline_lstmconv_evaluate():
    return Pipeline ([
        run_model_node('lstmconv'),
        evaluate_node('lstmconv')
    ])

def create_pipeline():
    return create_pipeline_prepare() + create_pipeline_bigru_evaluate() + create_pipeline_bilstm_evaluate() + create_pipeline_conv1d_evaluate() + create_pipeline_conv2d_evaluate() + create_pipeline_conv3layer_evaluate() + create_pipeline_lstmgru_evaluate() + create_pipeline_lstmconv_evaluate()
    
