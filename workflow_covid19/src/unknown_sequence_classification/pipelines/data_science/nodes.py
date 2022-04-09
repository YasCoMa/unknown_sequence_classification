#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:37:48 2022

@author: yasmmin
"""

from typing import Any, Dict, Union, List

import pandas as pd

from unknown_sequence_classification.pipelines.modules.process_ml_dl import RunningModel

from kedro_mlflow.io.metrics import MlflowMetricDataSet
import mlflow

def prepare_features_from_data(data: pd.DataFrame, voc: str) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.prepare_data(data, voc)
    return out

def run_bigru(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_bigru(data)
    return out

def run_bilstm(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_bilstm(data)
    return out

def run_conv1d(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_conv1d(data)
    return out

def run_conv2d(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_conv2d(data)
    return out

def run_lstmgru(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_lstmgru(data)
    return out

def run_conv3layer(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_conv3layer(data)
    return out

def run_lstmconv(data: Dict[str, Any]) -> Dict[str, Any]:
    obj = RunningModel()
    out=obj.run_lstmconv(data)
    return out

def evaluate(results: Dict[str, Any]) -> None:
    acc_ds = MlflowMetricDataSet(key="accuracy", save_args={"mode": "append"})
    prec_ds = MlflowMetricDataSet(key="precision", save_args={"mode": "append"})
    with mlflow.start_run(run_name="run_"+results['method'], nested=True):
        acc_ds.save( results['accuracy'] )
        prec_ds.save( results['precision'] )
    
    #metrics={
    #    'accuracy': { 'value': results['accuracy'], 'step': 1 },
    #    'precision': { 'value': results['precision'], 'step': 1] }
    #    }
    #return metrics
