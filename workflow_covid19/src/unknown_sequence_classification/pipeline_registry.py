"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from unknown_sequence_classification.pipelines import data_engineering as de
from unknown_sequence_classification.pipelines import data_science as ds
from unknown_sequence_classification.pipelines.data_science.pipeline import create_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    
    data_engineering_pipeline = de.create_pipeline()
    ml_pipeline = ds.create_pipeline()
    bigru_pipeline = ml_pipeline.only_nodes_with_tags('model_bigru')
    bilstm_pipeline = ml_pipeline.only_nodes_with_tags('model_bilstm')
    conv1d_pipeline = ml_pipeline.only_nodes_with_tags('model_conv1d')
    conv2d_pipeline = ml_pipeline.only_nodes_with_tags('model_conv2d')
    conv3layer_pipeline = ml_pipeline.only_nodes_with_tags('model_conv3layer')
    lstmgru_pipeline = ml_pipeline.only_nodes_with_tags('model_lstmgru')
    lstmconv_pipeline = ml_pipeline.only_nodes_with_tags('model_lstmconv')
    
    return {
        "de": data_engineering_pipeline,
        "ds": ml_pipeline,
        'model_bigru': bigru_pipeline,
        'model_bilstm': bilstm_pipeline,
        'model_conv1d': conv1d_pipeline,
        'model_conv2d': conv2d_pipeline,
        'model_conv3layer': conv3layer_pipeline,
        'model_lstmgru': lstmgru_pipeline,
        'model_lstmconv': lstmconv_pipeline,
        "__default__": data_engineering_pipeline + ml_pipeline,
    }
