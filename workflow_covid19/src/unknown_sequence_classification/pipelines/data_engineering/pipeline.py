from kedro.pipeline import node, pipeline
from .nodes import treat_filter_data

def treat_filter_data_node():
    return node(
        func=treat_filter_data ,
        inputs=["all_vocs_data"],
        outputs="treat_output",
        name="filter_data",
    )

def create_pipeline():
    return pipeline(
        [
            treat_filter_data_node()
        ]
        )