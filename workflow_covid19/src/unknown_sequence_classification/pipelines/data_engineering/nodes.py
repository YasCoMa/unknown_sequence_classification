#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:54:29 2022

@author: yasmmin
"""

from typing import Any, Dict

import pandas as pd

from unknown_sequence_classification.pipelines.modules.preprocessing import Preprocessing

def treat_filter_data(data: pd.DataFrame) -> Dict[str, Any]:
    obj = Preprocessing()
    dfout=obj.filter_quality(data)
    return dfout