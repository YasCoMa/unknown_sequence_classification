#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:32:29 2022

@author: yasmmin
"""

import math
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class Preprocessing:
    def filter_quality(self, data):
        df=data
        
        out=[]
        for i in range(len(df)):
            sequence=df.iloc[i,3].replace("*","")
            xcount=sequence.count("X")
            length=len(sequence)
            if(length>=1250 and xcount<=(length*0.3)):
                #f.write("%s\t%i\t%s" %(df.iloc[i,0], df.iloc[i,1], sequence) )
                out.append( [df.iloc[i,0], df.iloc[i,1], sequence] )
        
        dfout=pd.DataFrame(data = out, columns = ['voc','class','sequence'])     
        
        return dfout
        
