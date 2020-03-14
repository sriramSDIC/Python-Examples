#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:26:40 2020

@author: snarasim
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib 
matplotlib.style.use('ggplot')

ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()

ts_small = ts[200:600]
ts_small.plot()

