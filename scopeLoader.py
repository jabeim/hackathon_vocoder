# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:20:30 2019

@author: beimx004
"""
import numpy as np
from mat4py import loadmat
fPath = 'C:/Users/beimx004/Documents/GitHub/hackathon_vocoder/Hackathon_scope_demo/'
fName = 'ScopeData_17-May-2019_1.scope'
fN = fPath+fName
testData = loadmat(fN)
testStruct = testData['S']
