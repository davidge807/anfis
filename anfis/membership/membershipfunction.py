# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:41:58 2014

@author: tim.meggs
"""

from skfuzzy import gaussmf, gbellmf, sigmf
from inspect import signature


class MemFuncs:

    funcDict = {'gaussmf': gaussmf,
                'gbellmf': gbellmf,
                'sigmf': sigmf,
                }


    def __init__(self, MFList):
        
        # Check that MFList is not empty
        if not MFList:
            raise ValueError("MFList cannot be empty")

        # Check that each element of MFList is a non-empty list
        for i, mf in enumerate(MFList):
            if not mf:
                raise ValueError(f"MFList[{i}] cannot be empty")

        # Check that each function in each set is valid and has the correct number of parameters
        for i, mf_set in enumerate(MFList):
            for j, mf in enumerate(mf_set):
                if not isinstance(mf, tuple) and len(mf) != 2:
                    raise ValueError(f"MFList[{i}][{j}] must be a tuple with two elements: a function name and a dictionary of parameters")

                func_name, func_params = mf
                if func_name not in self.funcDict:
                    raise ValueError(f"Invalid function name in MFList[{i}][{j}]: {func_name}")

                num_params = len(signature(self.funcDict[func_name]).parameters) - 1 
                if len(func_params) != num_params:
                    raise ValueError(f"Wrong number of parameters in MFList[{i}][{j}] for function {func_name}. Expected {num_params}, got {len(func_params)}")

        # # Check that each set has the same number of functions
        # num_functions = len(MFList[0])
        # for i, mf_set in enumerate(MFList):
        #     if len(mf_set) != num_functions:
        #         print(mf_set)
        #         raise ValueError(f"Each set of functions in MFList must have the same number of functions. Set {i} has {len(mf_set)}, but expected {num_functions}")

        self.MFList = MFList


    def __str__(self):
        str_repr = "Membership Functions object\n"
        for i, mf_set in enumerate(self.MFList):
            str_repr += f"Variable {i+1}:\n"
            for j, mf in enumerate(mf_set):
                str_repr += f"\tMF{j+1}: {mf[0]}({', '.join([str(x) for x in mf[1].values()])})\n"
        return str_repr


    def evaluateMF(self, rowInput):
        if len(rowInput) != len(self.MFList):
            raise Exception("Number of variables does not match number of rule sets")

        return [[self.funcDict[self.MFList[i][k][0]](rowInput[i],**self.MFList[i][k][1]) 
                 for k in range(len(self.MFList[i]))] 
                for i in range(len(rowInput))]

