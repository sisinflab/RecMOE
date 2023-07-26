from pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce
import glob
import os

if __name__ == '__main__':
    """
    @dataset: amazon_music, goodreads, movielens1m
    @scenario1:
        obj1 = 'nDCG'
        opt1 = 'max'
        obj2 = 'Gini'
        opt2 = 'max'
        obj3 = 'EPC'
        opt3 = 'max' #Add obj3 and opt3 to the current code - line 26 - line 
    @scenario2: 
        obj1 = 'nDCG'
        opt1 = 'max'
        obj2 = 'APLT'
        opt2 = 'max'
    """
    dataset = 'movielens1m'
    dir = os.listdir(f'data/{dataset}')
    obj1 = 'nDCG'
    opt1 = 'max'
    obj2 = 'Gini'
    opt2 = 'max'
    obj3 = 'EPC'
    opt3 = 'max'
    """
    @scenario1:
        reference_point = np.array([0, 0, 0])
    @scenario2:
        reference_point = np.array([0, 0])
    """
    reference_point = np.array([0, 0, 0])
    results = []
    hypervolumes = []
    for element in dir:
        model = pd.read_csv(f'data/{dataset}/{element}', sep='\t')
        """
        @scenario1: 
            obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2, obj3: opt3})
        @scenario2: 
            obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2})
        """
        obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2, obj3: opt3})
        print('****** OPTIMAL *****')
        print(obj.get_nondominated())
        non_dominated = obj.get_nondominated()
        """
        @scenario1: 
            non_dominated.to_csv(f'results/{dataset}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_not_dominated.tsv', sep='\t', index=False)
        @scenario2:
            non_dominated.to_csv(f'results/{dataset}/{element[4:-4]}_{obj1}_{obj2}_not_dominated.tsv', sep='\t', index=False)
        """
        non_dominated.to_csv(f'results/{dataset}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_not_dominated.tsv', sep='\t', index=False)
        print('****** DOMINATED *****')
        print(obj.get_dominated())
        obj.plot(obj.get_nondominated(), obj.get_dominated(), reference_point)
        ms = obj.maximum_spread()
        sp = obj.spacing()
        er = obj.error_ratio()
        hv = obj.hypervolumes(reference_point)
        c = non_dominated.shape[0]
        hv_c = hv / c
        print(ms, sp, er, hv, c, hv_c)
        results.append([element[4:-4], ms, sp, er, hv, c, hv_c])
    res_df = pd.DataFrame(results, columns=['model', 'MS', 'SP', 'ER', 'HV', 'C', 'HV/C'])
    print('')
    pass