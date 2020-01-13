import pandas as pd
import numpy as np
import os
import statistics as stats

from code.outliers import get_stats

def make_fake_data():
    ''' 
    function takes makes fake data frame in the correct format
    '''
    test_df = pd.DataFrame(np.random.randint(100, size=(10, 3)), columns=['Group', 'Group2', 'count'])
    test_df['group_letter'] = np.where(test_df['Group'] > 50, "A", "B")
    test_df['colour'] = pd.qcut(test_df['Group2'], 5, labels=list(['Red', 'Blue', 'Green', 'Orange', 'Yellow']))
    test_df = test_df.groupby(['group_letter','colour']).agg({'count': 'sum'})    
    return test_df

def alternate_get_stats(test_df):
    '''
    function takes in dictionary with structure of dataframe and calculates mean and standard deviation. 
    Output as a tuple of mean, standard deviation
    '''
    test_dict = test_df.to_dict()
    
    a_list = []
    b_list = []
    for k, v in test_dict.items():
        for k2, v2 in v.items():
            if "A" in k2:
                a_list.append(v2)
            elif "B" in k2:
                b_list.append(v2)
            else:
                print("error")
    
    a_mean = stats.mean(a_list)
    a_std = round(stats.stdev(a_list), 6)
    
    return a_mean, a_std
    
def test_get_stats():
    ''' unit test that compares the output of get_stats() with alternate implementation 
    using dictionary'''
    # copy dataframe and make alternate implementation of function 
    df = make_fake_data()
    test_df = df.copy()
    
    test_res = alternate_get_stats(test_df)
    
    # Create final results dataframe from the get_stats() function 
    results_df = get_stats(df=df, measure='count', aggregators=['group_letter'])
    
    #assert statements
    assert results_df.iloc[0]['mean'] == test_res[0], "mean does not match expected value"
    assert round(results_df.iloc[0]['std'],6) == test_res[1], "standard deviation does not match expected value"
    

