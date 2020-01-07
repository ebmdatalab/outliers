import pandas as pd
import os
import statistics as stats

from code.outliers import get_stats

def test_get_stats(df):
    
    # Prepare unit test dataframe as dictionary
    unit_test_df = df.copy()
    unit_test_dict = unit_test_df.to_dict()
    
    # Create final results dataframe from the get_stats() function 
    results_df = get_stats(df=df, measure='count', aggregators=['group_letter'])
    
    # Create final results from unit test dictionary
    a_list = []
    b_list = []
    for k, v in unit_test_dict.items():
        for k2, v2 in v.items():
            if "A" in k2:
                a_list.append(v2)
            elif "B" in k2:
                b_list.append(v2)
            else:
                print("error")
    
    a_mean = stats.mean(a_list)
    a_std = round(stats.stdev(a_list), 6)
    
    if results_df.iloc[0]['mean'] == a_mean:
        if round(results_df.iloc[0]['std'],6) == a_std:
            return "tests passed"
        else:
            return "standard deviation test failed. Function returned {}, when it should have been {}".format(round(results_df.iloc[0]['std'],6), a_std)
    else:
        return "mean test failed. Function returned {}, when it should have been {}".format(results_df.iloc[0]['mean'], a_mean)

    

