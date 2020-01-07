import pandas as pd
import numpy as np
import statistics as stats
from unittest import TestCase, main

from outliers import get_stats

class TestGetStats(TestCase):
    
    def test_get_stats(self):
        
        # Make fake df
        test_df = pd.DataFrame(np.random.randint(100, size=(10, 3)), columns=['Group', 'Group2', 'count'])
        test_df['group_letter'] = np.where(test_df['Group'] > 50, "A", "B")
        test_df['colour'] = pd.qcut(test_df['Group2'], 5, labels=list(['Red', 'Blue', 'Green', 'Orange', 'Yellow']))
        test_df = test_df.groupby(['group_letter','colour']).agg({'count': 'sum'})
        
        # Prepare unit test dataframe as dictionary
        unit_test_df = test_df.copy()
        unit_test_dict = unit_test_df.to_dict()

        # Create final results dataframe from the get_stats() function 
        results_df = get_stats(df=test_df, measure='count', aggregators=['group_letter'])

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

        self.assertEqual(results_df.iloc[0]['mean'], a_mean, "should be equal")
        self.assertEqual(round(results_df.iloc[0]['std'],6), a_std, "should be equal")

if __name__ == '__main__':
    main()