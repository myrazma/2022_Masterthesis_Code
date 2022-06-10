import pandas as pd

d = pd.DataFrame.from_dict({'colA': [1, 2, 3, 4], 'colB': [2, 4, 6, 8]})

d.to_csv('can_be_deleted.csv')