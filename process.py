import pandas as pd
csvPD=pd.read_csv('/home/qiaozhi/MSTNE/graphscope/user_move.csv')
csvPD['weight:int64'] = (csvPD['weight:int64'] - csvPD['weight:int64'].min()) / (csvPD['weight:int64'].max() - csvPD['weight:int64'].min())
print(csvPD)
csvPD.to_csv('/home/qiaozhi/MSTNE/graphscope/user_move.csv')