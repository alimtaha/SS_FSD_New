import os
import sys
from tbparse import SummaryReader
import pandas as pd

log_dir = "/home/extraspace/Logs/MDE/PixelFormer/hyperparameters/1217_1304maxdepth:80_width:1120_height:352_lr:0.00001_multi_new_depth_inf/summaries/events.out.tfevents.1671282254.wmgubws06.wmgds.wmg.warwick.ac.uk.18153.0"
log_dir2 = "/home/extraspace/Logs/MDE/PixelFormer/hyperparameters/1217_1304maxdepth:80_width:1120_height:352_lr:0.00001_multi_new_depth_inf/summaries/1671296607.5780835/events.out.tfevents.1671296607.wmgubws06.wmgds.wmg.warwick.ac.uk.18153.1"
#reader = SummaryReader(log_dir)
#df = reader.scalars
#reader2 = SummaryReader(log_dir2)
#df2 = reader2.scalars
#print(list(df.columns), '\n',df2.to_string())

hyperparam_pth = '/home/extraspace/Logs/MDE/PixelFormer/hyperparameters'

#learning rate graphs
#lr_df = pd.DataFrame(data={'step': [], 'tag': [], 'value': [], 'experiment_ID': []})
for idx, path in enumerate(os.listdir(hyperparam_pth)):
    print(idx, path)
    path = os.path.join(hyperparam_pth, path, 'summaries')
    for f in os.listdir(path):
        if f.startswith('event'):
            event_path = os.path.join(path, f)
            break
    print(event_path)
    reader = SummaryReader(event_path)
    df = reader.scalars
    lr_only = df.query(" tag == 'learning_rate' ")
    if idx == 0:
        lr_df = pd.concat([lr_only, pd.DataFrame(data={'experiment_ID': [idx]*len(lr_only)})], axis=1, ignore_index=True)
    else:
        lr_df = pd.concat([lr_df, pd.concat([lr_only, pd.DataFrame(data={'experiment_ID': [idx]*len(lr_only)})], axis=1, ignore_index=True)], ignore_index=True)

print(lr_df.to_string())


