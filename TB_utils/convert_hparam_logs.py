import os
import sys
from tbparse import SummaryReader
import pandas as pd
from matplotlib import pyplot as plt

#Top NewCRFS Logs
nc_80_1 = '/home/extraspace/Logs/MDE/NewCRFs/hyperparameters/0309_1229maxdepth:80_width:352_height:704_lr:0.00004_depth_80'
nc_80_2 = '/home/extraspace/Logs/MDE/NewCRFs/hyperparameters/0309_0110maxdepth:80_width:352_height:704_lr:0.00002_depth_80'
nc_256_1 = '/home/extraspace/Logs/MDE/NewCRFs/hyperparameters/0308_0321maxdepth:256_width:352_height:704_lr:0.00004_multi_new_depth_inf'
nc_256_2 = '/home/extraspace/Logs/MDE/NewCRFs/hyperparameters/0307_0442maxdepth:256_width:352_height:704_lr:0.00002_multi_new_depth_inf'

nc_list = [nc_80_1, nc_80_2, nc_256_1, nc_256_2]

#Top PixelFormer Logs
# pf_80_1 = 
# pf_80_2 = 
# pf_256_1 = 
# pf_256_2 = 

pf_list = nc_list#[pf_80_1, pf_80_2, pf_256_1, pf_256_2]

hp_list = [nc_list, pf_list]

#reader = SummaryReader(log_dir)
#df = reader.scalars
#reader2 = SummaryReader(log_dir2)
#df2 = reader2.scalars
#print(list(df.columns), '\n',df2.to_string())

hyperparam_pth = '/home/extraspace/Logs/MDE/PixelFormer/hyperparameters'

#Unique Tags
#      step            tag       value
# 0     100  learning_rate    0.000040
# 124   100     silog_loss    2.569432
# 248  1250    val/abs_rel    0.186511
# 257  1250         val/d1    0.807054
# 266  1250         val/d2    0.971021
# 275  1250         val/d3    0.988021
# 284  1250      val/log10    0.061889
# 293  1250    val/log_rms    0.204139
# 302  1250        val/rms    3.814738
# 311  1250      val/silog   18.673086
# 320  1250     val/sq_rel    0.889977
# 329   100    var average -128.231735

#learning rate graphs
#lr_df = pd.DataFrame(data={'step': [], 'tag': [], 'value': [], 'experiment_ID': []})


for h_list in hp_list:
    for idx, path in enumerate(h_list):
        summary_path = os.path.join(path, 'summaries')
        for f in os.listdir(summary_path):
            if f.startswith('events'):
                event_path = os.path.join(summary_path, f)
                break
        reader = SummaryReader(event_path)
        current_df = reader.scalars
        #df = current_df.drop_duplicates(subset=['tag'])

        if idx == 0:
            items = ["step", "tag", "value"]
            silog_df = current_df.query(" tag == 'silog_loss' ").filter(items = items).rename(columns={'value': ('_').join(path.split('/')[-1].split('max', 1)[-1].split('_')[0:4])})
            abs_rel_df = current_df.query(" tag == 'val/abs_rel' ").filter(items = items).rename(columns={'value': path.split('/')[-1]})
            d1_df = current_df.query(" tag == 'val/d1' ").filter(items = items).rename(columns={'value': path.split('/')[-1]})
            rms_df = current_df.query(" tag == 'val/rms' ").filter(items = items).rename(columns={'value': path.split('/')[-1]})
            log_rms_df = current_df.query(" tag == 'val/log_rms' ").filter(items = items).rename(columns={'value': path.split('/')[-1]})
        else:
            items = ["value"]
            silog_df = pd.concat([silog_df, current_df.query(" tag == 'silog_loss' ").filter(items = items).rename(columns={'value':('_').join(path.split('/')[-1].split('max', 1)[-1].split('_')[0:4])})], axis=1, ignore_index=False)
            abs_rel_df = pd.concat([abs_rel_df, current_df.query(" tag == 'val/abs_rel' ").filter(items = items).rename(columns={'value':path})], axis=1, ignore_index=False)
            d1_df = pd.concat([d1_df, current_df.query(" tag == 'val/d1' ").filter(items = items).rename(columns={'value':path})], axis=1, ignore_index=False)
            rms_df = pd.concat([rms_df, current_df.query(" tag == 'val/rms' ").filter(items = items).rename(columns={'value':path})], axis=1, ignore_index=False)
            log_rms_df = pd.concat([log_rms_df, current_df.query(" tag == 'val/log_rms' ").filter(items = items).rename(columns={'value':path})], axis=1, ignore_index=False)

    abs_rel_df.plot(x = 0, y = [2, 3, 4, 5], use_index = False, kind ='bar', legend = True )
    plt.show()
    break

    #print(silog_df.to_string())

    # if idx == 0:
    #     lr_df = pd.concat([lr_only, pd.DataFrame(data={'experiment_ID': [idx]*len(lr_only)})], axis=1, ignore_index=True)
    # else:
    #     lr_df = pd.concat([lr_df, pd.concat([lr_only, pd.DataFrame(data={'experiment_ID': [idx]*len(lr_only)})], axis=1, ignore_index=True)], ignore_index=True)

#print(lr_df.to_string())


