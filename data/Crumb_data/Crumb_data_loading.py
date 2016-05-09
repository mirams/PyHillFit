
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys        #(just for version number)
#import matplotlib #(just for version number)
#print('Python version ' + sys.version)
#print('Pandas version ' + pd.__version__)
#print('Matplotlib version ' + matplotlib.__version__)


# In[ ]:

file_name = 'python_input_data.csv'
df = pd.read_csv(file_name, names=['Drug','Channel','Experiment','Concentration','Inhibition'])
df


# In[ ]:

drug_and_channel = df[['Concentration','Inhibition']][df['Drug'] == 'Amiodarone'][df['Channel'] == 'Cav1.2']
drug_and_channel
drug_and_channel.values


# In[ ]:

drugs = df.Drug.unique()
print(drugs)


# In[ ]:

channels = df.Channel.unique()
print(channels)


# In[ ]:

for drug in drugs:
    for channel in channels:
        drug_and_channel_values = df[['Concentration','Inhibition']][df['Drug'] == drug][df['Channel'] == channel]
        print(drug,channel)
        print(drug_and_channel_values)


# In[ ]:





