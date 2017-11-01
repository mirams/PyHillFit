# Plot violin plots from APD90 data from Chaste output

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import matplotlib.patches as mpatches

sns.set(font_scale=1.6)

results_files = glob.glob("*.dat")
#print results_files

control_apd90 = 268.747
percent_prolongation = 10
prolonged_apd90 = (1+percent_prolongation/100.)*control_apd90

medians = []

labels = []
all_samples = []
nums_samples = []
for rf in results_files:
    drug = rf.split('_')[0]
    #if (drug=="Quinidine"):
        #continue
    labels.append(drug)
    samples = pd.read_table(rf)["APD90(ms)"].tolist()
    samples.sort()
    all_samples.append(samples)
    medians.append(np.median(samples))
    nums_samples.append(len(samples))
    
#print medians
sorted_indices = np.argsort(medians)
sorted_labels = [labels[i] for i in sorted_indices]
#print sorted_labels
sorted_samples = [all_samples[i] for i in sorted_indices]
sorted_nums_samples = [nums_samples[i] for i in sorted_indices]

#print sorted_samples


risks = [0,0,0,0,2,3,1,0,3,0,3,2,1,3,3,3,3,3,3,3,1,0,2,0,0,2,1,3,3,3]
all_risks = []

labels = []
for i,lab in enumerate(sorted_labels):
    labels += [lab]*sorted_nums_samples[i]
    all_risks += [risks[i]]*sorted_nums_samples[i]
#print labels
all_samples = np.array(sorted_samples).flatten()

#print len(labels)
#print len(all_samples)

# risk levels
# 0 - unknown
# 1 - possible
# 2 - conditional
# 3 - known

#colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
colors = ['#ffffff','#1b9e77','#7570b3','#e7298a']
risk_colors = [colors[i] for i in risks]

risk0 = mpatches.Patch(color=colors[0], label='Not in database')
risk1 = mpatches.Patch(color=colors[1], label='Possible risk')
risk2 = mpatches.Patch(color=colors[2], label='Conditional risk')
risk3 = mpatches.Patch(color=colors[3], label='Known risk')

df = pd.DataFrame(columns=["Drug","APD90 (ms)"])

print "Last 4 Quinidine:", all_samples[-4:]

df["Drug"] = labels
df["APD90 (ms)"] = all_samples
df["Risk"] = all_risks
print df

palette_dir = {0:'#7fc97f',1:'#beaed4',2:'#fdc086',3:'#ffff99'}


print df["Risk"].map(palette_dir)
        
fig = plt.figure(figsize=(12,14))
ax = fig.add_subplot(111)
control_line = ax.axhline(control_apd90,color='blue',label='Control')
prolonged_line = ax.axhline(prolonged_apd90,color='red',label='{}% prolongation'.format(percent_prolongation))
sns.violinplot(y="APD90 (ms)",x="Drug",data=df, scale="width",palette=risk_colors,ax=ax)#,hue="Risk")
ax.set_xlabel('')
ax.set_ylim(200,600)

print control_line

ax.text(20,580,"Max Quinidine sample: {}".format(all_samples[-1]),size=14)
labels = ax.get_xticklabels()
xticklocs = ax.xaxis.get_ticklocs()
plt.xticks(xticklocs+0.5, labels, rotation=60, ha='right')
ax.legend(handles=[control_line,prolonged_line,risk0,risk1,risk2,risk3], loc=2)
#ax.legend(loc=2)
ax.set_title("O'Hara 2011 endo. AP model predictions with CredibleMeds TdP risks")


#fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("violin_plots.png")
fig.savefig("violin_plots.pdf")

"""
fig2 = plt.figure()
ax2 = sns.violinplot(x="APD90 (ms)",y="Drug",data=df,scale="width",font_scale=1.5)
ax2.axvline(control_apd90,color='red')
ax2.set_ylabel('')
"""

plt.show()
