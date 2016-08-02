import numpy as np
import numpy.random as npr
#import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
import time

def output_cdf_dir(drug,channel):
    temp_dir = 'output/hierarchical/drugs/{}/{}/cdfs/'.format(drug,channel)
    if not os.path.exists(temp_dir):
        sys.exit('\nThis CDF does not exist.\n')
    else:
        return temp_dir
        
def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)
    
def dose_response_model(dose,hill,IC50):
    return 100. * ( 1. - 1./(1.+(1.*dose/IC50)**hill) )

file_name = 'python_input_data.csv'
df = pd.read_csv(file_name, names=['Drug','Channel','Experiment','Concentration','Inhibition'])
drugs = df.Drug.unique()
channels = df.Channel.unique()

drug = drugs[0]
channel = channels[0]


cdf_dir = output_cdf_dir(drug,channel)
hill_cdf = np.loadtxt(cdf_dir+'{}_{}_posterior_predictive_Hill_cdf.txt'.format(drug,channel))
pic50_cdf = np.loadtxt(cdf_dir+'{}_{}_posterior_predictive_pIC50_cdf.txt'.format(drug,channel))


num_samples = 2000

uniform_hill_samples = npr.rand(num_samples)
hill_interpolated_inverse_cdf_samples = np.interp(uniform_hill_samples,hill_cdf[:,1],hill_cdf[:,0])

uniform_pic50_samples = npr.rand(num_samples)
pic50_interpolated_inverse_cdf_samples = np.interp(uniform_pic50_samples,pic50_cdf[:,1],pic50_cdf[:,0])





import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Embedding in TK")





start = time.time()
fig = Figure(figsize=(6,10))
ax = fig.add_subplot(211)
xmin = -4
xmax = 4
concs = np.logspace(xmax,xmin,101)
ax.set_yscale('log')
ax.set_xlim(0,100)
#plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel(r'{} concentration ($\mu$M)'.format(drug))

for xlabel_i in ax.get_xticklabels():
    xlabel_i.set_visible(False)

concline, = ax.plot([], [], color='red', lw=2)
top_CI_lower, = ax.plot([], [], color='blue', lw=2)
top_CI_upper, = ax.plot([], [], color='blue', lw=2)



ax.set_ylim(10**xmax,10**xmin)
colors = ['red','blue','orange','cyan','purple']
#for expt in experiment_numbers:
    #ax.scatter(experiments[expt][:,0],experiments[expt][:,1],label='Expt {}'.format(expt+1),color=colors[expt],s=100,zorder=10)
#ax.legend(loc=2)
for i in xrange(num_samples):
    ax.plot(dose_response_model(concs,hill_interpolated_inverse_cdf_samples[i],pic50_to_ic50(pic50_interpolated_inverse_cdf_samples[i])),concs,color='black',alpha=0.01)
    

ax2 = fig.add_subplot(212,sharex=ax)
ax2.set_xlabel('% {} block'.format(channel))
ax2.set_ylabel('Probability density')
ax2.set_ylim(0,0.035)

bottom_CI_lower, = ax2.plot([], [], color='blue', lw=2)
bottom_CI_upper, = ax2.plot([], [], color='blue', lw=2)
    
    
fig.tight_layout()
#fig.savefig(all_predictions_dir+'{}_{}_predictions.png'.format(drug,channel))

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)


e = Tk.Entry(master=root)
e.pack()

e.focus_set()

def callback():
    get_str = e.get()
    try:
        predict_conc = float(get_str)
    except:
        spl = get_str.split('^')
        predict_conc = float(spl[0])**float(spl[1])
    concline.set_xdata([0,100])
    concline.set_ydata([predict_conc,predict_conc])
        
    num_samples = 1000000

    uniform_hill_samples = npr.rand(num_samples)
    hill_interpolated_inverse_cdf_samples = np.interp(uniform_hill_samples,hill_cdf[:,1],hill_cdf[:,0])

    uniform_pic50_samples = npr.rand(num_samples)
    pic50_interpolated_inverse_cdf_samples = np.interp(uniform_pic50_samples,pic50_cdf[:,1],pic50_cdf[:,0])

    predict_block = dose_response_model(predict_conc,hill_interpolated_inverse_cdf_samples,pic50_to_ic50(pic50_interpolated_inverse_cdf_samples))
    predict_block.sort()
    credible_95_lower = predict_block[num_samples/40]
    credible_95_upper = predict_block[-num_samples/40]
    
    y,binEdges=np.histogram(predict_block,bins=100,normed=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

    ax2.cla()
    ax2.set_xlabel('% {} block'.format(channel))
    ax2.set_ylabel('Probability density')
    ax2.plot(bincenters,y,color='red',lw=2)
    
    top_CI_lower.set_xdata([credible_95_lower,credible_95_lower])
    top_CI_lower.set_ydata(ax.get_ylim())
    top_CI_upper.set_xdata([credible_95_upper,credible_95_upper])
    top_CI_upper.set_ydata(ax.get_ylim())

    ax2.axvline(credible_95_lower,color='blue',lw=2)
    ax2.axvline(credible_95_upper,color='blue',lw=2)
    
    fig.tight_layout()
    canvas.draw()

b = Tk.Button(master=root, text="Plot", width=10, command=callback)
b.pack()

Tk.mainloop()




#time_taken = time.time() - start
#print "\nTime taken to do {} + {}: {} s = {} min".format(drug,channel,int(time_taken),np.round(time_taken/60.,1))
#plt.show()


















