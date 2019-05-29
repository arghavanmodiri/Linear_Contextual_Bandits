from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


TODAY = date.today()


ax = plt.figure().add_subplot(111)
ind = np.arange(4) 
width = 0.2 
x = ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
'''x0_d0_count = [  91.182/250,  108.142/250 , 113.794/250 , 117.376/250]
x0_d1_count = [ 34.186/250,  17.024/250 , 11.218/250 ,  7.362/250]
x1_d0_count = [  90.736/250 , 107.794/250 , 113.812/250 , 117.944/250]
x1_d1_count = [ 33.896/250 , 17.04/250 ,  11.176/250 ,  7.318/250]'''
x0_d0_count = [  0.253112,  0.140736,  0.090128,  0.06304]
x0_d1_count = [ 0.647384,  0.759264 , 0.81012  , 0.836656]
x1_d0_count = [  0.028336 , 0.016224 , 0.01016 , 0.00712]
x1_d1_count = [ 0.071168 , 0.083776 , 0.089592,  0.093184]
rects1 = ax.bar(ind-2*width, x0_d0_count,width=0.2,color='r',align='center')
rects2 = ax.bar(ind-width, x0_d1_count,width=0.2,color='b',align='center')
rects3 = ax.bar(ind, x1_d0_count,width=0.2,color='g',align='center')
rects4 = ax.bar(ind+width, x1_d1_count,width=0.2,color='m',align='center')

ax.set_ylabel('Ratio of Users in each group')
ax.set_title("number of users in each quarter = 250")
ax.set_xticks(ind+width)
ax.set_xticklabels( ("1st quarter", "2nd quarter", "3rd quarter", "4th quarter") )
ax.legend(('Number of X=0,D=0', 'Number of X=0,D=1', 'Number of X=1,D=0','Number of X=1,D=1') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        print(rect.get_x()+rect.get_width()/2.)
        ax.text(rect.get_x()+rect.get_width()/2., 1.01*h, '%.2f'%h,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.show()