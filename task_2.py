#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


# In[99]:


fullevent_data = pd.read_csv("./2020_Problem_D_DATA/fullevents.csv")

base_path = './results/task2/{}{}'

match_ids = []
avgs = []
nums_types = []

for match in range(1, 39):
    print(match)
    # In[100]:


    pfed = fullevent_data.loc[:, ['MatchID', 'TeamID', 'MatchPeriod', 'EventTime', 'EventType', 'EventSubType']]
    pfed = pfed[pfed['MatchID']==match][pfed['TeamID'].isin(['Huskies'])]


    # In[101]:


    pfed = pfed.reset_index()
    pfed = pfed.loc[:, ['MatchID', 'TeamID', 'MatchPeriod', 'EventTime', 'EventType', 'EventSubType']]


    # In[102]:


    last_1H = max(pfed[pfed['MatchPeriod']=='1H'].EventTime)

    fed_2H = pfed[pfed['MatchPeriod']=='2H']
    fed_2H['EventTime'] += last_1H

    pfed = pd.concat([pfed[pfed['MatchPeriod']=='1H'], fed_2H])


    # In[103]:


    pfed


    # In[104]:


    points = [0]


    # In[105]:


    points.append(min(pfed[pfed['MatchPeriod']=='2H'].index))


    # In[106]:


    sindex = list(pfed[pfed['EventType']=='Substitution'].index)

    for p in sindex:
        points.append(p)


    # In[107]:


    points.append(max(pfed.index))


    # In[108]:


    points = list(set(points))
    points = sorted(points)


    # In[109]:


    points


    # In[117]:


    time_points = []


    # In[118]:


    for i in range(len(points)-1):
        start_time = pfed.iloc[points[i]].EventTime
        end_time = pfed.iloc[points[i+1]].EventTime
        # print(start_time, end_time)
        # time_points.append(start_time)

        duration = end_time - start_time
        num_points = ceil(duration / 360)
        if num_points <= 0:
            continue
        sub_quantum = duration / num_points

        while start_time < end_time:
            time_points.append(start_time)
            start_time += sub_quantum

    time_points.append(pfed.iloc[points[-1]].EventTime)


    # In[119]:


    time_points


    # In[124]:


    set(pfed[pfed.EventTime >= time_points[0]][pfed.EventTime < time_points[1]].EventType)


    # In[126]:


    num_types = []

    for i in range(len(time_points)-1):
        start_time = time_points[i]
        end_time = time_points[i+1]
        if i == len(time_points)-2:
            end_time += 360

        sub_fed = pfed[pfed.EventTime >= start_time][pfed.EventTime < end_time]
        types_set = set(sub_fed.EventType) | set(sub_fed.EventSubType)
        num_types.append(len(types_set))


    # In[129]:


    num_type_avg = np.mean(num_types)


    # In[132]:


    points_avg = [num_type_avg] * len(num_types)


    # In[142]:


    plt.plot(num_types, color='b', linewidth=3, label='...')
    plt.plot(points_avg, color='r', linewidth=2, linestyle=':', label='average of ...')
    plt.legend(loc=2)
    plt.xlabel('Quantum')
    plt.ylabel('Number of types')
    plt.savefig(base_path.format(match, '.png'))
    plt.cla()

    match_ids.append(match)
    avgs.append(num_type_avg)
    nums_types.append(num_types)


pd.DataFrame({'MatchID':match_ids, 'Avg':avgs, 'Number of Types':nums_types}).to_csv('./results/task2/num_types.csv')

# In[ ]:




