#!/usr/bin/env python
# coding: utf-8
import numpy
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

fullevent_data = pd.read_csv("./2020_Problem_D_DATA/fullevents.csv")

base_path = './results/task2/{}{}'

match_ids = []
avgs = []
avgs_op = []
nums_types = []
nums_types_op = []

for match in range(1, 39):
    print(match)

    pfed = fullevent_data.loc[:, ['MatchID', 'TeamID', 'MatchPeriod', 'EventTime', 'EventType', 'EventSubType']]
    pfed = pfed[pfed['MatchID']==match] # [pfed['TeamID'].isin(list(set(fullevent_data.TeamID) - {'Huskies'}))]

    pfed = pfed.reset_index()
    pfed = pfed.loc[:, ['MatchID', 'TeamID', 'MatchPeriod', 'EventTime', 'EventType', 'EventSubType']]

    last_1H = max(pfed[pfed['MatchPeriod']=='1H'].EventTime)

    fed_2H = pfed[pfed['MatchPeriod']=='2H']
    fed_2H['EventTime'] += last_1H

    pfed = pd.concat([pfed[pfed['MatchPeriod']=='1H'], fed_2H])

    points = [0]

    points.append(min(pfed[pfed['MatchPeriod']=='2H'].index))

    sindex = list(pfed[pfed['EventType']=='Substitution'].index)

    for p in sindex:
        points.append(p)

    points.append(max(pfed.index))

    points = list(set(points))
    points = sorted(points)

    points

    time_points = []

    for i in range(len(points)-1):
        start_time = pfed.iloc[points[i]].EventTime
        end_time = pfed.iloc[points[i+1]].EventTime
        # print(start_time, end_time)
        # time_points.append(start_time)

        duration = end_time - start_time
        num_points = ceil(duration / 120)
        if num_points <= 0:
            continue
        sub_quantum = duration / num_points

        while start_time < end_time:
            time_points.append(start_time)
            start_time += sub_quantum

    time_points.append(pfed.iloc[points[-1]].EventTime)

    num_types = []
    num_types_op = []

    for is_opponent in [False, True]:

        if is_opponent:
            pfed = fullevent_data[fullevent_data['MatchID']==match][fullevent_data['TeamID'].isin(list(set(fullevent_data.TeamID) - {'Huskies'}))]
        else:
            pfed = fullevent_data[fullevent_data['MatchID']==match][fullevent_data['TeamID'].isin(['Huskies'])]

        for i in range(len(time_points)-1):
            start_time = time_points[i]
            end_time = time_points[i+1]
            if i == len(time_points)-2:
                end_time += 120

            sub_fed = pfed[pfed.EventTime >= start_time][pfed.EventTime < end_time]
            types_set = set(sub_fed.EventType) | set(sub_fed.EventSubType)

            if is_opponent:
                num_types_op.append(len(types_set))
            else:
                num_types.append(len(types_set))

    num_types = list(filter(lambda x: x>0, num_types))
    num_types_op = list(filter(lambda x: x>0, num_types_op))

    if len(num_types) > len(num_types_op):
        for i in range(len(num_types) - len(num_types_op)):
            num_types_op.append(0)

    if len(num_types) < len(num_types_op):
        for i in range(len(num_types_op) - len(num_types)):
            num_types.append(0)

    num_type_avg = np.mean(num_types)
    num_type_avg_op = np.mean(num_types_op)

    points_avg = [num_type_avg] * len(num_types)
    points_avg_op = [num_type_avg_op] * len(num_types_op)


    # plt.plot(num_types, color='black', linewidth=2, label='Huskies')
    # plt.plot(points_avg, color='b', linewidth=1, linestyle=':', label='Average of Huskies')
    # plt.plot(num_types_op, color='r', linewidth=2, label='Opponent')
    # plt.plot(points_avg_op, color='g', linewidth=1, linestyle=':', label='Average of Opponent')
    # plt.legend(loc=1)
    # plt.xlabel('Quantum')
    # plt.ylabel('Number of types')
    # plt.savefig(base_path.format(match, '.png'))
    # plt.cla()

    temple_data = pd.DataFrame({'Huskies':num_types, 'Opponent':num_types_op, 
        'Average of Huskies':points_avg, 'Average of Opponent':points_avg_op})
    sns_lp = sns.lineplot(data=temple_data)
    sns_lp.set(xlabel='Time Quantum', ylabel='Number of Event Types')

    # plt.axvline(np.mean(num_types),color='y',linestyle=":",alpha=0.8) 
    # plt.text(np.mean(num_types)+2,0.005,'male_height_mean: %.1fcm' % (np.mean(num_types)), color = 'y')

    fig = sns_lp.get_figure()
    fig.savefig(base_path.format(match, '.png'))
    plt.cla()

    match_ids.append(match)
    avgs.append(num_type_avg)
    avgs_op.append(num_type_avg_op)
    nums_types.append(num_types)
    nums_types_op.append(num_types_op)


pd.DataFrame({'MatchID':match_ids, 'Avg':avgs, 'Number of Types':nums_types,
    'Avg Op':avgs_op, 'Number of Types Op':nums_types_op}).to_csv('./results/task2/num_types.csv')

