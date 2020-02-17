#!/usr/bin/env python
# coding: utf-8
import numpy
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

def get_raw_data_filtered(match_num, is_oppnent):
    if is_oppnent:
        player_list = list(set(fullevent_data.TeamID) - {'Huskies'})
    else:
        player_list = ['Huskies']
    
    passing_data = pd.read_csv("./2020_Problem_D_DATA/passingevents.csv")
    PPD = passing_data.loc[:, ['MatchID', 'TeamID', 'OriginPlayerID', 'DestinationPlayerID', 'EventTime', 'EventOrigin_x', 'EventOrigin_y', 'EventDestination_x', 'EventDestination_y']]
    PPD = PPD[PPD['MatchID']==match_num][PPD['TeamID'].isin(player_list)]

    return PPD


fullevent_data = pd.read_csv("./2020_Problem_D_DATA/fullevents.csv")

base_path = './results/task3/new_task_3/{}{}'

match_ids = []
avgs = []
avgs_op = []
nums_types = []
nums_types_op = []

for match in [5, 16, 25]:
    print(match)

    pfed = fullevent_data.loc[:, ['MatchID', 'TeamID', 'MatchPeriod', 'EventTime', 'EventType', 'EventSubType']]
    pfed = pfed[pfed['MatchID']==match] # [pfed['TeamID'].isin(list(set(fullevent_data.TeamID) - {'Huskies'}))]

    pfed = pfed.reset_index()
    pfed = pfed.loc[:, ['MatchID', 'TeamID', 'MatchPeriod', 'EventTime', 'EventType', 'EventSubType']]

    last_1H = max(pfed[pfed['MatchPeriod']=='1H'].EventTime)

    fed_2H = pfed[pfed['MatchPeriod']=='2H']
    fed_2H['EventTime'] += last_1H

    pfed = pd.concat([pfed[pfed['MatchPeriod']=='1H'], fed_2H])

    pfed

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
        num_points = ceil(duration / 180)
        if num_points <= 0:
            continue
        sub_quantum = duration / num_points

        while start_time < end_time:
            time_points.append(start_time)
            start_time += sub_quantum

    time_points.append(pfed.iloc[points[-1]].EventTime)

    nums_dist_shift = []
    nums_dist_shift_op = []

    for is_opponent in [False, True]:

        PPD = get_raw_data_filtered(match, is_opponent)

        for i in range(len(time_points)-1):
            start_time = time_points[i]
            end_time = time_points[i+1]
            if i == len(time_points)-2:
                end_time += 180

            ppd = PPD[PPD.EventTime >= start_time][PPD.EventTime < end_time]

            dists = [((xt-x0)**2+(yt-y0)**2)**0.5 >= 20 for x0,y0,xt,yt in zip(ppd.EventOrigin_x,ppd.EventOrigin_y,ppd.EventDestination_x,ppd.EventDestination_y)]

            count = 1
            for i in range(len(dists)-1):
                if dists[i] != dists[i+1]:
                    count += 1
            if not is_opponent:
                nums_dist_shift.append(count)
            else:
                nums_dist_shift_op.append(count)

    # num_type_avg = np.mean(num_types)
    # num_type_avg_op = np.mean(num_types_op)

    # points_avg = [num_type_avg] * len(num_types)
    # points_avg_op = [num_type_avg_op] * len(num_types_op)


    # plt.plot(nums_dist_shift, color='black', linewidth=2, label='Huskies')
    # # plt.plot(points_avg, color='b', linewidth=1, linestyle=':', label='average of Huskies')
    # plt.plot(nums_dist_shift_op, color='r', linewidth=2, label='Opponent')
    # # plt.plot(points_avg_op, color='g', linewidth=1, linestyle=':', label='average of Opponent')
    # plt.legend(loc=1)
    # plt.xlabel('Quantum')
    # plt.ylabel('Number of Shift Passing')
    # plt.savefig(base_path.format(match, '.png'))
    # plt.cla()

    # print(nums_dist_shift)

    pd.DataFrame({'num_dist_shift':nums_dist_shift}).to_csv('./results/short/match_{}/num_dist_shift.csv'.format(match))
    pd.DataFrame({'num_dist_shift':nums_dist_shift_op}).to_csv('./results/short/opponent/match_{}/num_dist_shift.csv'.format(match))

    nums_dist_shift = list(filter(lambda x: x>1, nums_dist_shift))
    nums_dist_shift_op = list(filter(lambda x: x>1, nums_dist_shift_op))

    if len(nums_dist_shift) > len(nums_dist_shift_op):
        for i in range(len(nums_dist_shift) - len(nums_dist_shift_op)):
            nums_dist_shift_op.append(1)

    if len(nums_dist_shift) < len(nums_dist_shift_op):
        for i in range(len(nums_dist_shift_op) - len(nums_dist_shift)):
            nums_dist_shift.append(1)

    temple_data = pd.DataFrame({'Huskies':nums_dist_shift, 'Opponent':nums_dist_shift_op})
    sns_lp = sns.lineplot(data=temple_data)
    sns_lp.set(xlabel='Time Quantum', ylabel='Frequency of Passing Types Shift')
    fig = sns_lp.get_figure()
    fig.savefig(base_path.format(match, '.png'))
    plt.cla()

    # match_ids.append(match)
    # avgs.append(num_type_avg)
    # avgs_op.append(num_type_avg_op)
    # nums_types.append(num_types)
    # nums_types_op.append(num_types_op)


# pd.DataFrame({'MatchID':match_ids, 'Avg':avgs, 'Number of Types':nums_types,
    # 'Avg Op':avgs_op, 'Number of Types Op':nums_types_op}).to_csv('./results/task2/num_types.csv')

