# Library Needed
from itertools import permutations, combinations
from math import ceil
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import interpolate


# ClusC
for i in range(len(time_points)-1):
    start_time = time_points[i]
    end_time = time_points[i+1]
    if i == len(time_points)-2:
        end_time += 360

    ppd = PPD[PPD.EventTime >= start_time][PPD.EventTime < end_time]
    
    ppd['count'] = 1
    ppd = ppd.groupby(['OriginPlayerID', 'DestinationPlayerID']).sum()
    ppd = ppd.sort_values(by='count').ix[::-1]

    temple_ppd = ppd.copy()
    if len(temple_ppd)==0:
        continue
    max_count = max(temple_ppd['count'])
    temple_ppd['count'] = temple_ppd['count'] / max_count
    
    d0 = temple_ppd.copy()
    d0 = pd.DataFrame({'i': list(map(lambda x: x[0], d0.index)), 'j': list(map(lambda x: x[1], d0.index)), 'count':d0.to_numpy().T[0]})
    d1 = d0.copy()
    d2 = d0.copy()

    d1 = d1.rename(columns={'i':'j', "j":'k'})
    d2 = d2.rename(columns={'i':'k', "j":'i'})

    merged_data = pd.merge(pd.merge(d0, d1, on='j'), d2, on=['k', 'i'])
    md = merged_data

    md = md.rename(columns={'count_x':'wij', 'count_y':'wjk', 'count':'wki'})

    md['www'] = [(wij * wjk * wki) for wij, wjk, wki in zip(md['wij'], md['wjk'], md['wki'])]
    md['ww'] = [(wij * wki) for wij, wki in zip(md['wij'], md['wki'])]

    md = md.groupby('i').sum()

    md['Clustering coefficient'] = [www/ww for www,ww in zip(md['www'], md['ww'])]
    md = md.sort_values(by='Clustering coefficient').ix[::-1]

    md.to_csv('./results/short/match_{}/{}-Clustering coefficient.csv'.format(match, i))


# ECD
def get_ecd(ppd):

    # ppd = PPD[PPD.EventTime >= start_time][PPD.EventTime < end_time]
    ppd 
    ppd = ppd.rename(columns={'OriginPlayerID':'i', 'DestinationPlayerID':'j'})
    ppd['count'] = 1
    ppd = ppd.groupby(['i', 'j']).sum()
    ppd = ppd.sort_values(by='count').ix[::-1]

    ppd = pd.DataFrame({'i': list(map(lambda x: x[0], ppd.index)), 'j': list(map(lambda x: x[1], ppd.index)), 'count':ppd.to_numpy().T[0]})
    
    players = list(set(ppd['i']))
    players = sorted(players)

    num_players = len(players)

    mat = np.zeros((num_players, num_players))

    for i, iplayer in enumerate(players):
        for j, jplayer in enumerate(players):
            qr = ppd[ppd['i']==iplayer][ppd['j']==jplayer].to_numpy()
            if len(qr) > 0:
                mat[i][j] = qr[-1][-1]
            else:
                mat[i][j] = 0
    
    A = mat

    G = nx.Graph()

    for i, p in enumerate(players):
        G.add_node(i)

    for i in range(len(players)):
        for j in range(len(players)):
            G.add_edge(i, j, weight=A[i][j])
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        # print(players)
        return None

    ps = []
    vs = []
    for p, v in ec.items():
        ps.append(p)
        vs.append(v)

    max_ec = max(vs)
    player_with_ec = ps[vs.index(max_ec)]

    dispersion_ec = np.std(vs)

    return dispersion_ec


# AlgeC
def get_ac(ppd):

    # ppd = PPD[PPD.EventTime >= start_time][PPD.EventTime < end_time]
    ppd = ppd.rename(columns={'OriginPlayerID':'i', 'DestinationPlayerID':'j'})
    ppd['count'] = 1
    ppd = ppd.groupby(['i', 'j']).sum()
    ppd = ppd.sort_values(by='count').ix[::-1]

    ppd = pd.DataFrame({'i': list(map(lambda x: x[0], ppd.index)), 'j': list(map(lambda x: x[1], ppd.index)), 'count':ppd.to_numpy().T[0]})
    
    players = list(set(ppd['i']))
    players = sorted(players)

    num_players = len(players)

    mat = np.zeros((num_players, num_players))

    for i, iplayer in enumerate(players):
        for j, jplayer in enumerate(players):
            qr = ppd[ppd['i']==iplayer][ppd['j']==jplayer].to_numpy()
            if len(qr) > 0:
                mat[i][j] = qr[-1][-1]
            else:
                mat[i][j] = 0
    
    A = mat

    S = np.zeros((num_players, num_players))

    for i, line in enumerate(A):
        S[i][i] = sum(line)

    L = S - A

    eigenvalueL, eigenvectorL = np.linalg.eig(L)

    if len(eigenvalueL) == 0:
        return 0
    elif len(eigenvalueL) == 1:
        return eigenvalueL[0].real

    return sorted(eigenvalueL)[1].real


# Team Formation & Triadic Configuration
from itertools import combinations, permutations

def get_tf(ppd, period):

    # ppd = PPD[PPD.EventTime >= start_time][PPD.EventTime < end_time]
    ppd = ppd.rename(columns={'OriginPlayerID':'i', 'DestinationPlayerID':'j'})
    ppd['count'] = 1
    ppd = ppd.groupby(['i', 'j']).sum()
    ppd = ppd.sort_values(by='count').ix[::-1]

    ppd = pd.DataFrame({'i': list(map(lambda x: x[0], ppd.index)), 'j': list(map(lambda x: x[1], ppd.index)), 'count':ppd.to_numpy().T[0]})
    
    players = list(set(ppd['i']))
    players = sorted(players)

    num_players = len(players)

    triples = combinations(players, 3)
    triples = list(triples)

    all_edges = [(x, y) for x,y in zip(ppd.i, ppd.j)]

    triangles = []

    for a, b, c in triples:
        if ((a,b) in all_edges or (b,a) in all_edges) and ((a,c) in all_edges or (c,a) in all_edges) and ((b,c) in all_edges or (c,b) in all_edges):
            triangles.append((a,b,c))

    values = [0] * len(triangles)

    for i, t in enumerate(triangles):
        edges = list(permutations(t, 2))
        for e in edges:
            qr = ppd[ppd.i == e[0]][ppd.j == e[1]].to_numpy()
            if len(qr) > 0:
                values[i] += qr[-1][-1]

    tvd = pd.DataFrame({'triangle':triangles, 'value':values})
    tvd = tvd.sort_values(by='value').ix[::-1]

    if len(tvd.value) <= 0:
        return None

    max_value = max(tvd.value)
    
    threshold = max_value * 0.75

    triangles_filtered = tvd[tvd['value'] >= threshold]
    tfd = triangles_filtered

    def count_shared_node(t0, t1):
        return len(set(t0) & set(t1)) 

    tf = list(tfd.triangle.to_numpy())
    pairs = combinations(tf, 2)
    pairs = list(pairs)

    weight = [count_shared_node(t0, t1)*(tvd[tvd.triangle==t0].value.to_numpy()[0]+tvd[tvd.triangle==t1].value.to_numpy()[0])/2 for t0, t1 in pairs]

    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]

    new_graph = pd.DataFrame({'i':a, 'j':b, 'weight':weight})

    new_graph = new_graph[new_graph['weight'] > 0]

    newMat = np.zeros((len(tf), len(tf)))
    for i, inode in enumerate(tf):
        for j, jnode in enumerate(tf):
            if inode==jnode:
                newMat[i][j] = 3 * tvd[tvd.triangle==inode].value.to_numpy()[0]
                continue
            qr = new_graph[new_graph['i']==inode][new_graph['j']==jnode].to_numpy()
            if len(qr) > 0:
                newMat[i][j] = qr[-1][-1]

    new_eigenvalue, new_eigenvector = np.linalg.eig(newMat)
    max_new_eigenvalue = max(new_eigenvalue)

    new_graph['eigenvalue'] = max_new_eigenvalue

    new_graph.to_csv(base_path + 'period-{}-team_inf.csv'.format(period))

    return len(triangles_filtered)


# Long Passing Link
def get_long_link_num(ppd):

    ppd = ppd.rename(columns={'OriginPlayerID':'i', 'DestinationPlayerID':'j'})

    pairs = [(i, j) for i,j in zip(ppd.i, ppd.j)]

    total = 0
    count = 0
    for j in range(1, len(pairs)):
        if pairs[j][0] == pairs[j-1][1]:
            count += 1
        else:
            if count >= 2:
                total += 1
            count = 0
    
    return total
