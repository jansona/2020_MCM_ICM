#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import pandas
pd = pandas

time_scales = ['long', 'medium', 'short']
base_path = time_scales[1]


# In[2]:


passing_data = pandas.read_csv("./2020_Problem_D_DATA/passingevents.csv")


# # In[3]:


# passing_data.head()


# # # Centroid coordinates and dispersion

# # ## long

# # In[4]:


# passing_data


# # In[5]:

for match in range(1, 39):

    print(match)

    ppd = passing_data.loc[:, ['TeamID', 'OriginPlayerID', 'EventOrigin_x', 'EventOrigin_y', 'MatchID']]
    ppd = ppd[ppd['TeamID'].isin(['Huskies'])][ppd['MatchID']==match]
    ppd = ppd.loc[:, ['TeamID', 'OriginPlayerID', 'EventOrigin_x', 'EventOrigin_y']]

    shared_ppd = ppd.copy()


    # In[6]:


    X_mean, Y_mean = ppd[['EventOrigin_x', 'EventOrigin_y']].mean()


    # In[7]:


    pomd = ppd.groupby('OriginPlayerID').mean()


    # In[8]:


    pomd['distance'] = [((x - X_mean)**2 + (y - Y_mean)**2)**0.5 for x,y in zip(pomd['EventOrigin_x'],pomd['EventOrigin_y'])]


    pomd.std()[2]


    # In[78]:


    centroid_coordinates_and_dispersion = pd.DataFrame({'Centroid coordinates':[(X_mean, Y_mean)], 'dispersion':pomd.std()[2]})


    # In[79]:


    centroid_coordinates_and_dispersion.to_csv("./results/{0}/{1}-centroid_coordinates_and_dispersion.csv".format(base_path, match))


    # # Clustering coefficient

    # ## long

    pure_passing_data = passing_data.loc[:, ['TeamID', 'OriginPlayerID', 'DestinationPlayerID', 'MatchID']]
    ppd = pure_passing_data
    ppd = ppd[ppd['TeamID'].isin(['Huskies'])][ppd['MatchID']==match]
    ppd = ppd.loc[:, ['TeamID', 'OriginPlayerID', 'DestinationPlayerID']]

    ppd['count'] = 1
    ppd = ppd.groupby(['OriginPlayerID', 'DestinationPlayerID']).sum()

    # In[15]:


    ppd = ppd.sort_values(by='count').ix[::-1]
    temple_ppd = ppd.copy()
    max_count = max(temple_ppd['count'])
    temple_ppd['count'] = temple_ppd['count'] / max_count

    d0 = temple_ppd.copy()
    d0 = pandas.DataFrame({'i': list(map(lambda x: x[0], d0.index)), 'j': list(map(lambda x: x[1], d0.index)), 'count':d0.to_numpy().T[0]})
    d1 = d0.copy()
    d2 = d0.copy()


    # In[18]:


    d1 = d1.rename(columns={'i':'j', "j":'k'})
    d2 = d2.rename(columns={'i':'k', "j":'i'})


    merged_data = pd.merge(pd.merge(d0, d1, on='j'), d2, on=['k', 'i'])
    md = merged_data
    md


    # In[21]:


    md = md.rename(columns={'count_x':'wij', 'count_y':'wjk', 'count':'wki'})


    # In[22]:


    md['www'] = [(wij * wjk * wki) for wij, wjk, wki in zip(md['wij'], md['wjk'], md['wki'])]
    md['ww'] = [(wij * wki) for wij, wki in zip(md['wij'], md['wki'])]


    # In[24]:


    md = md.groupby('i').sum()


    # In[25]:


    md['Clustering coefficient'] = [www/ww for www,ww in zip(md['www'], md['ww'])]
    md = md.sort_values(by='Clustering coefficient').ix[::-1]
    md


    # In[81]:


    md.to_csv('./results/{0}/{1}-Clustering coefficient.csv'.format(base_path, match))


    # # Largest eigenvalue of the adjacency matrix

    # ## long

    # In[26]:


    # pure_passing_data = passing_data.loc[:, ['TeamID', 'OriginPlayerID', 'DestinationPlayerID']]
    # ppd = pure_passing_data
    # ppd = ppd[ppd['TeamID'].isin(['Huskies'])]


    # In[27]:


    ppd = d0.copy() # 使用上一章的d0数据


    # In[28]:


    players = list(set(ppd['i']))
    players = sorted(players)

    import numpy as np


    # In[30]:


    mat = np.zeros((30, 30))


    # In[31]:


    for i, iplayer in enumerate(players):
        for j, jplayer in enumerate(players):
            qr = ppd[ppd['i']==iplayer][ppd['j']==jplayer].to_numpy()
            if len(qr) > 0:
                mat[i][j] = qr[-1][-1]
            else:
                mat[i][j] = 0


    # In[32]:


    A = mat


    eigenvalue, eigenvector = np.linalg.eig(mat)


    # In[34]:


    eigenvalue


    # In[35]:


    lec = max(eigenvalue)
    lec


    # In[36]:


    eigenvector


    # In[84]:


    pd.DataFrame({'Largest eigenvalue':[lec]}).to_csv('./results/{0}/{1}-Largest eigenvalue.csv'.format(base_path, match))


    # # Algebraic connectivity

    # ## long

    # In[37]:


    S = np.zeros((30, 30))


    # In[38]:


    for i, line in enumerate(A):
        S[i][i] = sum(line)


    # In[39]:


    S


    # In[40]:


    L = S - A


    # In[41]:


    L


    # In[85]:


    eigenvalueL, eigenvectorL = np.linalg.eig(L)


    # In[87]:


    sorted(eigenvalueL)[1]


    # In[88]:


    pd.DataFrame({'Algebraic connectivity':[sorted(eigenvalueL)[1]]}).to_csv('./results/{0}/{1}-Algebraic connectivity.csv'.format(base_path, match))


    # # Eigenvector centrality

    # ## long

    # In[42]:


    # 不确定
    eigenvector[list(eigenvalue).index(lec)]


    # In[43]:


    import networkx as nx


    # In[44]:


    G = nx.Graph()


    # In[45]:


    for i, p in enumerate(players):
        G.add_node(i)


    # In[46]:


    for i in range(len(players)):
        for j in range(len(players)):
            G.add_edge(i, j, weight=A[i][j])


    # In[89]:


    ec = nx.eigenvector_centrality_numpy(G, weight='weight')
    ec


    # In[93]:


    ps = []
    vs = []
    for p, v in ec.items():
        ps.append(p)
        vs.append(v)


    # In[94]:

    max_ec = max(vs)
    player_with_ec = ps[vs.index(max_ec)]

    dispersion_ec = np.var(vs)

    pd.DataFrame({'player':ps, 'Eigenvector centrality':vs,
        'max':max_ec, 'player with max':player_with_ec, 'dispersion':dispersion_ec}).to_csv('./results/{0}/{1}-Eigenvector centrality.csv'.format(base_path, match))


    # # triadic configurations

    # ## long

    # In[48]:


    from itertools import combinations, permutations


    # In[49]:


    triples = combinations(players, 3)


    # In[50]:


    triples = list(triples)
    triples


    # In[51]:


    all_edges = [(x, y) for x,y in zip(ppd.i, ppd.j)]


    # In[52]:


    triangles = []

    for a, b, c in triples:
        if ((a,b) in all_edges or (b,a) in all_edges) and ((a,c) in all_edges or (c,a) in all_edges) and ((b,c) in all_edges or (c,b) in all_edges):
            triangles.append((a,b,c))


    # In[53]:


    len(triangles)


    # # triadic configurations value

    # ## long

    # In[54]:


    values = [0] * len(triangles)


    # In[55]:


    for i, t in enumerate(triangles):
        edges = list(permutations(t, 2))
        for e in edges:
            qr = ppd[ppd.i == e[0]][ppd.j == e[1]].to_numpy()
            if len(qr) > 0:
                values[i] += qr[-1][-1]


    # In[56]:


    tvd = pd.DataFrame({'triangle':triangles, 'value':values})


    # In[57]:


    tvd = tvd.sort_values(by='value').ix[::-1]
    tvd


    # In[58]:


    max_value = max(tvd.value)


    # In[97]:


    pd.DataFrame({'amount':[len(triangles)], 'max value':[max_value]}).to_csv('./results/{0}/{1}-triadic configurations.csv'.format(base_path, match))


    # # team formationteam formation

    # In[59]:


    threshold = max_value / 2


    # In[60]:


    triangles_filtered = tvd[tvd['value'] >= threshold]
    tfd = triangles_filtered


    # In[61]:


    tfd


    # In[62]:


    # def count_shared_edge(t0, t1):
    #     num_shared_node = len(set(t0) & set(t1))
    #     if num_shared_node <= 1:
    #         return 0
    #     elif num_shared_node == 2:

    def count_shared_node(t0, t1):
        return len(set(t0) & set(t1)) 


    # In[63]:


    tf = list(tfd.triangle.to_numpy())


    # In[64]:


    pairs = combinations(tf, 2)
    pairs = list(pairs)
    len(pairs)


    # In[65]:


    weight = [count_shared_node(t0, t1)*(tvd[tvd.triangle==t0].value.to_numpy()[0]+tvd[tvd.triangle==t1].value.to_numpy()[0])/2 for t0, t1 in pairs]


    # In[66]:


    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]


    # In[67]:


    new_graph = pd.DataFrame({'i':a, 'j':b, 'weight':weight})


    # In[68]:


    new_graph = new_graph[new_graph['weight'] > 0]


    # In[69]:


    new_graph


    # In[98]:
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

    new_graph.to_csv('./results/{0}/{1}-team formation.csv'.format(base_path, match))

    # newG = nx.Graph()

    # for n in a:
    #     newG.add_node(n)

    # for n in b:
    #     newG.add_node(n)

    # for i, w in enumerate(weight):
    #     newG.add_edge(a[i], b[i], weight=w)
    # # In[99]:

    # newG = nx.Graph()

    # for n in a:
    #     newG.add_node(n)

    # for n in b:
    #     newG.add_node(n)

    # for i, w in enumerate(weight):
    #     newG.add_edge(a[i], b[i], weight=w)


# # In[100]:


# import matplotlib.pyplot as plt


# # In[128]:


# nx.draw_spring(G)


# # In[129]:


# nx.draw_spring(newG)


# # In[ ]:




