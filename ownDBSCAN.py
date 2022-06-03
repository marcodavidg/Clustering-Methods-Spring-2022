from sklearn import cluster, datasets, mixture, metrics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.preprocessing import StandardScaler

centers = [(0, 1), (15, 5) , (18,20)]
cluster_std = [2, 3.4, 2.5]


dbscan = cluster.DBSCAN(eps=0.12, min_samples = 5)

X, y= datasets.make_blobs(n_samples=2000, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
X = StandardScaler().fit_transform(X)

dbscan.fit(X)
if hasattr(dbscan, "labels_"):
    y_pred = dbscan.labels_.astype(int)
else:
    y_pred = dbscan.predict(X)

colors = np.array(
    list(
        islice(
            cycle(
                [
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                    "#aade00",
                ]
            ),
            int(max(y_pred) + 1),
        )
    )
)
# add black color for outliers
colors = np.append(colors, ["#000000"])
plt.subplot(1,2, 1)
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

plt.title('SKlearn implementation')

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xticks(())
plt.yticks(())


# plt.show()

def check_core_point(eps,minPts, df, index):
    #get points from given index
    x, y = df.iloc[index]['X']  ,  df.iloc[index]['Y']
    
    #check available points within radius
    temp =  df[((np.abs(x - df['X']) <= eps) & (np.abs(y - df['Y']) <= eps)) & (df.index != index)]
    
    #check how many points are present within radius
    if len(temp) >= minPts:
        #return format (dataframe, is_core, is_border, is_noise)
        return (temp.index , True, False, False)
    
    elif (len(temp) < minPts) and len(temp) > 0:
        #return format (dataframe, is_core, is_border, is_noise)
        return (temp.index , False, True, False)
    
    elif len(temp) == 0:
        #return format (dataframe, is_core, is_border, is_noise)
        return (temp.index , False, False, True)


def cluster_with_stack(eps, minPts, df):
    
    #initiating cluster number
    C = 1
    #initiating stacks to maintain
    current_stack = set()
    unvisited = list(df.index)
    clusters = []
    
    
    while (len(unvisited) != 0): #run until all points have been visited

        #identifier for first point of a cluster
        first_point = True
        
        #choose a random unvisited point
        current_stack.add(random.choice(unvisited))
        
        while len(current_stack) != 0: #run until a cluster is complete
            
            #pop current point from stack
            curr_idx = current_stack.pop()
            
            #check if point is core, neighbour or border
            neigh_indexes, iscore, isborder, isnoise = check_core_point(eps, minPts, df, curr_idx)
            
            #dealing with an edge case
            if (isborder & first_point):
                #for first border point, we label it aand its neighbours as noise 
                clusters.append((curr_idx, 0))
                clusters.extend(list(zip(neigh_indexes,[0 for _ in range(len(neigh_indexes))])))
                
                #label as visited
                unvisited.remove(curr_idx)
                unvisited = [e for e in unvisited if e not in neigh_indexes]
    
                continue
                
            unvisited.remove(curr_idx) #remove point from unvisited list
            
            
            neigh_indexes = set(neigh_indexes) & set(unvisited) #look at only unvisited points
            
            if iscore: #if current point is a core
                first_point = False
                
                clusters.append((curr_idx,C)) #assign to a cluster
                current_stack.update(neigh_indexes) #add neighbours to a stack

            elif isborder: #if current point is a border point
                clusters.append((curr_idx,C))
                
                continue

            elif isnoise: #if current point is noise
                clusters.append((curr_idx, 0))
                
                continue
                
        if not first_point:
            #increment cluster number
            C+=1
        
    return clusters
            

#radius of the circle defined as 0.6
eps = 0.12
#minimum neighbouring points set to 3
minPts = 5

data = pd.DataFrame(X, columns = ["X", "Y"] )
clustered = cluster_with_stack(eps, minPts, data)

idx , cluster = list(zip(*clustered))

cluster_df = pd.DataFrame(clustered, columns = ["idx", "cluster"])

colors = np.array(
    list(
        islice(
            cycle(
                [
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                    "#aade00",
                ]
            ),
            int(max(np.unique(cluster)) + 1),
        )
    )
)
# add black color for outliers
colors[0] = "#000000"

plt.subplot(1,2, 2)
# plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], 
    	X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], 
    	s=10, 
    	label=f"Cluster{clust}",
    	color=colors[clust]
    	)

plt.title('Own implementation')

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xticks(())
plt.yticks(())

plt.show()