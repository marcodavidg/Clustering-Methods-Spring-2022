**Programming Assignment #1**

**Introduction**

With the rapid increase of the information that is handed nowadays, it is not uncommon to see huge datasets for all different types of applications, for instance, speech recognition, market studies, etc. Clustering techniques come as an aid to obtain the most information out of this extensive gathered data, by separating data into meaningful groups that can be analyzed more easily. The purpose of the following assignment is to experiment with different clustering algorithms, showing the cases where they perform the best and the worst; all while also being compared to each other with different kinds of metrics.

**Objectives**

The main objectives of this assignment are the following:

- Implement some of the most known clustering algorithms discussed in the lectures.
- Experiment with different parameters of the algorithms to see different results.
- Compare the results of own implementations with those made by popular clustering libraries.

**Explanation of the experiments done**

The nine different algorithms were tested in all six datasets at once. The result of the clustering algorithms can be seen in Figure 4 and the correct results can be seen in Figure 5. The results also show the execution time of every algorithm on the datasets. Right away, comparing both figures, by some qualitative analysis, we can conclude that the best algorithms are the DBSCAN and OPTICS. However, we can see that OPTICS is the slowest algorithm, so just by the information given by these results, we would choose DBSCAN for these types of clustering problems.
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/75504945-e6b0-49d6-845c-3e3503387aee)

Figure Clustering results side by side for different algorithms. Processing time included.

![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/30c7254b-ca5a-4663-b10d-96b33c31db09)


Figure Correct clustering labels for datasets.

In the first dataset, we can also notice that the single link agglomerative clustering is the only other algorithm that can correctly separate the first dataset, however, it has a lot of problems in other datasets, and we can conclude that the algorithm works only when there is not any sort of connection between clusters, like in datasets 1, 2 and 4.

Moreover, we can also notice that all algorithms can successfully detect the three clusters in dataset 5, because they are clearly defined and not touching each other at all.

In the last dataset, there is not a clustering structure in the data, but still, we can observe the algorithms trying to create clusters with the given information. The only algorithms that we can observe did not create unnecessary clusters were again the DBSCAN and OPTICS. From the last dataset we can also sort of see the intuition behind the search for clusters during different algorithms, such as the clear split at the half for the mean-shift algorithm, as both halves have the same density of points.

However, because qualitative results are sometimes hard to agree upon, quantitative metrics were calculated for the results. The results of four different metrics for every dataset are shown on Table 1 to Table 5.

Table Scores for dataset 1 (Noisy circles)
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/bbebf3d9-9edb-49f0-b519-f822fc048d56)


In table 1, we can see that the algorithms that had the best qualitative performance have the worst internal metric scores. However, this makes sense, as these two internal metrics measure the shapes of clusters, so they tend to have higher scores for non-convex, such as the ones generated correctly in this case by Single, DBSCAN and OPTICS. Moreover, these metrics are not meant to be used to compare different algorithms, but different parameters of the algorithm. To demonstrate different internal metrics interpretations, the results in Figure 6 show the K-Means algorithm evaluated with different K values. We would be able to rely on this metrics to define the K value, however, we can see that the metrics give very different results, and it would be up to us to interpret their meaning and usefulness in each specific situation. Yet, we can see some interesting behaviors such as the DB-Index choosing K=6 for the third and last datasets, as this value is the one that creates the most symmetrical clusters for the given data. Overall, for this set of results the silhouette index is the one that chooses the best K parameter, but this may not be true for other parameters in other algorithms.
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/db68dae0-8577-4d0b-8da3-a00dd5a2dc5d)



Figure K-means with different K value. The Silhouette score is on top of the DB-Index in the bottom right of each result. The best results for each dataset for both metrics are marked with a red box.

Table Scores for dataset 2 (Noisy moons)
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/bb5809cc-8ac2-4e97-99c4-d8cd5d665a0f)


Table Scores for dataset 3 (Blobs with varied variance)
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/448ac5d2-fa36-44c1-a174-86ea0459aa6b)


Table Scores for dataset 4 (Anisotropicly distributed blobs)
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/17891037-2676-4303-99fd-eac31ffba294)


Table Scores for dataset 5 (Equally sized blobs)
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/dc3ea638-93f9-4488-b1e6-00ad17986d69)


What we can fully conclude by looking at the tables is that in most cases, the RAND and NMI indexes agree with the qualitative analysis of clusters, as they give their maximum scores to DBSCAN and OPTICS several times, for the first, second and fifth dataset. However, for dataset 3 the Ward algorithm has the highest scores, and that also reflects in the qualitative analysis, as it correctly classifies points that we Also noting that the fifth dataset is the easiest and every algorithm gets the maximum or same score, as they all achieve a perfect clustering.

**Own Implementations**

The Gaussian Mixture Model and DBSCAN were both implemented, and the results were compared with the methods provided by the sklearn library that is used throughout the rest of this assignment.

- DBSCAN

The results of both implementations can be seen in Figure 7. The epsilon value and minimum points in a core point’s vicinity parameters are the same for both methods (0.1 and 5 respectively). Both results are practically the same, except for some minor differences on the edges of the clusters, possibly due to the order of calculations in the methods, but overall both clustering results are what we would expect.
![image](https://github.com/marcodavidg/ClusteringMethodsSpring2022/assets/11068920/1c7f8041-e2ff-4c28-8908-1fefe20909e7)

Figure DBSCAN implementation comparison

**Discussions**

As always, the results for clustering are very hard to objectively describe, as the result’s quality will highly depend on what the data means to us. However, with the assistance and good interpretation of internal and external metrics, we can aid our decision for the best hyperparameters and algorithms for different situations.

On the other hand, even if judging a result is often tricky, sometimes when the data only has 2 dimensions, we can also qualitatively see which algorithm fits data the best, and realize that some algorithms are clearly meant for different situations, so there are situations where even if the K-Means cannot create non-convex shapes, it may still be the algorithm we choose due to its simplicity and results in that situation.

**Main References**

- Xu, D., Tian, Y. A Comprehensive Survey of Clustering Algorithms. Ann. Data. Sci. 2, 165–193 (2015). <https://doi.org/10.1007/s40745-015-0040-1>
- Wong, K. C. (2015, November). A short survey on data clustering algorithms. In 2015 Second international conference on soft computing and machine intelligence (ISCMI) (pp. 64-68). IEEE.
- Siwei Causevic, Implement Expectation-Maximization Algorithm(EM) in Python from Scratch, 2020. <https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137> (Accessed May 2022).
- Daniel Foley, Gaussian Mixture Modelling (GMM), 2019. <https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f> (Accessed May 2022).
- Moosa Ali, DBSCAN Clustering Algorithm Implementation from scratch, 2021. <https://github.com/Moosa-Ali/DBscan-Clustering-Implementation/blob/main/DBSCAN%20implementation.ipynb>
