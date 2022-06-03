import time
import warnings
import ownGMM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import cluster, datasets, mixture, metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn_extra.cluster import KMedoids
from scipy import stats
from scipy.special import logsumexp

np.random.seed(3)

# ============
# Generate toy datasets.
# ============
n_samples = 1000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# centers = [(0, 4), (5, 5) , (8,2)]
# cluster_std = [1.2, 1, 1.1]
# blobs = datasets.make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.7, -0.7], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    # (no_structure, {}),
]


# ========================
# Calculate clusters with all algorithms for all datasets
# ========================
metrics_values = np.zeros((5,9,4)) 

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # ========================
    # Create cluster objects
    # ========================

    # -------------- K-Means -------------
    kmeans = KMeans(n_clusters=params["n_clusters"], random_state=0)
    kmeans3 = KMeans(n_clusters=3, random_state=0)
    kmeans2 = KMeans(n_clusters=2, random_state=0)
    kmeans4 = KMeans(n_clusters=4, random_state=0)
    kmeans6 = KMeans(n_clusters=6, random_state=0)
    kmedoids = KMedoids(n_clusters=params["n_clusters"], random_state=0)
    # two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])


    # -------------- Mixture decomposition -------------

    # own implementation 
    # https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137
    # https://github.com/VXU1230/Medium-Tutorials/blob/master/em/em.py

    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"], covariance_type="full"
    )

    if params["n_clusters"] < 2:
    
        random_params = initialize_random_params()
        unsupervised_forecastsforecasts, unsupervised_posterior, unsupervised_loglikelihoods, learned_params = run_em(X, random_params)
        print("total steps: ", len(unsupervised_loglikelihoods))
        # plt.plot(unsupervised_loglikelihoods)
        # plt.title("unsupervised log likelihoods")
        # plt.savefig("unsupervised.png")
        # plt.close()

        weights = [1 - learned_params["phi"], learned_params["phi"]]
        means = [learned_params["mu0"], learned_params["mu1"]]
        covariances = [learned_params["sigma0"], learned_params["sigma1"]]
        
        model = mixture.GaussianMixture(n_components=params["n_clusters"],
                                covariance_type='full',
                                weights_init=weights)
        model.fit(X)
        print("\nscikit learn:\n\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
                   % (model.weights_[1], model.means_[0, :], model.means_[1, :], model.covariances_[0, :], model.covariances_[1, :]))
        sklearn_forecasts, posterior_sklearn = model.predict(X), model.predict_proba(X)[:,1]



        output_df = pd.DataFrame({'unsupervised_forecastsforecasts': unsupervised_forecastsforecasts, 
                                  'unsupervised_posterior': unsupervised_posterior[:, 1],
                                  'sklearn_forecasts': sklearn_forecasts,
                                  'posterior_sklearn': posterior_sklearn})
        print("\n%s%% of forecasts matched." % (output_df[output_df["unsupervised_forecastsforecasts"] == output_df["sklearn_forecasts"]].shape[0] /output_df.shape[0] * 100))   


        plt.subplot(2,2, plot_num)
        if i_dataset == 0:
            plt.title("GMM", size=18)

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
                        ]
                    ),
                    int(max(sklearn_forecasts) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[sklearn_forecasts])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

        plt.subplot(2,2, plot_num)
        if i_dataset == 0:
            plt.title("OWN", size=18)

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
                        ]
                    ),
                    int(max(unsupervised_forecastsforecasts) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[unsupervised_forecastsforecasts])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
        
        if plot_num == 5:
            break
        else:
            continue


    #---------- Hierarchical Clustering -------------
    # ‘ward’ minimizes the variance of the clusters being merged.
    # ‘single’ uses the minimum of the distances between all observations of the two sets.

    # connectivity matrix
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    single_linkage = cluster.AgglomerativeClustering(
        linkage="single",
        affinity="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )

    ward_linkage = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )

    birch = cluster.Birch(n_clusters=params["n_clusters"])

    #---------- Density Based -------------

    # OWN DBSCAN 
    # https://github.com/Moosa-Ali/DBscan-Clustering-Implementation/blob/main/DBSCAN%20implementation.ipynb
    # https://becominghuman.ai/dbscan-clustering-algorithm-implementation-from-scratch-python-9950af5eed97

    dbscan = cluster.DBSCAN(eps=params["eps"])
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )

    #---------- Mode-Seeking -------------
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # ========================
    # Print graphs
    # ========================

    clustering_algorithms = (
        ("KMeans", kmeans),
        # ("KMeans K = 2", kmeans2),
        # ("KMeans K = 3", kmeans3),
        # ("KMeans K = 4", kmeans4),
        # ("KMeans K = 6", kmeans6),
        ("kmedoids", kmedoids),
        ("GMM", gmm),
        ("Single", single_linkage),
        ("Ward", ward_linkage),
        ("BIRCH", birch),
        ("DBSCAN", dbscan),
        ("OPTICS", optics),
        ("Mean-Shift", ms),
    )
    
    algorith_index = 0
    for name, algorithm in clustering_algorithms:
        algorith_index += 1

        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X)

        t1 = time.time()

        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)


        DB_Index = metrics.davies_bouldin_score(X, y_pred)
        Silhoulette = metrics.silhouette_score(X, y_pred, metric='euclidean')
        Rand_Index = metrics.rand_score(y, y_pred)
        NMI = metrics.normalized_mutual_info_score(y, y_pred)
        np. set_printoptions(suppress=True)

        metrics_values[i_dataset, algorith_index - 1, 0] = np.round(DB_Index,3)
        metrics_values[i_dataset, algorith_index - 1, 1] = np.round(Silhoulette,3)
        metrics_values[i_dataset, algorith_index - 1, 2] = np.round(Rand_Index,3)
        metrics_values[i_dataset, algorith_index - 1, 3] = np.round(NMI,3)

        print("Name: ", name)
        print("Dataset: ", i_dataset)
        # print("inertia: ", algorithm.inertia_)
        print("Davies-Bouldin Index: ", DB_Index)
        print("Silhouette Index: ", Silhoulette)
        print("Rand Index: ", Rand_Index)
        print("Normalized Mutual Information: ", NMI)



        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=15)

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
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # print both internal metrics inside plot
        bothInternal = str(metrics_values[i_dataset, algorith_index - 1, 1]) + "\n" + str(metrics_values[i_dataset, algorith_index - 1, 0])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            # bothInternal,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1

for i_dataset, _ in enumerate(datasets):
    df = pd.DataFrame (metrics_values[i_dataset,:,:])
    filepath = 'dataset' + str(i_dataset) + '.xlsx'
    df.to_excel(filepath, index=False)


print(metrics_values)
plt.show()
