# 1.smooth on graph
# 2. gene clustering
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import scanpy as sc

class AgglomerativeClustering:
    def __init__(self, distance_matrix, update_method='single', stopping_threshold=None, n_clusters=None):
        self.distance_matrix = squareform(distance_matrix)
        self.update_method = update_method
        self.stopping_threshold = stopping_threshold
        self.n_clusters = n_clusters

    def fit(self):
        linkage_matrix = sch.linkage(self.distance_matrix, method=self.update_method)
        if self.stopping_threshold:
            cluster_labels = sch.fcluster(linkage_matrix, self.stopping_threshold, criterion='distance')
        elif self.n_clusters:
            cluster_labels = sch.fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')
        else:
            cluster_labels = None
        return linkage_matrix, cluster_labels
    
    def plot_dendrogram(self, linkage_matrix):
        Z = linkage_matrix
        no_labels = False
        if Z.shape[0] > 10:
            no_labels = True
        if self.stopping_threshold is not None:
            R = sch.dendrogram(Z, no_labels=no_labels)
            # print(R['icoord'])
            dist_threshold_ycoord = self.stopping_threshold
            plt.plot([0, max(max(R['icoord']))], [dist_threshold_ycoord, dist_threshold_ycoord], '--', c='r')
            plt.title('Hierarchical Clustering Dendrogram (Stop distance = {})'.format(self.stopping_threshold))
        elif self.n_clusters is not None:
            plt.title('Hierarchical Clustering Dendrogram (Number of Clusters = {})'.format(self.n_clusters))
            sch.dendrogram(Z, truncate_mode='lastp', p=self.n_clusters, no_labels=no_labels)
        else:
            plt.title('Hierarchical Clustering Dendrogram')
            sch.dendrogram(Z, no_labels=no_labels)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

def clustring_genes(expr_mat, corr_thr=None, n_cluster=10, method='complete', plot=False):
    gene_corr = np.corrcoef(expr_mat.T)
    # gene_corr, _ = spearmanr(expr_mat)
    gene_corr = np.round(gene_corr, 9)
    dist_mat = 1 - gene_corr
    # print(np.min(gene_corr))
    # Only if stopping_threshold is None, n_clusters will work.
    if corr_thr is None:
        ac = AgglomerativeClustering(dist_mat, update_method=method, stopping_threshold=None, n_clusters=n_cluster)
    else:
        ac = AgglomerativeClustering(dist_mat, update_method=method, stopping_threshold=1-corr_thr, n_clusters=n_cluster)
    linkage_matrix, cluster_labels = ac.fit()
    # print(cluster_labels)
    if plot:
        ac.plot_dendrogram(linkage_matrix)

    num_clusters = len(set(cluster_labels))
    expr_mat_clustered = []
    cluster2gene = {}
    for i in range(num_clusters):
        ids = np.where(cluster_labels==(i+1))[0]
        mean_expr = expr_mat[:, ids].mean(axis=1)
        expr_mat_clustered.append(mean_expr)
        cluster2gene[i+1] = ids.tolist()
    expr_mat_clustered = np.array(expr_mat_clustered).T
    return expr_mat_clustered, cluster2gene

def avg_genes_by_clusters(expr_mat, cluster2gene):
    num_clusters = len(cluster2gene)
    expr_mat_clustered = []
    for i in range(num_clusters):
        ids = cluster2gene[i+1]
        mean_expr = expr_mat[:, ids].mean(axis=1)
        expr_mat_clustered.append(mean_expr)
    expr_mat_clustered = np.array(expr_mat_clustered).T
    return expr_mat_clustered

def process_test_data(adata, refgenes):
    common_genes = np.intersect1d(refgenes, adata.var_names)
    if len(common_genes) == len(refgenes):
        adata_ = adata[:, refgenes].copy()
        # adata_ = adata.copy()
    else:
        gene_prop = len(common_genes) / len(refgenes)
        print(f'Warning: test adata does not contain all reference genes (only {100*gene_prop:.2f}% used).')
        exp_mat = np.zeros((adata.shape[0], len(refgenes)))
        exp_df = pd.DataFrame(exp_mat, columns=refgenes, index=adata.obs.index)
        
        try:
            exp_df.loc[:, common_genes] = adata[:, common_genes].X
        except:
            exp_df.loc[:, common_genes] = adata[:, common_genes].X.todense()
        
        adata_ = sc.AnnData(exp_df)
        adata_.obs = adata.obs
    return adata_

def adata_dense_X(adata):
    try:
        X = adata.X.toarray()
    except:
        X = adata.X
    return X


if __name__== "__main__":
    expr_mat = np.random.rand(5000, 2000)
    spatial_coords = np.random.rand(5000, 2)
    expr_mat_clustered, cluster2gene = clustring_genes(expr_mat, corr_thr=None, n_cluster=5, plot=True)