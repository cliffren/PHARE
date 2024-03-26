#%%
import scanpy as sc
import numpy as np
import pandas as pd
import pickle
import os

def get_markers(adata_log, topk='all', genetype='up', if_clean=True):
    markers = {}
    markers_list = []
    for celltype in adata_log.obs['celltype2'].unique():
        gene_names = adata_log.uns['rank_genes_groups']['names'][celltype]
        pvals_adj = adata_log.uns['rank_genes_groups']['pvals_adj'][celltype]
        logfc = adata_log.uns['rank_genes_groups']['logfoldchanges'][celltype]

        sort_index = np.argsort(pvals_adj)
        
        gene_names = gene_names[sort_index]
        pvals_adj = pvals_adj[sort_index]
        logfc = logfc[sort_index]
        
        if genetype == 'up':
            celltype_markers = gene_names[(pvals_adj < 0.05) & (logfc > 0.25)]
        elif genetype == 'up&down':
            celltype_markers = gene_names[(pvals_adj < 0.05) & (np.abs(logfc) > 0.25)]
            
        if topk != 'all':
            celltype_markers = celltype_markers[0:topk]
    
        # print(len(celltype_markers))
        markers_list.extend(celltype_markers.tolist())
        markers[celltype] = celltype_markers.tolist()

    markers_list = list(set(markers_list))
    print('markers_before_clean:', len(markers_list))

    if if_clean:
        markers = clean_markers(markers)
        markers = markers_set_to_list(markers)
        print('markers_after_clean:', len(markers))
    else:
        markers = markers_list
    return markers

def clean_markers(markers):
    markers_clean = markers.copy()
    from functools import reduce
    for celltype in markers.keys():
        marker_set_celltype = set(markers[celltype])
        other_celltypes = set(markers.keys()) - set([celltype])
        marker_set_other_celltypes = reduce(lambda x,y: x.union(y), [set(markers[ct]) for ct in other_celltypes])
        markers_clean[celltype] = list(marker_set_celltype - marker_set_other_celltypes)
    return markers_clean

def markers_set_to_list(markers):
    markers_list = []
    for celltype in markers.keys():
        markers_list.extend(markers[celltype])
    return markers_list

def get_signature_matrix(adata_norm, markers):
    adata_proc = adata_norm[:, markers].copy()
    # to normed data
    adata_proc.X = np.exp(adata_proc.X.toarray()) - 1
    
    selections = np.isin(adata_proc.var_names, markers)
    
    clusters = adata_proc.obs['celltype2'].unique()
    sc_mean = pd.DataFrame(index=adata_proc.var_names,columns=clusters)
    for cluster in clusters:
        cells = adata_proc.obs['celltype2'] == cluster
        sc_part = adata_proc[cells,:].X.T
        sc_mean[cluster] = pd.DataFrame(np.mean(sc_part,axis=1),index=adata_proc.var_names)
    
    return sc_mean.T, selections

#%%
# load scRNA-seq data
sc_adata_0 = sc.read_h5ad("../Datasets/adata/panage_pbmc_filter.h5ad")
sc_adata = sc.AnnData(X=sc_adata_0.raw.X, obs=sc_adata_0.obs, var=sc_adata_0.raw.var)
del sc_adata_0


#%%
# find markers
# adata_log = sc_adata.copy()
# sc.tl.rank_genes_groups(adata_log, 'celltype2', method='wilcoxon')
data_path = '../Datasets/adata/'
# sc.write(os.path.join(data_path, 'adata_log.h5ad'), adata_log)
adata_log = sc.read(os.path.join(data_path, 'adata_log.h5ad'))

#%%
markers = get_markers(adata_log, topk='all', genetype='up&down', if_clean=True)

# get signature matrix and selections
signature, selections = get_signature_matrix(sc_adata, markers)
signature_info = {'signature': signature, 'selections': selections}
pickle.dump(signature_info, open(os.path.join('./model/deconv/', 'signature_info.pkl'), 'wb'))

#%% visualize the signature matrix
import seaborn as sns
a = np.log(signature + 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
a = pd.DataFrame(scaler.fit_transform(a), index=a.index, columns=a.columns) 

sns.clustermap(a.loc[adata_log.obs['celltype2'].unique(), markers], method='average', metric='euclidean', col_cluster=False, row_cluster=False, cmap='RdBu_r')

# %%
