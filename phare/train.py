#%%
from .immAGE import *
from .utils import *

#%%
train_datafiles = [
    "../Datasets/adata/panage_pbmc_filter.h5ad",
    "../Datasets/adata/GSE161918_healthy.h5ad",
    "../Datasets/adata/misc_SeuratObj_fiilter_hc.h5ad",
    "../Datasets/adata/GSE157007_pbmc.h5ad",
    "../Datasets/adata/GSE174188_CLUES1_adjusted_Healthy.h5ad",
    "../Datasets/adata/GSE213516_allcell_HVG.h5ad"
]

print(train_datafiles)

train = load_adata(train_datafiles, load_celltypist=True, celltypist_dir='celltypist_by_ref')
try:
    sc.pl.umap(train, color=['Group', 'celltype'])
except:
    pass
print('train:', train.shape)

# remove the cells with no age information
train.obs.Age = train.obs.Age.astype('float')
a = train.obs.Age
print(train[a.isnull()].obs['orig.ident'].value_counts())
train = train[a.notnull()]
print('train:', train.shape)

#%%
immage = ImmAge(add_gene_features=False)
immage.fit(train, 
sample_column='orig.ident', 
celltype_column='celltypist', 
target_column='Age'
)

