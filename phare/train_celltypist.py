#%%
import scanpy as sc
import celltypist
import time
import numpy as np

#%%
adata_ref = sc.read_h5ad("../Datasets/adata/panage_pbmc_filter.h5ad")
var_names = adata_ref.var_names
adata_ref = sc.AnnData(
    X=adata_ref.raw.X, 
    obs = adata_ref.obs,
    var=adata_ref.raw.var,
    uns=adata_ref.uns,
    obsm=adata_ref.obsm
)
try:
    adata_ref.var_names = var_names
except:
    pass
            
#%%
log_transformed =  np.abs(np.expm1(adata_ref.X[0]).sum() - 1e4) < 1# check if the data is log transformed
print(f"Data is log transformed: {log_transformed}")
if not log_transformed:   
    sc.pp.normalize_total(adata_ref, target_sum = 1e4)
    sc.pp.log1p(adata_ref)

# %%
adata_ref.obs.celltype2.unique()

# %% 
# Downsample (Optional)
print(adata_ref.shape)
sampled_cell_index = celltypist.samples.downsample_adata(adata_ref, mode = 'total', n_cells = 100000, by = 'celltype2', return_index = True, balance_cell_type=True)
print(f"Number of downsampled cells for training: {len(sampled_cell_index)}")

# %%
# Feature selection (Suggested)
t_start = time.time()
model_fs = celltypist.train(adata_ref[sampled_cell_index], 'celltype2', n_jobs = 10, max_iter = 5, use_SGD = True, check_expression=False)

gene_index = np.argpartition(np.abs(model_fs.classifier.coef_), -300, axis = 1)[:, -300:]
gene_index = np.unique(gene_index)
print(f"Number of genes selected: {len(gene_index)}")
t_end = time.time()
print(f"Time elapsed: {t_end - t_start} seconds")

# %%
# model training and saving
t_start = time.time()
# model = celltypist.train(adata_ref[sampled_cell_index, gene_index], 'celltype2', check_expression = False, n_jobs = 10, max_iter = 100)
model = celltypist.train(adata_ref[sampled_cell_index, :], 'celltype2', check_expression = False, n_jobs = 10, max_iter = 100)
t_end = time.time()
print(f"Time elapsed: {(t_end - t_start)/60} minutes")
# Save the model.
model.write('./model/celltypist_model_from_panage_fairSample(10w).pkl')

# %%
# model training and saving wihout feature selection and downsampling
t_start = time.time()
model = celltypist.train(adata_ref, 'celltype2', check_expression = False, n_jobs = 10, max_iter = 100)
t_end = time.time()
print(f"Time elapsed: {(t_end - t_start)/60} minutes")
# Save the model.
model.write('./model/celltypist_model_from_panage.pkl')