#%%
from phare.immAGE import *
from phare.utils import *

#%% load the test set
# test_datafiles: 
    # "Datasets/adata/chuanqi_after_filter.h5ad",
    # "Datasets/adata/chuanqi_predicted.h5ad", 
    # "Datasets/adata/disco_blood_v1.h5ad",
    # "Datasets/adata/GSE174188_CLUES1_adjusted_SLE.h5ad"

# test = load_one_adata(path="Datasets/adata/chuanqi_predicted.h5ad")
# test_ = load_one_adata(path="Datasets/adata/test_small.h5ad")
test = load_one_adata(path="../code/Datasets/adata/misc_final.h5ad")
sc.pl.umap(test, color=['orig.ident'])

# test = test_.copy()
print('test:', test.shape)
print('test.raw:', test.raw.shape)


#%% predict the age of the test set
immage = load_model(save_dir='./model/fast')
immage.fast_annotation = True
pred_test = immage.predict(test, sample_column='orig.ident')
print(pred_test["predicted_Age"])

#%%
# immage = load_model(save_dir='./model/')
# immage.fast_annotation = False
# pred_test = immage.predict(test, sample_column='orig.ident')
# print(pred_test["predicted_Age"])

#%% evaluate the prediction
test_sample_labels = immage.get_sample_labels(test, sample_column='orig.ident', target_column='Age')
compare = pd.concat([pred_test["predicted_Age"], test_sample_labels], axis=1)
compare.columns = ['Predicted age', 'Actual age']
print('Results on test set:')
print(compare)
eval_prediction(compare['Actual age'], compare['Predicted age'])
plot_pred_results(compare, band_width=10, save_path='./results/sc_prediction_misc.pdf')

#%% evaluate the prediction
rst_df = compare[['Actual age', 'Predicted age']]   
rst_df['Difference'] = rst_df['Actual age'] - rst_df['Predicted age']
rst_df['Absolute difference'] = rst_df['Difference'].abs()
rst_df['data_source'] = 'misc_final'

rst_df.to_csv('./results/sc_prediction_misc.csv')
rst_df

# %%
