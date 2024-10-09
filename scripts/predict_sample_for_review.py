#%%
from phare import *

#%% load the test set
test = sc.read_h5ad("./Datasets/review/all_pbmcs/all_pbmcs_rna.h5ad")
test
meta = pd.read_csv("./Datasets/review/all_pbmcs/all_pbmcs_metadata.csv", index_col=0)
test.obs = test.obs.join(meta)
print('test:', test.shape)

#%%
# sc.pp.normalize_total(test, target_sum=1e4)
# sc.pp.log1p(test)
# sc.pl.umap(test, color=['orig.ident'])
# print('test:', test.shape)
# print('test.raw:', test.raw.shape)

#%% predict the age of the test set
immage = load_model(save_dir='./phare/model/fast')
immage.fast_annotation = True

donors = test.obs['Donor_id'].unique()
donors_list_split = np.array_split(donors, 50)
pred_test = pd.DataFrame()
for i, donor_list in enumerate(donors_list_split):
    print('Prdicting for the %dth batch...' % i)
    pred_test_sub = immage.predict(
        test[test.obs['Donor_id'].isin(donor_list)], 
        sample_column='Donor_id'
        )
    pred_test = pd.concat([pred_test, pred_test_sub])

# pred_test = immage.predict(test, sample_column='Donor_id')
print(pred_test["predicted_Age"])

#%%
# immage = load_model(save_dir='./model/')
# immage.fast_annotation = False
# pred_test = immage.predict(test, sample_column='orig.ident')
# print(pred_test["predicted_Age"])

#%% evaluate the prediction
test_sample_labels = immage.get_sample_labels(test, sample_column='Donor_id', target_column='Age')
compare = pd.concat([pred_test["predicted_Age"], test_sample_labels], axis=1)
compare.columns = ['Predicted age', 'Actual age']
print('Results on test set:')
print(compare)
eval_prediction(compare['Actual age'], compare['Predicted age'])

plt.figure(figsize=(5, 5))
plot_pred_results(compare, band_width=10, save_path='./Datasets/review/all_pbmcs/results/sc_prediction.pdf')

#%% evaluate the prediction
rst_df = compare[['Actual age', 'Predicted age']]   
rst_df['Difference'] = rst_df['Actual age'] - rst_df['Predicted age']
rst_df['Absolute difference'] = rst_df['Difference'].abs()
# rst_df['data_source'] = 'misc_final'

rst_df.to_csv('./Datasets/review/all_pbmcs/results/sc_prediction.csv')
rst_df

# %%


