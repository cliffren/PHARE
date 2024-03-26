#%%
from phare.immAGE import *
from phare.utils import *

#%% load the test bulk tpm data
# data_names = ["GSE166253", 
#               "GSE192829", 
#               "GSE198533"]

data_name = "GSE192829"
bulk_test =  pd.read_csv(os.path.join('./Datasets/', 'bulk', f'{data_name}_tpm.csv'), index_col=0)  
print('bulk_test:', bulk_test.shape)


#%% predict the age of the test set
immage = load_model(save_dir='./phare/model/fast')
immage.fast_annotation = True
pred_test = immage.predict_bulk(bulk_test, deconv_method='nusvr')
print(pred_test["predicted_Age"])

#%% evaluate the prediction
meta_test = pd.read_csv(f'./Datasets/bulk/{data_name}_clinical.csv', index_col=0)
meta_test = meta_test.loc[bulk_test.columns, :]
test_sample_labels = meta_test['age']

compare = pd.concat([pred_test["predicted_Age"], test_sample_labels], axis=1)
compare.columns = ['Predicted age', 'Actual age']
print('Results on test set:')
print(compare)
eval_prediction(compare['Actual age'], compare['Predicted age'])
plot_pred_results(compare, band_width=10)

