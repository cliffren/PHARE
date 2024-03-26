#%%
from immAGE import *
from utils import *

#%% load the test bulk tpm data
# data_names = ["GSE166253", 
#               "GSE192829", 
#               "GSE198533",
#               "GSE180081"]

data_names = ["GSE166253", 
            "GSE192829"]

# data_names = ["GSE180081"]

compare_all = pd.DataFrame()   
for data_name in data_names:
    # data_name = "GSE192829"
    bulk_test =  pd.read_csv(os.path.join('./Datasets/', 'bulk', f'{data_name}_tpm.csv'), index_col=0)  
    print('bulk_test:', bulk_test.shape)

    ## %% predict the age of the test set
    immage = load_model(save_dir='./model/fast')
    immage.fast_annotation = True
    pred_test = immage.predict_bulk(bulk_test, deconv_method='nnls')
    # print(pred_test["predicted_Age"])

    ##%% evaluate the prediction
    meta_test = pd.read_csv(f'./Datasets/bulk/{data_name}_clinical.csv', index_col=0)
    meta_test = meta_test.loc[bulk_test.columns, :]
    test_sample_labels = meta_test['age']

    compare = pd.concat([pred_test["predicted_Age"], test_sample_labels], axis=1)
    compare.columns = ['Predicted age', 'Actual age']
    
    compare['data_source'] = data_name
    celltype_percentage = pred_test.drop(columns=['predicted_Age'])
    compare = pd.concat([compare, celltype_percentage], axis=1)
    
    # print('Results on test set:')
    # print(compare)
    
    compare_all = pd.concat([compare_all, compare], axis=0)
    
print('Results on test set:')
print(compare_all[['Predicted age', 'Actual age']]) 
eval_prediction(compare_all['Actual age'], compare_all['Predicted age'])
# plot_pred_results(compare_all, band_width=10, save_path='./results/bulk_prediction_gse180081.pdf')
# compare_all.to_csv('./results/bulk_prediction_gse180081.csv')

#%%
def plot_pred_results_(voting_results, band_width=10, save_path=None, show=True):
    plt.figure(figsize=(5, 5))
    try:
        sns.scatterplot(x='Actual age', y='Predicted age', data=voting_results, hue='data_source', palette='Set2')
    except:
        sns.scatterplot(x='Actual age', y='Predicted age', data=voting_results)
    
    voting_results_ = voting_results[['Actual age', 'Predicted age']]
    plt.plot(
        [voting_results_.min().min(), 
         voting_results_.max().max()], 
        [voting_results_.min().min(), 
         voting_results_.max().max()], 
        'k--', lw=2)
    x_min, y_min = voting_results_.min().min(), voting_results_.min().min()
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_min = x_min - 3
    y_min = y_min - 3
    
    x_max, y_max = voting_results_.max().max()+3, voting_results_.max().max()+3
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.fill_between(
        [voting_results_.min().min(), 
         voting_results_.max().max()], 
        [voting_results_.min().min()-band_width, 
         voting_results_.max().max()-band_width],                    
        [voting_results_.min().min()+band_width,
         voting_results_.max().max()+band_width], 
        alpha=0.2)
    plt.title('Actual vs Predicted Values')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontsize(7)
    
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
plot_pred_results_(compare_all, band_width=10, save_path='./results/bulk_prediction_.pdf')
compare_all.to_csv('./results/bulk_prediction_.csv')
# %%
