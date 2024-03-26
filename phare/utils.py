import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os 
import torch
from sklearn.metrics import mean_absolute_error
import pandas as pd 

def plot_predictions(y_true, y_pred, xlabel='Group', band_width=10):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, y_pred, 'o')
    x_min, x_max = plt.xlim()
    plt.plot([x_min, x_max], [x_min, x_max], 'k--')
    plt.xlabel(f'True {xlabel}')
    plt.ylabel(f'Predicted {xlabel}')
    plt.xlim(x_min, None)
    plt.ylim(y_pred.min(), None)
    plt.fill_between([y_pred.min(), y_pred.max()], [y_pred.min()-band_width, y_pred.max()-band_width], [y_pred.min()+band_width, y_pred.max()+band_width], alpha=0.2)
    
    plt.show()
    
def find_sample_sources(adata, sample_list):
    sources = []
    for each_sample in sample_list:
        each_source =  adata.obs.loc[adata.obs['orig.ident'] == each_sample, 'data_source'][0]
        sources.append(each_source)
    return sources
        
    
def plot_pred_results(voting_results, band_width=10, save_path=None, show=True):
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
    x_min, y_min = voting_results_.min().min()-5, voting_results_.min().min()-5
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    
    plt.xlim(x_min, None)
    plt.ylim(y_min, None)

    plt.fill_between(
        [voting_results_.min().min(), 
         voting_results_.max().max()], 
        [voting_results_.min().min()-band_width, 
         voting_results_.max().max()-band_width],                    
        [voting_results_.min().min()+band_width,
         voting_results_.max().max()+band_width], 
        alpha=0.2)
    plt.title('Actual vs Predicted Values')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def eval_prediction(y_true, y_pred, band_width=10, index=None):
    mae = mean_absolute_error(y_true, y_pred)
        
    voting_results = pd.DataFrame({'Actual age': y_true, 'Predicted age': y_pred})
    if index is not None:
        voting_results.index = index
        
    voting_results['Difference'] = voting_results['Actual age'] - voting_results['Predicted age']
    voting_results['Absolute difference'] = voting_results['Difference'].abs()
        
    acc_within_bw = voting_results['Absolute difference'].apply(lambda x: 1 if x <= band_width else 0).mean()
        
    pearson_corr = voting_results['Actual age'].corr(voting_results['Predicted age'], method='pearson')
        
    print(f'MAE = {mae:.2f}, Accuracy within {band_width} years = {acc_within_bw:.2f}, Pearson correlation = {pearson_corr:.2f}')
        
    rst = {
        'mae': mae, 
        'acc': acc_within_bw, 
        'pearson': pearson_corr
    }
        
    return rst
    
def save_model(immage, save_dir='./model'):
    try:
        os.mkdir(save_dir)
    except:
        pass
    path=f'{save_dir}/immage.pkl'
    with open(path, 'wb') as f:
        pickle.dump(immage, f)
            
    if immage.backend == 'torch':
        weight_path=f'{save_dir}/predictor_torch.pt'
        torch.save(immage.predictor.state_dict(), weight_path)
    elif immage.backend == 'sklearn':
        model_path = f'{save_dir}/predictor_sklearn.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(immage.predictor, f)
            
    print(f'save model to {save_dir}')
        
def load_model(save_dir='./model'):
    path=f'{save_dir}/immage.pkl'
    with open(path, 'rb') as f:
        immage = pickle.load(f)
        
    if immage.backend == 'torch':    
        weight_path=f'{save_dir}/predictor_torch.pt'    
        try:
            immage.predictor.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        except:
            print('cannot load weights, please train the model.')
                
        if torch.cuda.is_available():
            immage.predictor.device = 'cuda'  
        else:
            immage.predictor.device = 'cpu'
        immage.predictor.to(immage.predictor.device)
    elif immage.backend == 'sklearn':
        model_path = f'{save_dir}/predictor_sklearn.pkl'
        with open(model_path, 'rb') as f:
            immage.predictor = pickle.load(f)
            
    return immage
    