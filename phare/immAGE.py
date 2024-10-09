# from immAGE import *
import warnings
warnings.filterwarnings('ignore')

import os

import warnings
from copy import deepcopy
from random import shuffle

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report as clf_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import column_or_1d

from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from .preprocessing import *
from .utils import eval_prediction
import autogenes as ag
import pickle

# lables = ['Cd', 'Tg', 'Ad', 'Ed', 'Sc']
def load_adata(paths, load_raw=True, label=None, label_names=None, celltype=None, load_celltypist=False, celltypist_dir='celltypist_ref', ctpst_column='majority_voting'):
    adatas = []
    if not isinstance(paths, list):
        paths = [paths]
        
    for path in paths:  
        celltypist_pred = None
        panage_load_failed = False
        file_name = path.split('/')[-1][0:-5] # remove '.h5ad'
        if load_celltypist:
            work_dir = '/'.join(path.split('/')[0:-1])
            try:
                celltypist_path = f'{work_dir}/{celltypist_dir}/{file_name}/predicted_labels.csv'
                celltypist_pred = pd.read_csv(celltypist_path, index_col=0)
            except Exception as e:
                # print('Exception:', e)
                assert file_name == 'panage_pbmc_filter', 'only for panage.'
                print('Load celltype2 for panage_pbmc_filter.')
                panage_load_failed = True
                
        adata = sc.read_h5ad(path)
        adata.obs['data_source'] = file_name
        
        if panage_load_failed:
            adata.obs['celltypist'] = adata.obs['celltype2']
        
        if celltypist_pred is not None:
            adata.obs['celltypist'] = celltypist_pred[ctpst_column]
            # adata.obs['celltypist'] = celltypist_pred['predicted_labels']
            # adata.obs['celltypist'] = celltypist_pred['majority_voting']
        if load_raw:
            var_names = adata.var_names
            adata = anndata.AnnData(
                X=adata.raw.X, 
                obs = adata.obs,
                var=adata.raw.var,
                uns=adata.uns,
                obsm=adata.obsm
            )
            try:
                adata.var_names = var_names
            except:
                pass
            
        if label is not None and label_names is not None:
            cells_check = list(map(lambda x: x in label_names, list(adata.obs[label].values)))
            adata = adata[cells_check, :]
        adatas.append(adata)
        
    if len(adatas) > 1:
        adata = sc.AnnData.concatenate(*adatas, join='inner')
    else:
        adata = adatas[0]
    
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    if celltype is not None:
        adata = adata[adata.obs.celltype == celltype, :]
        
    return adata

def log1p_normalize_raw(adata):
    adata_raw = sc.AnnData(X = adata.raw.X, obs = adata.obs, var = adata.raw.var)
    sc.pp.normalize_total(adata_raw, target_sum = 1e4)
    sc.pp.log1p(adata_raw)
    adata.raw = adata_raw
    return adata
    
def check_adata(adata):
    data = adata.raw.X[0]
    log_transformed =  np.abs(np.expm1(data).sum() - 1e4) < 1
    print(f"Data is log transformed: {log_transformed}")
    if not log_transformed:
        adata = log1p_normalize_raw(adata)
        
        if 'X_pca' in adata.obsm.keys():
            if adata.obsm['X_pca'].shape[1] < 50:  
                print('There is PCs but the n_comp is less than 50.')  
                print('Re-run pca...')
                if adata.shape[1] > 5000:
                    data_ = adata.X[0]
                    log_transformed_ =  np.abs(np.expm1(data_).sum() - 1e4) < 1
                    if not log_transformed_:
                        sc.pp.normalize_total(adata, target_sum = 1e4)
                        sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000)
                adata.raw = adata
                adata = adata[:, adata.var.highly_variable]
                sc.pp.scale(adata)
                sc.tl.pca(adata, n_comps=50)
    
    return adata

def load_one_adata(path):
    adata = sc.read_h5ad(path)
    adata = check_adata(adata)
    return adata
    
class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
    
class Predictor(torch.nn.Module):
    def __init__(self, input_size, output_size=None, hidden_size=64, dropout=0.3, scaler=None, model_type='Classification', output_scale=1.0):
        super(Predictor, self).__init__()
        self.model_type = model_type
        self.input_size = input_size
        if model_type == 'Classification':
            self.output_size = output_size
        elif model_type == 'Regression':
            self.output_size = 1
        self.hidden_size = hidden_size
        
        self.feature_weights = torch.nn.Parameter(torch.randn(self.input_size, ))
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu') 
            
        self.to(self.device)
        self.dropout = torch.nn.Dropout(dropout)
        
        # self.scaler = MinMaxScaler()
        if scaler=="minmax":
            self.scaler = MinMaxScaler()
        elif scaler=="standard":
            self.scaler = StandardScaler()
        elif scaler=="none":
            self.scaler = None
        else:
            raise ValueError("scaler must be one of 'minmax', 'standard', 'none'")
        self.output_scale = output_scale
        
    def forward(self, x):
        x = x * self.feature_weights
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.model_type == 'Regression':
            x = x.view(-1)
        return x * self.output_scale

    def regularizer(self, alpha=0.9):
        l1_reg = torch.sum(torch.abs(self.feature_weights))
        l2_reg = torch.sum(torch.pow(self.feature_weights, 2))
        reg = alpha * l1_reg + 0.5 * (1 - alpha) * l2_reg #elastic net
        return reg
        
    def train_model(self, X, y, epochs=10, batch_size=None, lr=1e-3, weighted=True, val_prob=0.2, patience=10, verbose=False, lamda_reg=1e-3, alpha_reg=0.9, **kwargs):
        # TODO: check if minmax-scale works.abs
        
        X, X_val, y, y_val = train_test_split(X, y, test_size=val_prob, random_state=1234)
        # X = MinMaxScaler().fit_transform(X)
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
            X_val = self.scaler.transform(X_val)
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        
        if self.model_type == 'Classification':
            if weighted:
                class_weights = torch.bincount(torch.Tensor(y).long())
                class_weights = class_weights.sum() / class_weights
                class_weights = class_weights.to(self.device)
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = torch.nn.CrossEntropyLoss()
        else:
            # criterion = torch.nn.MSELoss()
            criterion = torch.nn.SmoothL1Loss()
            # criterion = torch.nn.L1Loss()
            
        X = torch.Tensor(X).to(self.device)
        if self.model_type == 'Classification':
            y = torch.Tensor(y).long().to(self.device)
        else:
            y = torch.Tensor(y).to(self.device)
        
        if batch_size is not None and batch_size < X.shape[0]:
            train_dataset = TensorDataset(X, y)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = [(X, y)]
            
        X_val = torch.Tensor(X_val).to(self.device)
        if self.model_type == 'Classification':
            y_val = torch.Tensor(y_val).long().to(self.device)
        else:
            y_val = torch.Tensor(y_val).to(self.device)
        
        best_loss = np.inf
        best_metric = {"train_acc": 0, "val_acc": 0}
        best_model = None
        counter = 0
        early_stop = False
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            train_acc = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.forward(X_batch)
                if lamda_reg == 0:
                    loss = criterion(output, y_batch)
                else:
                    loss = criterion(output, y_batch) + lamda_reg * self.regularizer(alpha=alpha_reg)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
                if self.model_type == 'Classification':
                    train_acc += torch.sum(torch.argmax(output, dim=1) == y_batch).item()
                else:
                    # print(output, y_batch)
                    train_acc += torch.sum(torch.abs(output - y_batch)).item()
            train_loss /= X.shape[0]
            train_acc /= X.shape[0]
            
            with torch.no_grad():
                self.eval()
                output_val = self.forward(X_val)
                loss_val = criterion(output_val, y_val).item()
                if self.model_type == 'Classification':
                    val_acc = torch.sum(torch.argmax(output_val, dim=1) == y_val).item() / len(y_val)
                else:
                    val_acc = torch.sum(torch.abs(output_val - y_val)).item() / len(y_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = self.state_dict()
                counter = 0
                best_metric = {
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                }
            else:
                counter += 1
                if counter >= patience:
                    self.load_state_dict(best_model)
                    early_stop = True
                    break
                
            if verbose:
                if self.model_type == 'Classification':
                    print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {val_acc:.4f}", end='\r')
                else:
                    print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Train MAE: {train_acc:.4f} | Valid MAE: {val_acc:.4f}", end='\r')
        
        if self.model_type == 'Classification':
            print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {val_acc:.4f}")
        else:
            print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Train MAE: {train_acc:.4f} | Valid MAE: {val_acc:.4f}")
               
        if early_stop:
            print(f"Early stopping after {epoch} epochs")
        
        if self.model_type == 'Classification':
            print(f"Best model: Train Acc: {best_metric['train_acc']:.4f} | Valid Acc: {best_metric['val_acc']:.4f}")
        else:
            print(f"Best model: Train MAE: {best_metric['train_acc']:.4f} | Valid MAE: {best_metric['val_acc']:.4f}")
        
    def predict(self, X):
        # X = MinMaxScaler().fit_transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        self.eval()
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            output = self.forward(X)
            self.output = output
        
        if self.model_type == 'Classification':
            return torch.argmax(output, dim=1).cpu().numpy()
        else:
            return output.view(-1).cpu().numpy()

class SklearnEnsemblePredictor():
    def __init__(self, scaler=None, model_type='Regression', output_scale=1.0):
        
        if model_type == 'Classification':
            raise ValueError("SklearnEnsemblePredictor classifcation is not supported yet.")
        
        elif model_type == 'Regression':
        
            random_forest = RandomForestRegressor(random_state=42)
            gradient_boosting = GradientBoostingRegressor(random_state=42)
            ada_boost = AdaBoostRegressor(random_state=42)
            models = [random_forest, gradient_boosting, ada_boost]
            model_names = ['Random Forest', 'Gradient Boosting', 'AdaBoost']
            
            self.model = VotingRegressor(estimators=[(model_names[i], models[i]) for i in range(len(models))])
            
            
        if scaler=="minmax":
            self.scaler = MinMaxScaler()
        elif scaler=="standard":
            self.scaler = StandardScaler()
        elif scaler=="none":
            self.scaler = None
        else:
            raise ValueError("scaler must be one of 'minmax', 'standard', 'none'")
        self.output_scale = output_scale
        
    def train_model(self, X, y, val_prob=None, **kwargs):
        if val_prob is not None and val_prob > 0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=val_prob, random_state=1234)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
                
        self.model.fit(X, y)
        
        # validate
        if val_prob is not None and val_prob > 0:
            if self.scaler is not None:
                X_val = self.scaler.transform(X_val)
            y_pred = self.model.predict(X_val)
            eval_prediction(y_val, y_pred)
        
    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    

class ImmAge():
    def __init__(self, model_size=256, dropout=0.5, backend='sklearn', scaler='standard', add_gene_features=False, model_type='Regression', output_scale=1.0):
        '''
        backend: sklearn or torch
        '''
        self.model_size = model_size
        self.scaler_type = scaler
        self.add_gene_features = add_gene_features
        self.samples_train = None
        self.model_type = model_type
        self.output_scale = output_scale
        
        self.label_encoded = False
        self.backend = backend
        
        print(f'Backend: {self.backend}')    
        
        self.fast_annotation = False
    
    def celltype_annotation(self, adata, celltype_column):
        """by celltypist"""
        # TODO: check the if the merged adata can work with celltypist. 
        import celltypist
        # print('preprocessing data.')
        # adata = self.preprocess(adata)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件的相对路径
        
        
        if self.fast_annotation:
            predictions = celltypist.annotate(adata, 
                                              majority_voting=False,
                                              model=f'{current_dir}/model/celltypist_model_from_panage_fairSample(10w).pkl')
            adata.obs[celltype_column] = predictions.predicted_labels.predicted_labels.values
        else:
            predictions = celltypist.annotate(adata, 
                                              majority_voting=True,
                                              model=f'{current_dir}/model/celltypist_model_from_panage_fairSample(10w).pkl')
            adata.obs[celltype_column] = predictions.predicted_labels.majority_voting.values
        
        print('celltypist: annotation done.')
        return adata
    
    def preprocess(self, adata):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata
    
    def allcelltypes(self, adata, celltype_column):
        self.all_celltypes = list(set(adata.obs[celltype_column].values))
    
    def get_celltype_percentages(self, adata, sample_column='orig.ident', celltype_column='celltype'): 
        unique_samples = adata.obs[sample_column].unique() 
        celltype_percentage_df = pd.DataFrame(columns=self.all_celltypes)
        for sample in unique_samples:
            sample_adata = adata[adata.obs[sample_column] == sample, :]
            celltype_percentages_ = sample_adata.obs[celltype_column].value_counts(normalize=False)
            celltype_percentages = []
            for each in self.all_celltypes:
                if each in list(celltype_percentages_.index):
                    each_count = celltype_percentages_[each]
                    # if each_count < 100:
                    #     each_count = 0
                    celltype_percentages.append(each_count)
                else:
                    celltype_percentages.append(0)
            celltype_percentages = np.array(celltype_percentages)
            celltype_percentages = celltype_percentages / celltype_percentages.sum() # normalize
            celltype_percentages = celltype_percentages.reshape(1, -1)
            row = pd.DataFrame(celltype_percentages, columns=self.all_celltypes, index=[sample])

            celltype_percentage_df = pd.concat([celltype_percentage_df, row])
            
        return celltype_percentage_df
    
    def get_sample_labels(self, adata, sample_column='orig.ident', target_column='Group'):
        unique_samples = adata.obs[sample_column].unique()
        sample_labels = []
        for sample in unique_samples:
            sample_adata = adata[adata.obs[sample_column] == sample, :]
            label = sample_adata.obs[target_column].values[0]
            try:
                label = float(label)
            except:
                pass
            sample_labels.append(label)
        sample_labels = np.array(sample_labels).reshape(-1, 1)
        sample_labels = pd.DataFrame(sample_labels, columns=[target_column], index=unique_samples)
        
        return sample_labels
    
    def get_gene_features(self, adata, sample_column='orig.ident', corr_thr=None, n_cluster=None, n_mvg=2000, data_mode='train', **kwargs):
        # check adata is loged or not 
        # if adata.X.max() > 100:
        if 'log1p' not in adata.uns_keys():
            print('preprocessing data.')
            adata = self.preprocess(adata)
        
        if data_mode == 'train':
            sc.pp.highly_variable_genes(adata, n_top_genes=n_mvg)
            adata_ = adata[:, adata.var['highly_variable']]
            self.refgenes = adata_.var_names
            expr_mat = adata_dense_X(adata_)
            if corr_thr is None and n_cluster is None:
                expr_mat_clustered = expr_mat
            else:
                expr_mat_clustered, cluster2gene = clustring_genes(expr_mat=expr_mat, corr_thr=corr_thr, n_cluster=n_cluster)
                self.cluster2gene = cluster2gene
        elif data_mode == 'test':
            adata_ = process_test_data(adata, refgenes=self.refgenes)
            expr_mat = adata_dense_X(adata_)
            try:
                expr_mat_clustered = avg_genes_by_clusters(expr_mat, self.cluster2gene)
            except:
                expr_mat_clustered = expr_mat
        
        unique_samples = adata.obs[sample_column].unique()
        sample_exprs = []
        for sample in unique_samples:
            sample_cells_check = adata.obs[sample_column] == sample
            sample_expr = expr_mat_clustered[sample_cells_check, :].mean(axis=0)
            sample_expr /= sample_expr.sum()
            sample_exprs.append(sample_expr)
        
        sample_exprs = np.array(sample_exprs)
        try:
            columns = [f'gene_cluster_{i}' for i in self.cluster2gene.keys()]
        except:
            columns = self.refgenes
        sample_exprs = pd.DataFrame(sample_exprs, columns=columns, index=unique_samples)
        
        return sample_exprs
    
    
    def fit(self, train_adata, sample_column='orig.ident', celltype_column=None, target_column='Group', target_label_order=None, **kwargs):
        self.target_column = target_column
        if celltype_column is None:
            print('annotating celltypes...')
            celltype_column = 'anno_celltype'
            train_adata = self.celltype_annotation(train_adata, celltype_column)
            
        print('calculate celltype percentages...')
        self.allcelltypes(train_adata, celltype_column)
        # self.unique_samples = train_adata.obs[sample_column].unique()
        try:
            celltype_percentage_df = self.samples_train["celltype_percentage_df"]
            print('  use pre-calculated celltype percentages.')
        except:
            celltype_percentage_df = self.get_celltype_percentages(train_adata, sample_column, celltype_column)
        n_samples, n_celltypes = celltype_percentage_df.shape
        print(f'  find {n_samples} samples and {n_celltypes} sub-celltypes.')
        
        if self.add_gene_features:
            print('calculate clustering gene features...')
            try:
                gene_features_df = self.samples_train["gene_features_df"]
                print('  use pre-calculated gene features.')
            except: 
                gene_features_df = self.get_gene_features(train_adata, sample_column, data_mode='train', **kwargs)
            n_gene_cluster = gene_features_df.shape[1]
            print(f'  find {n_gene_cluster} gene clusters.')
            feature_df = pd.concat([celltype_percentage_df, gene_features_df], axis=1)
            feature_matrix = feature_df.values
        else:
            feature_df = celltype_percentage_df
            
            feature_matrix = feature_df.values
        
        print(f'train {self.model_type} predictor...')
        try:
            sample_labels = self.samples_train["labels"]
        except:
            sample_labels = self.get_sample_labels(train_adata, sample_column, target_column)
            sample_labels = sample_labels.loc[celltype_percentage_df.index, target_column].values
        
        if isinstance(sample_labels[0], str):
            self.label_encoded = True
            self.label_encoder = MyLabelEncoder()
            if target_label_order is not None:
                unique_targets = target_label_order
            else:
                unique_targets = train_adata.obs[target_column].unique()
            self.label_encoder.fit(unique_targets)
            numeric_sample_labels = self.label_encoder.transform(sample_labels)
        else:
            numeric_sample_labels = sample_labels
            
        input_dim = feature_matrix.shape[1]
        try:
            output_dim = len(unique_targets)
        except:
            output_dim = None

        if self.backend == 'torch':
            self.predictor = Predictor(input_size=input_dim, 
                                        output_size=output_dim, 
                                        hidden_size=self.model_size, 
                                        dropout=0.3, 
                                        scaler=self.scaler_type, 
                                        model_type=self.model_type, 
                                        output_scale=self.output_scale)
        elif self.backend == 'sklearn':
            self.predictor = SklearnEnsemblePredictor(scaler=self.scaler_type, 
                                                      model_type=self.model_type,
                                                      output_scale=self.output_scale)
            
        self.predictor.train_model(feature_matrix, numeric_sample_labels, **kwargs)
        
        if self.add_gene_features:
            self.samples_train = {
                "celltype_percentage_df": celltype_percentage_df.copy(),
                "gene_features_df": gene_features_df.copy(),
                "labels": sample_labels
            }
        else:
            self.samples_train = {
                "celltype_percentage_df": celltype_percentage_df.copy(),
                "labels": sample_labels
            }
        self.save_sample_features()
        print('done.')
        
    def save_sample_features(self, save_dir='./tmp', mtype='train'):
        try:
            os.mkdir(save_dir)
        except:
            pass
        
        if mtype == 'train':
            if not self.add_gene_features:
                df = self.samples_train['celltype_percentage_df'].copy()
                df['Age'] = self.samples_train['labels']
                df.to_csv('./tmp/cta.csv', index=True)
            else:
                df = self.samples_train['celltype_percentage_df'].copy()
                df_gene = self.samples_train['gene_features_df'].copy()
                df = pd.concat([df, df_gene], axis=1)
                df['Age'] = self.samples_train['labels']
                df.to_csv('./tmp/cta_with_gene.csv', index=True)
        elif mtype == 'test':
            pass
        
        
    
    def predict(self, test_adata, sample_column='orig.ident', celltype_column=None):
        if celltype_column is None:
            print('annotating celltypes...')
            celltype_column = 'anno_celltype'
            test_adata = self.celltype_annotation(test_adata, celltype_column)
        
        print('calculate celltype percentages...')
        celltype_percentage_df = self.get_celltype_percentages(test_adata, sample_column, celltype_column)
        n_samples, n_celltypes = celltype_percentage_df.shape
        print(f'  find {n_samples} samples and {n_celltypes} sub-celltypes.')
        
        if self.add_gene_features:
            print('calculate clustering gene features...')
            gene_features_df = self.get_gene_features(test_adata, sample_column, data_mode='test')
            n_gene_cluster = gene_features_df.shape[1]
            print(f'  find {n_gene_cluster} gene clusters.')
            feature_matrix = pd.concat([celltype_percentage_df, gene_features_df], axis=1).values
        else:
            feature_matrix = celltype_percentage_df.values
        
        if self.add_gene_features:
            self.samples_test = {
                "celltype_percentage_df": celltype_percentage_df.copy(),
                "gene_features_df": gene_features_df.copy(),
            }
        else:
            self.samples_test = {
                "celltype_percentage_df": celltype_percentage_df.copy(),
            }
            
        print('predicting...')
        pred = self.predictor.predict(feature_matrix)
        if self.label_encoded and self.model_type == 'classification':
            pred_labels = self.label_encoder.inverse_transform(pred)
        else:
            pred_labels = pred
        
        rst = celltype_percentage_df.copy()
        rst[f'predicted_{self.target_column}'] = pred_labels
        print('done.')
        return rst
    
    def deconvolute_bulk(self, bulk_data, deconv_method='nnls'):
        '''
        deconv_method: `nusvr`, `nnls`, `linear`
        '''
        signature_info = pickle.load(open(os.path.join('./model/deconv', 'signature_info.pkl'), 'rb'))
        signature_info['selections']

        ag.init(signature_info['signature'])
        ag.main._selection = signature_info['selections']

        bulk_data = pd.DataFrame(data=bulk_data.values, index=list(bulk_data.index), columns=bulk_data.columns)
        bulk_data = bulk_data / bulk_data.sum(axis=0).values.reshape(1, -1) * 1e4
        # 
        coef = ag.deconvolve(bulk_data.T, model=deconv_method)  #`nusvr`, `nnls`, `linear`
        # print(coef)
    
        coef[coef<0] = 0
        ## normalize the coef
        # coef = (coef - coef.min()) / (coef.max() - coef.min())
        coef = coef / coef.sum(axis=1, keepdims=True)
    
        # coef = softmax(coef, axis=1)
        celltype_fraction = pd.DataFrame(data=coef, index=list(bulk_data.columns), columns=signature_info['signature'].index)
        return celltype_fraction
    
    def predict_bulk(self, bulk_data, deconv_method='nnls'):
        '''
        deconv_method: `nusvr`, `nnls`, `linear`
        '''
        # bulk deconvolution
        print('deconvoluting bulk data...')
        celltype_fraction = self.deconvolute_bulk(bulk_data, deconv_method=deconv_method)
        # bulk prediction
        
        feature_df = celltype_fraction.loc[:, self.all_celltypes]
        feature_matrix = feature_df.values
        
        print('predicting...')
        pred = self.predictor.predict(feature_matrix)
        if self.label_encoded and self.model_type == 'classification':
            pred_labels = self.label_encoder.inverse_transform(pred)
        else:
            pred_labels = pred
        
        rst = celltype_fraction.copy()
        rst[f'predicted_{self.target_column}'] = pred_labels
        print('done.')
        return rst
    
