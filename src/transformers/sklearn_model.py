import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import shap
from src.transformers.embedder import Embedder
from src.config import Config
import dill
from src.train_utils import TrainUtils
from src.transformers.load_data_unique import LoadDataUnique
from src.molecular_pairs_set import MolecularPairsSet
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from src.transformers.postprocessing import Postprocessing
import numpy as np
import pandas as pd
from src.transformers.CustomDatasetUnique import CustomDatasetUnique
from src.transformers.CustomDatasetMultitasking import CustomDatasetMultitasking
from src.ordinal_classification.load_data_multitasking import LoadDataMultitasking
import lightning.pytorch as pl

class SklearnModel(BaseEstimator, ClassifierMixin):
    """
    wrapper for using shap values
    """

    def __init__(self, model_path=None, d_model=None, n_layers=None, model_loaded=None, multitasking=False, max_peaks=None):

        if model_loaded is None:
            self.model_path = model_path
            self.d_model = d_model
            self.n_layers = n_layers
            self.pytorch_object = Embedder.load_from_checkpoint(
                model_path, d_model=d_model, n_layers=n_layers
            )
        else:
            self.pytorch_object =model_loaded 

        self.model = pl.Trainer(max_epochs=0, enable_progress_bar=True)
        self.size_per_key = {}  # size of each key of the data loaded
        self.dataset_test = None
        self.dataloader_test = None
        self.explainer = None
        self.multitasking=multitasking
        self.max_peaks=max_peaks
    def fit(self, X, y):
        return self



    def predict(self, X):

        # Convert numpy array to PyTorch tensor
        #item = self.x_to_item(X.values, self.size_per_key)
        
        ### put X directly to model. wrapper for converting it into something that the model accepts
        
        item = self.x_to_item(X, self.size_per_key)

        print(item)
        pred_test = self.model.predict(
            self.pytorch_object,
            item,
        )

        # flat the results
        if self.multitasking:
            predictions = []
            for pred in pred_test: # in the batch dimension
                #get the results of each similarity
                pred1= pred[0]
                pred2= pred[1]
                predictions = predictions + [p.item() for p in pred2]

            predictions = np.array(predictions)
        else:
            flat_pred_test = []
            for pred in pred_test:
                flat_pred_test = flat_pred_test + [float(p) for p in pred]

            predictions = np.array([float(p) for p in flat_pred_test])
        return predictions

    def load_size_per_key(self, item):
        """
        load the size of each key in the data loaded. for instance mz is normally 100 values
        """
        for k in item.keys():
            self.size_per_key[k] = item[k].shape[1]
        return self.size_per_key

    def get_columns(self, item):
        cols = []
        for k in item.keys():
            number_new_columns = item[k].shape[1]
            cols = cols + [(k + "_" + str(i)) for i in range(0, number_new_columns)]
        return cols

    def item_to_x(self, item):
        """
        from dict to X  shap format
        """
        # create np array
        first_key = list(item.keys())[0]
        number_samples = item[first_key].shape[0]
        number_features = 0

        for k in item.keys():
            number_features = number_features + item[k].shape[1]
        X = np.zeros((number_samples, number_features), dtype=np.float32)

        for n in range(0, number_samples):
            # fill X
            index_temp = 0
            for k in item.keys():
                last_index = index_temp + item[k].shape[1]
                X[n, index_temp:last_index] = np.array(item[k][n], dtype=np.float32)
                # update index
                index_temp = last_index

        cols = self.get_columns(item)

        X = pd.DataFrame(data=X.astype(np.float32), columns=cols)
        return X

    def x_to_item(self, x, size_per_key):
        """
        from X  shap format to dict
        """
        index_temp = 0
        new_item = {}

        for k in size_per_key:
            new_item[k] = np.zeros((x.shape[0], size_per_key[k]), dtype=np.float32)

        print(f'Number of samples processed: {x.shape[0]}')
        for n in range(x.shape[0]):
            index_temp = 0

            #for k in x.keys():
            for k in size_per_key:
                size = size_per_key[k]

                if k != 'similarity':
                    valid_columns= [column for column in x.keys() if column.startswith(k)]
                else:
                    valid_columns= [column for column in x.keys() if (column.startswith('similarity') and (not(column.startswith('similarity2'))))]
                data_collected = x.loc[n, valid_columns]
                new_item[k][n] = np.array(data_collected, dtype=np.float32)
                #print(x.values[n, index_temp : index_temp + size])
                #try: ## if it is a df

                #    new_item[k][n] = np.array(
                #        [x.values[n, index_temp : index_temp + size]], dtype=np.float32
                #    )
                #except: 
                #    new_item[k][n] = np.array(
                #        [x[n, index_temp : index_temp + size]], dtype=np.float32
                #    )
                #index_temp = index_temp + size
        return new_item

    def get_X_from_all_molecule_pairs(self, all_molecule_pairs):

        if self.multitasking:
            dataset_test= LoadDataMultitasking.from_molecule_pairs_to_dataset(all_molecule_pairs, 
                                                max_num_peaks=self.max_peaks)
        else:
            dataset_test = LoadDataUnique.from_molecule_pairs_to_dataset(all_molecule_pairs)
        # dictionary = dataset_test.data
        #dictionary = dataset_test.get_original_dictionary(max_num_peaks=self.max_peaks)
        #dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        ##get item
        #dataiter = iter(dataloader_test)
        #item = next(dataiter)
        
        
        items= dataset_test.get_original_dictionary(max_num_peaks=self.max_peaks)

        print('original dictionary:')
        print(items)
        return self.item_to_x(items)

    def get_explainer(self, all_molecule_pairs):
        
        X_total = self.get_X_from_all_molecule_pairs(all_molecule_pairs)



        # get example of one item
        if self.multitasking:
            
            dataset_test= LoadDataMultitasking.from_molecule_pairs_to_dataset(all_molecule_pairs, 
                                                max_num_peaks=self.max_peaks)
        else:
            dataset_test = LoadDataUnique.from_molecule_pairs_to_dataset(all_molecule_pairs)

        # save the data inside the object
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        ##get item
        dataiter = iter(self.dataloader_test)
        item = next(dataiter)

        # load size of each key:

        self.size_per_key = self.load_size_per_key(item)


        explainer = shap.Explainer(
            self.predict,
            X_total,
         )

        #explainer = shap.KernelExplainer(
        #    self.predict,
        #    X_total,
        # )

        return explainer, X_total




    def compute_shap_values(self, explainer, one_molecule_pair):

        X = self.get_X_from_all_molecule_pairs(one_molecule_pair)

        #update the dataset loader:
         # get example of one item
        if self.multitasking:
            
            dataset_test= LoadDataMultitasking.from_molecule_pairs_to_dataset(one_molecule_pair, 
                                                max_num_peaks=self.max_peaks)
        else:
            dataset_test = LoadDataUnique.from_molecule_pairs_to_dataset(all_molecule_pairs)

        # save the data inside the object
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        # Compute SHAP values
        #return X
        return explainer.shap_values(X), X
