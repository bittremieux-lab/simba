import torch
import numpy as np
from tqdm import tqdm
from src.ordinal_classification.embedder_multitask import EmbedderMultitask

class FcLayerAnalogDiscovery:
    '''
    since we are using fully connected layers for the latest models we need to apply the fc layers to the embeddings computed
    '''

    @staticmethod
    def load_full_model(model_path, config):
        return EmbedderMultitask.load_from_checkpoint(
            model_path,
            d_model=config.D_MODEL,
            n_layers=config.N_LAYERS,
            weights=None,
            n_classes=config.EDIT_DISTANCE_N_CLASSES,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            lr=config.LR,
            use_cosine_distance=config.use_cosine_distance,
            strict=False)

    @staticmethod
    def compute_all_combinations(model_path, emb0, emb1, config):
        # load full model
        model = FcLayerAnalogDiscovery.load_full_model(model_path, config)
        model.eval()
        similarities1= np.zeros((emb0.shape[0],emb1.shape[0],config.EDIT_DISTANCE_N_CLASSES)) #edit distance
        similarities2= np.zeros((emb0.shape[0],emb1.shape[0],))
        for index,emb_row in tqdm(enumerate(emb0)):
            #repeat the vector for broadcasting
            emb_tiled= np.zeros((emb1.shape[0], emb1.shape[1]))
            for n in range(0,emb_tiled.shape[0]):
                emb_tiled[n,:]= emb_row

            # compute the similarities
            sim1, sim2 = FcLayerAnalogDiscovery.compute_emb_from_existing_embeddings(model, emb_tiled, emb1)
            similarities1[index] = sim1.detach().numpy()
            similarities2[index] = sim2.detach().numpy().reshape(-1)
        return similarities1, similarities2

    @staticmethod
    def compute_emb_from_existing_embeddings(model, emb0, emb1, ):
        """
        This function computes the final `emb` and `emb_sim_2` using the already computed `emb0` and `emb1`.
        """

       

        # Convert emb0 and emb1 to PyTorch tensors if they are numpy arrays
        if isinstance(emb0, np.ndarray):
            emb0 = torch.tensor(emb0, dtype=torch.float32)
        if isinstance(emb1, np.ndarray):
            emb1 = torch.tensor(emb1, dtype=torch.float32)
            
        # Apply ReLU activation
        emb0 = model.relu(emb0)
        emb1 = model.relu(emb1)


        # for cosine similarity, tanimoto
        if model.use_cosine_distance:
            #emb_sim_2 = self.cosine_similarity(emb0, emb1)
            ## apply transformation before apply cosine distance

            # emb0
            emb0_transformed = model.linear2(emb0)
            emb0_transformed= model.dropout(emb0_transformed)
            emb0_transformed=model.relu(emb0_transformed)
            emb0_transformed = model.linear2_cossim(emb0_transformed)
            emb0_transformed=model.relu(emb0_transformed)

            # emb1
            emb1_transformed = model.linear2(emb1)
            emb1_transformed= model.dropout(emb1_transformed)
            emb1_transformed=model.relu(emb1_transformed)
            emb1_transformed = model.linear2_cossim(emb1_transformed)
            emb1_transformed=model.relu(emb1_transformed)

            # cos sim
            emb_sim_2 = model.cosine_similarity(emb0_transformed, emb1_transformed)
        else:
            emb_sim_2 = emb0 + emb1
            emb_sim_2 = model.linear2(emb_sim_2)
            emb_sim_2 = model.relu(emb_sim_2)
            emb_sim_2 = model.linear_regression(emb_sim_2)

            # avoid  values higher than 1
            x =model.relu(emb_sim_2-1)
            emb_sim_2 = emb_sim_2 - x



            
        if model.use_edit_distance_regresion:
            # emb0
            emb0_transformed_1 = model.linear1(emb0)
            emb0_transformed_1= model.dropout(emb0_transformed_1)
            emb0_transformed_1=model.relu(emb0_transformed_1)
            emb0_transformed_1 = model.linear1_cossim(emb0_transformed_1)
            emb0_transformed_1=model.relu(emb0_transformed_1)

            # emb1
            emb1_transformed_1 = model.linear1(emb1)
            emb1_transformed_1= model.dropout(emb1_transformed_1)
            emb1_transformed_1=model.relu(emb1_transformed_1)
            emb1_transformed_1 = model.linear1_cossim(emb1_transformed_1)
            emb1_transformed_1=model.relu(emb1_transformed_1)

            # cos sim
            emb = model.cosine_similarity(emb0_transformed_1, emb1_transformed_1)

            #round to integers
            emb=emb*5
            emb = emb + emb.round().detach() - emb.detach() # trick to make round differentiable
            emb=emb/5
        else:

            if not(hasattr(model, 'linear1_2')):  ## if it does not have a linea1_2 layer
                emb = emb0 + emb1
                emb = model.linear1(emb)
                emb = model.dropout(emb)
                emb = model.relu(emb)
                emb= model.classifier(emb)
            else:
                emb_0_ = model.linear1(emb0)
                emb_0_ = model.relu(emb_0_)
                emb_0_ = model.linear1_2(emb_0_)

                emb_1_ = model.linear1(emb1)
                emb_1_ = model.relu(emb_1_)
                emb_1_ = model.linear1_2(emb_1_)

                emb = emb_0_ + emb_1_
                
                #emb = self.dropout(emb)
                emb = model.relu(emb)
                emb= model.classifier(emb)


        #if self.gumbel_softmax:
        #    emb = self.gumbel_softmax(emb)
        #else:
        #    emb = F.softmax(emb, dim=-1)
        return emb, emb_sim_2