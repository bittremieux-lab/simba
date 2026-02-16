import numpy as np
import torch
from tqdm import tqdm

from simba.core.models.ordinal.embedder_multitask import EmbedderMultitask


class FcLayerAnalogDiscovery:
    """
    since we are using fully connected layers for the latest models we need to apply the fc layers to the embeddings computed
    """

    @staticmethod
    def load_full_model(model_path, config):
        d_model = config.model.transformer.d_model
        n_layers = config.model.transformer.n_layers
        n_classes = config.model.tasks.edit_distance.n_classes
        use_gumbel = config.model.tasks.edit_distance.use_gumbel
        lr = config.optimizer.lr
        use_cosine_distance = config.model.tasks.cosine_similarity.use_cosine_distance
        use_fingerprints = config.model.tasks.fingerprints.enabled

        return EmbedderMultitask.load_from_checkpoint(
            model_path,
            d_model=d_model,
            n_layers=n_layers,
            weights=None,
            n_classes=n_classes,
            use_gumbel=use_gumbel,
            lr=lr,
            use_cosine_distance=use_cosine_distance,
            strict=False,
            use_fingerprints=use_fingerprints,
        )

    @staticmethod
    def compute_all_combinations(
        model_path, emb0, emb1, config, fingerprints_0=None, fingerprint_index=1
    ):
        edit_distance_n_classes = config.model.tasks.edit_distance.n_classes

        # load full model
        model = FcLayerAnalogDiscovery.load_full_model(model_path, config)
        model.eval()
        similarities1 = np.zeros(
            (emb0.shape[0], emb1.shape[0], edit_distance_n_classes)
        )  # edit distance
        similarities2 = np.zeros(
            (
                emb0.shape[0],
                emb1.shape[0],
            )
        )
        for index, emb_row in tqdm(enumerate(emb0)):
            # repeat the vector for broadcasting
            emb_tiled = np.zeros((emb1.shape[0], emb1.shape[1]))
            for n in range(0, emb_tiled.shape[0]):
                emb_tiled[n, :] = emb_row

            # compute the similarities
            sim1, sim2 = FcLayerAnalogDiscovery.compute_emb_from_existing_embeddings(
                model,
                emb_tiled,
                emb1,
                fingerprints_0=fingerprints_0,
                fingerprint_index=fingerprint_index,
            )
            similarities1[index] = sim1.detach().cpu().numpy()
            similarities2[index] = sim2.detach().cpu().numpy().reshape(-1)
        return similarities1, similarities2

    @staticmethod
    def compute_emb_from_existing_embeddings(
        model, emb0, emb1, fingerprints_0=None, fingerprint_index=1
    ):
        # convert to tensors & apply relu/fingerprint exactly as forward() does
        model_device = next(model.parameters()).device
        emb0 = torch.tensor(emb0, dtype=torch.float32, device=model_device)
        emb1 = torch.tensor(emb1, dtype=torch.float32, device=model_device)
        emb0 = model.relu(emb0)
        emb1 = model.relu(emb1)

        if fingerprints_0 is not None:
            # same fingerprint logic as in forward…
            # fing = model.relu(model.linear_fingerprint_1(
            #            (model.relu(
            #                model.linear_fingerprint_0(torch.tensor(fingerprints_0, dtype=torch.float32))
            #             ))
            #          ))

            if fingerprint_index == 0:
                fp0 = torch.tensor(
                    fingerprints_0, dtype=torch.float32, device=model_device
                )
                fp_proj = model.relu(model.linear_fp0(fp0))  # (B, d_model//2)
                joint = torch.cat([emb0, fp_proj], dim=-1)  # (B, d_model + d_model//2)
                emb0 = model.norm_mix(model.relu(model.linear_mix(joint)))
            else:
                fp0 = torch.tensor(
                    fingerprints_0, dtype=torch.float32, device=model_device
                )
                fp_proj = model.relu(model.linear_fp0(fp0))  # (B, d_model//2)
                joint = torch.cat([emb1, fp_proj], dim=-1)  # (B, d_model + d_model//2)
                emb1 = model.norm_mix(model.relu(model.linear_mix(joint)))

        # now just delegate to your new helper:
        return model.compute_from_embeddings(emb0, emb1)

    '''
    @staticmethod
    def compute_emb_from_existing_embeddings(
        model,
        emb0,
        emb1,
        fingerprints_0=None,
    ):
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


        if fingerprints_0 is not None:
            fing_0 = torch.tensor(fingerprints_0, dtype=torch.float32)
            fing_0 = model.linear_fingerprint_0(fing_0)
            fing_0 = model.relu(fing_0)
            fing_0 = model.dropout(fing_0)
            fing_0 = model.linear_fingerprint_1(fing_0)
            fing_0 = model.relu(fing_0)
            fing_0 = model.dropout(fing_0)
            emb0 = emb0 + fing_0
            emb0 = model.relu(emb0)

        # for cosine similarity, tanimoto
        if model.use_cosine_distance:
            # emb_sim_2 = self.cosine_similarity(emb0, emb1)
            ## apply transformation before apply cosine distance

            # emb0
            emb0_transformed = model.linear2(emb0)
            emb0_transformed = model.dropout(emb0_transformed)
            emb0_transformed = model.relu(emb0_transformed)
            emb0_transformed = model.linear2_cossim(emb0_transformed)
            emb0_transformed = model.relu(emb0_transformed)

            # emb1
            emb1_transformed = model.linear2(emb1)
            emb1_transformed = model.dropout(emb1_transformed)
            emb1_transformed = model.relu(emb1_transformed)
            emb1_transformed = model.linear2_cossim(emb1_transformed)
            emb1_transformed = model.relu(emb1_transformed)

            # cos sim
            emb_sim_2 = model.cosine_similarity(emb0_transformed, emb1_transformed)
        else:
            emb_sim_2 = emb0 + emb1
            emb_sim_2 = model.linear2(emb_sim_2)
            emb_sim_2 = model.relu(emb_sim_2)
            emb_sim_2 = model.linear_regression(emb_sim_2)

            # avoid  values higher than 1
            x = model.relu(emb_sim_2 - 1)
            emb_sim_2 = emb_sim_2 - x

        if model.use_edit_distance_regresion:
            # emb0
            emb0_transformed_1 = model.linear1(emb0)
            emb0_transformed_1 = model.dropout(emb0_transformed_1)
            emb0_transformed_1 = model.relu(emb0_transformed_1)
            emb0_transformed_1 = model.linear1_cossim(emb0_transformed_1)
            emb0_transformed_1 = model.relu(emb0_transformed_1)

            # emb1
            emb1_transformed_1 = model.linear1(emb1)
            emb1_transformed_1 = model.dropout(emb1_transformed_1)
            emb1_transformed_1 = model.relu(emb1_transformed_1)
            emb1_transformed_1 = model.linear1_cossim(emb1_transformed_1)
            emb1_transformed_1 = model.relu(emb1_transformed_1)

            # cos sim
            emb = model.cosine_similarity(emb0_transformed_1, emb1_transformed_1)

            # round to integers
            emb = emb * 5
            emb = (
                emb + emb.round().detach() - emb.detach()
            )  # trick to make round differentiable
            emb = emb / 5
        else:

            # if not(hasattr(model, 'linear1_2')):  ## if it does not have a linea1_2 layer
            #    emb = emb0 + emb1
            #    emb = model.linear1(emb)
            #    emb = model.dropout(emb)
            #    emb = model.relu(emb)
            #    emb= model.classifier(emb)
            # else:
            emb_0_ = model.linear1(emb0)
            emb_0_ = model.relu(emb_0_)
            emb_0_ = model.linear1_2(emb_0_)

            emb_1_ = model.linear1(emb1)
            emb_1_ = model.relu(emb_1_)
            emb_1_ = model.linear1_2(emb_1_)

            emb = emb_0_ + emb_1_

            # emb = self.dropout(emb)
            emb = model.relu(emb)
            emb = model.classifier(emb)

        # if self.gumbel_softmax:
        #    emb = self.gumbel_softmax(emb)
        # else:
        #    emb = F.softmax(emb, dim=-1)
        return emb, emb_sim_2
    '''
