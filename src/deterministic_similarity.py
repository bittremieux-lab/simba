from src.molecule_pair import MoleculePair
from typing import List
from src.similarity import cosine, modified_cosine, neutral_loss
from src.config import Config
from tqdm import tqdm
import pandas as pd
from src.tanimoto import Tanimoto
from src.transformers.load_data_unique import LoadDataUnique
from torch.utils.data import DataLoader
from src.transformers.embedder import Embedder
import lightning.pytorch as pl
import numpy as np
from src.config import Config
from scipy.stats import spearmanr
from src.plotting import Plotting
import copy

# from src.ml_model import MlModel


class DetSimilarity:
    """
    class for computing similarity for cosine distance
    """

    @staticmethod
    def compute_deterministic_similarity(
        molecule_pairs: List[MoleculePair], similarity_metric="cosine", config=None
    ):
        """
        compute cosine ('cosine'), modified cosine ('modified_cosine') or neutral loss ('neutral_loss')
        """
        computing_function = DetSimilarity.select_function(similarity_metric)
        total_scores = []
        for m in tqdm(molecule_pairs):
            spectra_0 = m.spectrum_object_0
            spectra_1 = m.spectrum_object_1
            scores = computing_function(
                spectra_0, spectra_1, config.FRAGMENT_MZ_TOLERANCE
            )
            # set score
            m.set_det_similarity_score(scores, similarity_metric)
            total_scores.append(scores)
        return molecule_pairs, total_scores

    @staticmethod
    def select_function(similarity_metric):
        if similarity_metric == "cosine":
            return cosine
        elif similarity_metric == "modified_cosine":
            return modified_cosine
        elif similarity_metric == "neutral_loss":
            return neutral_loss

    @staticmethod
    def call_saved_model(molecule_pairs, model_file):
        # siamese
        model = MlModel(input_dim=molecule_pairs[0].vector_0.shape[0])
        model.load_best_model(model_file)
        return model.predict(molecule_pairs)

    @staticmethod
    def call_saved_transformer_model(
        molecule_pairs,
        model_file,
        d_model=64,
        n_layers=2,
        batch_size=128,
        use_cosine_distance=True,
    ):
        # transformer
        dataset_test = LoadDataUnique.from_molecule_pairs_to_dataset(molecule_pairs)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        best_model = Embedder.load_from_checkpoint(
            model_file,
            d_model=d_model,
            n_layers=n_layers,
            use_cosine_distance=use_cosine_distance,
        )
        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
        pred_test = trainer.predict(best_model, dataloader_test)

        # flat the tensor
        flat_pred_test = []
        for pred in pred_test:
            flat_pred_test = flat_pred_test + [float(p) for p in pred]

        flat_pred_test = np.array(flat_pred_test)

        # pred_test = np.array([float(p[0]) for p in pred_test])

        # clip to 0 and 1
        pred_test = np.clip(flat_pred_test, 0, 1)
        return pred_test

    @staticmethod
    def preprocessing_for_deterministic_metrics(
        spectrum,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        scale_intensity="root",
    ):
        spectrum_copy = copy.deepcopy(spectrum)
        return (
            spectrum_copy.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
            .filter_intensity(min_intensity=min_intensity, max_num_peaks=max_num_peaks)
            .set_mz_range(min_mz=10, max_mz=1400)
            #.scale_intensity(scale_intensity)
        )

    @staticmethod
    def compute_all_scores(
        molecule_pairs,
        write=False,
        write_file='"./gnps_libraries.parquet"',
        model_file="./best_model.h5",
        config=None,
    ):
        scores = []
        # model_scores = DetSimilarity.call_saved_model(molecule_pairs, model_file)
        model_scores = DetSimilarity.call_saved_transformer_model(
            molecule_pairs,
            model_file,
            d_model=config.D_MODEL,
            n_layers=config.N_LAYERS,
            use_cosine_distance=config.use_cosine_distance,
        )

        for i, m in tqdm(enumerate(molecule_pairs)):

            # get the spectra
            spectra_0 = m.spectrum_object_0
            spectra_1 = m.spectrum_object_1

            # apply specific preprocessing for spectra going into the deterministic metrics
            spectra_0_det = DetSimilarity.preprocessing_for_deterministic_metrics(spectra_0)
            spectra_1_det = DetSimilarity.preprocessing_for_deterministic_metrics(spectra_1)

            # compute deterministic similarities
            cos = cosine(spectra_0_det, spectra_1_det, config.FRAGMENT_MZ_TOLERANCE)
            mod_cos = modified_cosine(
                spectra_0_det, spectra_1_det, config.FRAGMENT_MZ_TOLERANCE
            )
            # nl = neutral_loss(
            #    spectra_0, spectra_1, config.FRAGMENT_MZ_TOLERANCE
            # )
            # TODO: There is a bug with neutral loss that makes it produce a division zero, possibly because of the computation
            nl = cos

            # model_score= model_scores[i,0]   #for sieamese network
            model_score = model_scores[i]

            fp1 = Tanimoto.compute_fingerprint(spectra_0_det.smiles)
            fp2 = Tanimoto.compute_fingerprint(spectra_1_det.smiles)
            tan = Tanimoto.compute_tanimoto(fp1, fp2)
            scores.append(
                (
                    cos[0],
                    cos[1],
                    mod_cos[0],
                    mod_cos[1],
                    nl[0],
                    nl[1],
                    model_score,
                    0,
                    tan,
                )
            )

        # Compute the spearman correlation
        # Calculate Spearman correlation
        tanimoto_temp = np.array([s[8] for s in scores])
        mod_cosine_temp = np.array([s[2] for s in scores])
        model_temp = np.array([s[6] for s in scores])

        corr_mod_cos, p_value_mod_cos = spearmanr(tanimoto_temp, mod_cosine_temp)
        corr_model_temp, p_value_model_temp = spearmanr(tanimoto_temp, model_temp)

        # Print the correlation coefficient and p-value
        print("Spearman correlation coefficient for modified cosine:", corr_mod_cos)
        print("P-value:", p_value_mod_cos)
        print("Spearman correlation coefficient for model:", corr_model_temp)
        print("P-value:", p_value_model_temp)

        # roc curves
        x_class = tanimoto_temp.copy()
        x_class[tanimoto_temp < config.threshold_class] = 0
        x_class[tanimoto_temp >= config.threshold_class] = 1

        Plotting.plot_roc_curve_comparison(
            x_class,
            #[model_temp, mod_cosine_temp],
            [model_temp,],
            title="ROC Curve",
            roc_file_path=config.CHECKPOINT_DIR
            + f"roc_curve_comparison_{config.MODEL_CODE}.png",
            #labels=["model", "mod_cosine"],
            labels=["model"],
            #colors=["r", "b"],
            colors=["r",],
        )

