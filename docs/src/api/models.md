# Models API

## Similarity Models

### Base Similarity Model

The base SIMBA model for learning spectrum similarity.

```{eval-rst}
.. autoclass:: simba.core.models.similarity_models.SimilarityModel
    :members:
    :undoc-members:
    :show-inheritance:
```

### Multitask Similarity Model

Extended SIMBA model with multitask learning for edit distance and MCES prediction.

```{eval-rst}
.. autoclass:: simba.core.models.similarity_models.SimilarityModelMultitask
    :members:
    :undoc-members:
    :show-inheritance:
```

### Embedding Extractor

Utility for extracting embeddings from trained models.

```{eval-rst}
.. autoclass:: simba.core.models.similarity_models.EmbeddingExtractor
    :members:
    :undoc-members:
    :show-inheritance:
```

## Spectrum Encoder

Custom spectrum transformer encoder with metadata support.

```{eval-rst}
.. autoclass:: simba.core.models.spectrum_encoder.SpectrumTransformerEncoderCustom
    :members:
    :undoc-members:
    :show-inheritance:
```

## SIMBA Model (Public API)

High-level API for running SIMBA inference.

```{eval-rst}
.. autoclass:: simba.core.models.simba_model.Simba
    :members:
    :undoc-members:
    :show-inheritance:
```
