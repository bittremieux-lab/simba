from depthcharge.data import SpectrumDataset
from collections.abc import Generator, Iterable
from os import PathLike
import lance
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import uuid
from collections.abc import Generator, Iterable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lance
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import depthcharge
from depthcharge.data import arrow
from depthcharge import utils
from depthcharge.data import preprocessing


def _get_records_temp(
    data: list[pl.DataFrame | PathLike], **kwargs: dict
) -> Generator[pa.RecordBatch]:
    """Yields RecordBatches for data.

    Parameters
    ----------
    data : list of polars.DataFrame or PathLike
        The data to add.
    **kwargs : dict
        Keyword arguments for the parser.
    """
    for spectra in data:
        try:

            spectra = spectra.lazy().collect().to_arrow().to_batches()
        except AttributeError:
            try:
                spectra = pq.ParquetFile(spectra).iter_batches()
            except (pa.ArrowInvalid, TypeError):
                spectra = arrow.spectra_to_stream(
                    spectra,
                    preprocessing_fn=[
                        preprocessing.set_mz_range(min_mz=140),
                        preprocessing.filter_intensity(max_num_peaks=200),
                        preprocessing.scale_intensity(scaling="root"),
                        preprocessing.scale_to_unit_norm,
                    ],
                    **kwargs,
                )

        for batch in spectra:
            yield batch


class SpectrumDatasetSiamese(SpectrumDataset):

    def __init__(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        path: PathLike | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize a SpectrumDataset."""
        self._tmpdir = None
        if path is None:
            # Create a random temporary file:
            self._tmpdir = TemporaryDirectory()
            path = Path(self._tmpdir.name) / f"{uuid.uuid4()}.lance"

        self._path = Path(path)
        if self._path.suffix != "lance":
            self._path = path.with_suffix(".lance")

        # Now parse spectra.
        if spectra is not None:
            spectra = utils.listify(spectra)
            batch = next(_get_records_temp(spectra, **kwargs))
            lance.write_dataset(
                _get_records_temp(spectra, **kwargs),
                self._path,
                mode="overwrite",
                schema=batch.schema,
            )

        elif not self._path.exists():
            raise ValueError("No spectra were provided")

        self._dataset = lance.dataset(self._path)

        self._dataset = lance.dataset(self._path)
