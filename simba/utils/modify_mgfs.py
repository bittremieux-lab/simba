


from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf


input_mgf = "/data/simba_files/msnlib_filtered_original.mgf" 
output_mgf = "/data/simba_files/msnlib_filtered_cleaned.mgf" 

def clean_ce(value):
    """
    Convert values like '35', '35.0', '[35]', '[35.0]' -> '35'
    """
    value = str(value).strip()

    if "[" in value:
        value = value.strip("[]")

    return str(int(float(value)))


spectra_out = []

for spectrum in load_from_mgf(input_mgf):
    metadata = spectrum.metadata.copy()

    if "collision_energy" in metadata:
        metadata["ce"] = clean_ce(metadata["collision_energy"])

    if "ion_source" in metadata:
        metadata["ionization_method"] = metadata["ion_source"]

    if "msn_fragmentation_methods" in metadata:
        metadata["ion_activation"] = metadata["msn_fragmentation_methods"]
    if "fragmentation_method" in metadata:
        metadata["ion_activation"] = metadata["fragmentation_method"]

    if "adduct" in metadata:
        metadata["adduct"] = metadata["adduct"]

    if "ionmode" in metadata:
        metadata["ionmode"] = str(metadata["ionmode"]).lower()

    spectrum.metadata = metadata
    spectra_out.append(spectrum)


save_as_mgf(spectra_out, output_mgf)
print('Writing finished succesfully')
