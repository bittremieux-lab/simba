
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup

class Plotting:

    @staticmethod
    def plot_spectrum(spectrum):
        # Plot the spectrum.
        fig, ax = plt.subplots(figsize=(12, 6))
        sup.spectrum(spectrum, grid=False, ax=ax)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        #plt.savefig("quickstart.png", bbox_inches="tight", dpi=300, transparent=True)

    @staticmethod
    def plot_molecule_pair_spectrum(molecule_pair, verbose=True):
        plt.plot(molecule_pair.vector_0, label = 'spectrum 0', alpha=0.5)
        plt.plot(molecule_pair.vector_1, label= 'spectrum 1', alpha=0.5)
        plt.xlabel('m/z')
        plt.ylabel('intensity')
        plt.legend()
        #plt.title(f'molecule_0: {high_molecule_pairs[index].smiles_0}, molecule_1 = {high_molecule_pairs[index].smiles_1}, similarity: {high_molecule_pairs[index].similarity}')
        plt.grid()

        if verbose:
            print(f'molecule_0: {molecule_pair.smiles_0}, molecule_1 = {molecule_pair.smiles_1}, similarity: {molecule_pair.similarity}')