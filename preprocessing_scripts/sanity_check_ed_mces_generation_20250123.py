import numpy as np
import pickle

data_path = "/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_20250118/"
file_path = data_path + "mces_indexes_tani_incremental_train_1.npy"
molecular_file = data_path + "edit_distance_neurips_nist_exhaustive.pkl"


array = np.load(file_path)


with open(molecular_file, "rb") as f:
    data = pickle.load(f)

print(f"key of data loaded: {data.keys()}")
spectrums = data["molecule_pairs_train"].spectrums
print(array)
print(f"Size of array: {array.shape}")
print(f"Unique values: {np.unique(array[:,2], return_counts=True)}")


from myopic_mces import MCES


for i in range(0, 20):
    index_0 = array[i, 0]
    index_1 = array[i, 1]

    smiles_0 = spectrums[int(index_0)].params["smiles"]
    smiles_1 = spectrums[int(index_1)].params["smiles"]

    print(f"Smiles 0: {smiles_0}")

    print(f"Smiles 1: {smiles_1}")

    result = MCES(
        smiles_0,
        smiles_1,
        threshold=20,
        i=0,
        # solver='CPLEX_CMD',       # or another fast solver you have installed
        solver="PULP_CBC_CMD",
        solver_options={
            "threads": 1,
            "msg": False,
            "timeLimit": 2,  # Stop CBC after 1 seconds
        },
        # solver_options={'threads': 1, 'msg': False},  # use single thread + no console messages
        no_ilp_threshold=False,  # allow the ILP to stop early once the threshold is exceeded
        always_stronger_bound=False,  # use dynamic bounding for speed
        catch_errors=False,  # typically raise exceptions if something goes wrong
    )
    distance = result[1]
    time_taken = result[2]
    exact_answer = result[3]

    print(f"{i}")
    print(f"Result from sanity check computation: {distance}")
    print(f"Result from loading : {array[i,2]}")
