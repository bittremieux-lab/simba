from myopic_mces import MCES 



#s0= r'CC1OC(OC(=O)C23CCC(C)(C)CC2C2=CCC4C5(C)CCC(OC6OC(C(=O)O)C(O)C(OC7OCC(O)C(O)C7O)C6OC6OC(CO)C(O)C(O)C6O)C(C)(C=O)C5CCC4(C)C2(C)CC3O)C(OC2OC(C)C(OC3OCC(O)C(OC4OCC(O)(CO)C4O)C3O)C(OC3OC(CO)C(O)C(O)C3O)C2O)C(O)C1O'
#s1=r'CC(=O)OC1C(C)OC(OC2C(C)OC(OC(=O)C34CCC(C)(C)CC3C3=CCC5C6(C)CCC(OC7OC(C(=O)O)C(O)C(OC8OCC(O)C(O)C8O)C7OC7OC(CO)C(O)C(O)C7O)C(C)(C=O)C6CCC5(C)C3(C)CC4)C(OC3OC(C)C(O)C(OC4OCC(O)C(OC5OCC(O)C(O)C5O)C4O)C3O)C2O)C(O)C1OC(C)=O'
#s0=r'COC(=O)c1cc(S(C)(=O)=O)ccc1[N+](=O)[O-]'
#s1=r'N[C@@H]1[C@@H](O)[C@H](O)[C@@H](COP(=O)(O)O)O[C@@H]1O'
s0=r'NS(=O)(=O)c1ccc(Cl)c(C(F)(F)F)c1'
s1=r'NS(=O)(=O)c1cc(C(F)(F)F)ccc1Cl'
print('Starting computation:')

import time


import pandas as pd
import numpy as np 
df= pd.read_csv('/scratch/antwerpen/209/vsc20939/data/smiles_train_16012025.csv')

smiles= df['Smiles'].values

N=1000

random_integers_0= np.random.randint(0, len(smiles), N)
random_integers_1= np.random.randint(0, len(smiles), N)

s0_list= [smiles[r] for r in random_integers_0]
s1_list= [smiles[r] for r in random_integers_1]

from pulp import CPLEX_CMD

start_time = time.perf_counter()
distance_total=[]




for index, (s0,s1) in enumerate(zip(s0_list, s1_list)):
    TIME_LIMIT=2
    result =     MCES(
                                            s0,
                                            s1,
                                            threshold=20,
                                            i=0,
                                            solver='PULP_CBC_CMD',
                                            solver_options={
                                                'threads': 1, 
                                                'msg': False,
                                                'timeLimit':TIME_LIMIT# Stop CBC after 0.1 seconds
                                            },  
                                            no_ilp_threshold=False,   # allow the ILP to stop early once the threshold is exceeded
                                            always_stronger_bound=False,  # use dynamic bounding for speed
                                            catch_errors=False        # typically raise exceptions if something goes wrong
                                        )

    print(f'{index}: {result}')
    distance = result[1]
    time_taken=result[2]
    exact_answer= result[3]

    if (time_taken >=(0.9*TIME_LIMIT) and (exact_answer != 1)):
        distance= np.nan
    distance_total.append(distance)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print('Finished computation:')

distance_total=np.array(distance_total)
print(f'distances: {distance_total}')
print(f'all samples: {distance_total.shape[0]} ')
print(f' less than 20: {distance_total[distance_total<20].shape[0]}')
print(f'Percentage of   indetermined samples: {100*np.isnan(distance_total).sum()/N}')

print(f'Unique values of distance_total: {np.unique(distance_total)}')
