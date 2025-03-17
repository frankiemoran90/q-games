import numpy as np
from toqito.states import basis
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame
import time

# Computes stats for an MOE game with N bases where n 
#is a prime >=5
def MOE_n_bases(NB):
     # Hardcode these values, they cannot be found using this setup
     # order is unentangles, ns, quantum
     if(NB == 2):
          return 0.85355, 0.85355, 0.85355

     if(NB == 3):
          return 0.78867, 0.85355, 0.78867

     if(NB == 4):
          return 0.65451, 0.78867, 0.66099


     start_time = time.time()
     # NB is the number of bases, dim is one less
     dim = NB - 1

     # for MOE prob matrix is the identity
     prob_mat = 1 / NB * np.identity(NB)

     # generate the first computational basis
     comp = [basis(dim, i) for i in range(dim)]
     mubs = [comp]
     basis_vec = []
     temp_b = 0

     # const eta
     eta = np.exp((2 * np.pi * 1j) / dim)

     # generate the mutually unbiased bases
     for a in range(dim):
          for b in range(dim):
               for x in range(dim):
                    temp_b += ((eta ** ((a * x ** 2) + (b * x)) *comp[x]) / np.sqrt(dim))
               basis_vec.append(temp_b)
               temp_b = 0
          mubs.append(basis_vec)
          basis_vec = []

     # generate the predicate matrix
     num_in = NB
     num_out = NB - 1
     pred_mat = np.zeros([dim, dim, num_out, num_out, num_in, num_in], dtype=complex)

     for o in range(num_out):
          for i in range(num_in):
               pred_mat[:, :, o, o, i, i] = mubs[i][o] @ mubs[i][o].conj().T

     # setup the game and compute stats
     g_mub = ExtendedNonlocalGame(prob_mat, pred_mat)

     unent_val = g_mub.unentangled_value()
     ns_val = g_mub.nonsignaling_value()
     q_val = g_mub.quantum_value_lower_bound()

     end_time = time.time()
     print('\n')
     print('\n')
     print('Time to generate {NB} bases: {time}'.format(NB=NB, time=end_time - start_time))
     # return 3 IMPORTANT VALS
     return unent_val, ns_val, q_val
