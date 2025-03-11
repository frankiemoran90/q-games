import numpy as np
from toqito.states import basis


# The basis: {|0>, |1>}:
e_0, e_1 = basis(2, 0), basis(2, 1)

# The basis: {|+>, |->}:
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

# The basis: {|y+>, |y->}
e_y_p = (e_0 + 1j * e_1) / np.sqrt(2)  # |y+>
e_y_m = (e_0 - 1j * e_1) / np.sqrt(2)  # |y->

# The dimension of referee'sp measurement operators:
dim = 2
# The number of outputs for Alice and Bob:
a_out, b_out = 2, 2
# The number of inputs for Alice and Bob:num_in
a_in, b_in = 3, 3

# Define the predicate matrix V(a,b|x,y) \in Pos(R)
bb84_pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])
bb84_pred_mat = bb84_pred_mat.astype(complex)

# V(0,0|0,0) = |0><0|
bb84_pred_mat[:, :, 0, 0, 0, 0] = e_0 @ e_0.conj().T
# V(1,1|0,0) = |1><1|
bb84_pred_mat[:, :, 1, 1, 0, 0] = e_1 @ e_1.conj().T
# V(0,0|1,1) = |+><+|
bb84_pred_mat[:, :, 0, 0, 1, 1] = e_p @ e_p.conj().T
# V(1,1|1,1) = |-><-|
bb84_pred_mat[:, :, 1, 1, 1, 1] = e_m @ e_m.conj().T
# V(0,0|2,2) = |y+><y+|
bb84_pred_mat[:, :, 0, 0, 2, 2] = e_y_p @ e_y_p.conj().T
# V(1,1|2,2) = |y-><-|
bb84_pred_mat[:, :, 1, 1, 2, 2] = e_y_m @ e_y_m.conj().T

# The probability matrix encode \pi(0,0) = \pi(1,1) = 1/2
bb84_prob_mat = 1/3*np.identity(3)

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame
import numpy as np


print(bb84_prob_mat)
print(bb84_pred_mat)
# Define an ExtendedNonlocalGame object based on the BB84 game.
bb84 = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# The unentangled value is cos(pi/8)**2 \approx 0.85356
print("Unentangled value: ")
print(np.around(bb84.unentangled_value(), decimals=5))

print("Nonsignaling value: ")
print(np.around(bb84.nonsignaling_value(), decimals=5))
#print(np.around(bb84.(), decimals=5))

print("Quantum value Lower Bound: ")
print(np.around(bb84.quantum_value_lower_bound(), decimals=5))

# get classical value
# try 4 bases in toquito
# Mutually unbiased bases for monogoamy 

