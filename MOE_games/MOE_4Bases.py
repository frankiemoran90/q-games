import numpy as np
from toqito.states import basis
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame


prob_mat = 1 / 4 * np.identity(4)

dim = 3
e_0, e_1, e_2 = basis(dim, 0), basis(dim, 1), basis(dim, 2)

eta = np.exp((2 * np.pi * 1j) / dim)
mub_0 = [e_0, e_1, e_2]
mub_1 = [
     (e_0 + e_1 + e_2) / np.sqrt(3),
     (e_0 + eta ** 2 * e_1 + eta * e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + eta ** 2 * e_2) / np.sqrt(3),
]
mub_2 = [
     (e_0 + e_1 + eta * e_2) / np.sqrt(3),
     (e_0 + eta ** 2 * e_1 + eta ** 2 * e_2) / np.sqrt(3),
     (e_0 + eta * e_1 + e_2) / np.sqrt(3),
]
mub_3 = [
     (e_0 + e_1 + eta ** 2 * e_2) / np.sqrt(3),
     (e_0 + eta ** 2 * e_1 + e_2) / np.sqrt(3),
     (e_0 + eta * e_1 + eta * e_2) / np.sqrt(3),
]

# List of measurements defined from mutually unbiased basis.
mubs = [mub_0, mub_1, mub_2, mub_3]
print(mubs)

num_in = 4
num_out = 3
pred_mat = np.zeros([dim, dim, num_out, num_out, num_in, num_in], dtype=complex)

pred_mat[:, :, 0, 0, 0, 0] = mubs[0][0] @ mubs[0][0].conj().T
pred_mat[:, :, 1, 1, 0, 0] = mubs[0][1] @ mubs[0][1].conj().T
pred_mat[:, :, 2, 2, 0, 0] = mubs[0][2] @ mubs[0][2].conj().T

pred_mat[:, :, 0, 0, 1, 1] = mubs[1][0] @ mubs[1][0].conj().T
pred_mat[:, :, 1, 1, 1, 1] = mubs[1][1] @ mubs[1][1].conj().T
pred_mat[:, :, 2, 2, 1, 1] = mubs[1][2] @ mubs[1][2].conj().T

pred_mat[:, :, 0, 0, 2, 2] = mubs[2][0] @ mubs[2][0].conj().T
pred_mat[:, :, 1, 1, 2, 2] = mubs[2][1] @ mubs[2][1].conj().T
pred_mat[:, :, 2, 2, 2, 2] = mubs[2][2] @ mubs[2][2].conj().T

pred_mat[:, :, 0, 0, 3, 3] = mubs[3][0] @ mubs[3][0].conj().T
pred_mat[:, :, 1, 1, 3, 3] = mubs[3][1] @ mubs[3][1].conj().T
pred_mat[:, :, 2, 2, 3, 3] = mubs[3][2] @ mubs[3][2].conj().T

g_mub = ExtendedNonlocalGame(prob_mat, pred_mat)
unent_val = g_mub.unentangled_value()

print("Unentangled value: ")
print(np.around(unent_val, decimals=5))

print("Nonsignaling value: ")
print(np.around(g_mub.nonsignaling_value(), decimals=5))

q_val = g_mub.quantum_value_lower_bound()

print("Quantum value Lower Bound: ")
print(np.around(q_val, decimals=5))