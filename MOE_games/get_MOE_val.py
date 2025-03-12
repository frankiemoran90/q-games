from MOE_n_bases import MOE_n_bases
import numpy as np
import matplotlib.pyplot as plt

x_vals = []
unentangled_vals = []
nonsignaling_vals = []
quantum_vals = []

for x in [6, 8]:
    ue, ns, qu = MOE_n_bases(x)

    x_vals.append(x)
    unentangled_vals.append(ue)
    nonsignaling_vals.append(ns)
    quantum_vals.append(qu)

# Plot results over x
plt.plot(x_vals, unentangled_vals, marker='o', label="Unentangled Values")
plt.plot(x_vals, nonsignaling_vals, marker='s', label="Non-Signaling Values")
plt.plot(x_vals, quantum_vals, marker='d', label="Quantum Values")

plt.xlabel("Num Bases")
plt.legend()
plt.show()
