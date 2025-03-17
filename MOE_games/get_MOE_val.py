from MOE_n_bases import MOE_n_bases
import numpy as np
import matplotlib.pyplot as plt

x_vals = []
unentangled_vals = []
nonsignaling_vals = []
quantum_vals = []

for x in [2, 3, 4, 6, 8]:
    ue, ns, qu = MOE_n_bases(x)

    x_vals.append(x)
    unentangled_vals.append(ue)
    nonsignaling_vals.append(ns)
    quantum_vals.append(qu)

# save results to a csv file 
np.savetxt("MOE_diffbases.csv", [x_vals, unentangled_vals, nonsignaling_vals, quantum_vals], delimiter=",")

# Get ratio between entangled and unentangled values
quantum_vals = np.array(quantum_vals)
unentangled_vals = np.array(unentangled_vals)
ratio = quantum_vals / unentangled_vals

# Plot results over x
plt.figure(figsize=(8, 6))
plt.plot(x_vals, unentangled_vals, marker='o', label="Unentangled")
plt.plot(x_vals, nonsignaling_vals, marker='s', label="Non-Signaling")
plt.plot(x_vals, quantum_vals, marker='d', label="Quantum")

plt.xlabel("Num Bases")
plt.legend()
plt.title("MOE Game Values for Different Bases")
plt.show()
plt.savefig("MOE_diffbases.png")

plt.figure(figsize=(8, 6))
plt.plot(x_vals,  ratio, marker='o')
plt.xlabel("Num Bases")
plt.title("MOE Game We(g) / Wue(g) for Different Bases")
plt.show()
plt.savefig("MOE_diffbases_ratio.png")
