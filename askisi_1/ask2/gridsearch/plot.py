import gs
import gs_parallel_pool
import subprocess
import struct
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

samples = 1500
features = 10
v1 = 16
v2 = 32

processes = [2, 4, 6, 8, 10, 12, 14, 16]

times = np.zeros((len(processes), 4))

#Run serial once and store time in the entire row
times[:,0] = gs.gs_serial(samples,features,v1,v2)

for i, proc in enumerate(processes):
    
    subprocess.run(["mpiexec", "-n", str(proc), "python3", "gs_parallel_mpi.py", "-ns", str(samples), "-nf", str(features), "-v1", str(v1), "-v2", str(v2)])
    with open("mpi_time.bin", "rb") as f:
        times[i, 1] = struct.unpack("f", f.read(4))[0]
    os.remove("mpi_time.bin")

    subprocess.run(["mpiexec", "-n", str(proc), "python3", "gs_parallel_futures.py", "-ns", str(samples), "-nf", str(features), "-v1", str(v1), "-v2", str(v2)])
    with open("futures_time.bin", "rb") as f:
        times[i, 2] = struct.unpack("f", f.read(4))[0]
    os.remove("futures_time.bin")
    
    times[i, 3] = gs_parallel_pool.gs_pool(proc, samples, features, v1, v2)

df = pd.DataFrame(times, columns=["Serial", "MPI Master-Worker", "MPI Futures", "Multiprocessing Pool"], index=processes)
df.to_excel("time.xlsx", index_label="Processes", index=True)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)
ax.plot(processes, times, linewidth=2)
ax.set_xticks(range(processes[0], processes[-1]+1))
ax.set_yticks(np.arange(0, math.ceil(np.max(times))+1, 2))
fig.suptitle("Grid Search Time For Various Implementations")
ax.set_xlabel("Processes", fontsize=13)
ax.set_ylabel("Time (s)", fontsize=13)
ax.legend(["Serial", "MPI Master-Worker", "MPI Futures", "Multiprocessing Pool"])
ax.tick_params(labelsize=10)
ax.grid(color="b", alpha=0.25)
fig.savefig("plot.png")
plt.show(block=False)

#Calculate speedup
for col in range(1,4):
    times[:, col] = times[:, 0] / times[:, col]

fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors[1:])
ax.plot(processes, times[:,1:], linewidth=2)
ax.set_xticks(range(processes[0], processes[-1]+1))
ax.set_yticks(np.arange(0, math.ceil(np.max(times))+1, 2))
fig.suptitle("Speedup For Various Implementations")
ax.set_xlabel("N", fontsize=13)
ax.set_ylabel("Speedup", fontsize=13)
ax.legend(["MPI Master-Worker", "MPI Futures", "Multiprocessing Pool"])
ax.tick_params(labelsize=10)
ax.grid(color="b", alpha=0.25)
fig.savefig("speedup.png")
plt.show()
