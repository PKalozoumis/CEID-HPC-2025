import subprocess
from multiprocessing import shared_memory
import struct
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse
import time

TIME_INIT = 0
TIME_CPU = 1
TIME_GPU_NAIVE = 2
TIME_CMP  = 3
TIME_GPU = 4
TIME_CMP_2 = 5

parser = argparse.ArgumentParser(description='Driver script for the complex matrix multiplication in CUDA', allow_abbrev=False)
parser.add_argument("--gpu-only", action="store_true", required=False, default=False)
args = parser.parse_args()

if args.gpu_only:
    N_tests = list(range(250, 10250, 250))
else:
    N_tests = list(range(250, 5250, 250))

#Each row is for a different N value
#Each column is for a different type of time (e.g. gpu time)
times = np.zeros((len(N_tests), 6), dtype=np.float64)

#Run cuda program multiple times
#Get back times through shared mem
#=============================================================================

shm = shared_memory.SharedMemory(create=True, size=6*8)
ptr = memoryview(shm.buf)

try:
    for i, N in enumerate(N_tests):
        print(f"Running CUDA program with N = {N}\n" + "="*64 + "\n")

        subprocess.run(["./cuda", str(N), "6" if args.gpu_only else "7", shm.name])

        for j in range(6):
            times[i, j] = round(struct.unpack('d', ptr[j*8 : (j+1)*8])[0], 3)

except KeyboardInterrupt:
    print("\nReceived interrupt. Showing plots with the data collected so far...")
    i = i-1
finally:
    i = i+1
    ptr.release()
    shm.close()
    shm.unlink()

#Save statistics
#=============================================================================
np.savez("arr.npz", array=times, metadata={"N": N_tests, "stopped": i})

df = pd.DataFrame(times[:, [TIME_INIT, TIME_CPU, TIME_GPU_NAIVE, TIME_GPU, TIME_CMP_2]], columns=["Init", "CPU", "GPU (naive)", "GPU", "Comparison"], index=N_tests)

if args.gpu_only:
    df[["Init", "GPU (naive)", "GPU"]].to_excel("time.xlsx", index_label="N", index=True)
else:
    df.to_excel("time.xlsx", index_label="N", index=True)

#Plot
#=============================================================================

if not args.gpu_only:
    fig, ax = plt.subplots()
    ax.plot(N_tests[:i], times[:i, TIME_CPU] / times[:i, TIME_GPU], linewidth=2)
    fig.suptitle("Speedup for various matrix sizes")
    ax.set_xlabel("N", fontsize=13)
    ax.set_ylabel("Speedup", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(color="b", alpha=0.25)
    fig.savefig("speedup.png")

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=['tab:blue', 'tab:green'])
    ax.plot(N_tests[:i], times[:i, [TIME_CPU, TIME_GPU]], linewidth=2)
    fig.suptitle("Computation Time for various matrix sizes")
    ax.set_xlabel("N", fontsize=13)
    ax.set_ylabel("Time (s)", fontsize=13)
    ax.legend(["CPU", "GPU"])
    ax.tick_params(labelsize=10)
    ax.grid(color="b", alpha=0.25)
    fig.savefig("cpu_gpu.png")

fig, ax = plt.subplots()
ax.set_prop_cycle(color=['tab:blue', 'tab:green'])
ax.plot(N_tests[:i], times[:i, [TIME_GPU, TIME_GPU_NAIVE]], linewidth=2)
fig.suptitle("Computation Time between two implementations")
ax.set_xlabel("N", fontsize=13)
ax.set_ylabel("Time (s)", fontsize=13)
ax.legend(["GPU (single kernel)", "GPU (multiple kernels)"])
ax.tick_params(labelsize=10)
ax.grid(color="b", alpha=0.25)
fig.savefig("gpu.png")

plt.show()