import subprocess
from multiprocessing import shared_memory
import struct
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse

N = [1e6, 1e7, 1.5e7 ,0.5e8, 1e8]
times = np.zeros((len(N), 4))

shm = shared_memory.SharedMemory(create=True, size=len(N)*4*8)
ptr = memoryview(shm.buf)

try:
    subprocess.run(["./bench", shm.name])
    
    for i in range(len(N)):
        times[i, :] = list(struct.unpack('4d', ptr[32*i : 32*(i+1)]))

    #Save statistics
    #=============================================================================
    features = ["Reference (original)", "Reference (automatic vect.)", "OpenMP SIMD", "AVX-512"]
    df = pd.DataFrame(times, columns=features, index=N)
    df.to_excel("time.xlsx", index_label="Entries", index=True)

    #Plot
    #=============================================================================

    fig, ax = plt.subplots()
    #ax.set_prop_cycle(color=['tab:blue', 'tab:green'])
    ax.plot(N, times, linewidth=2)
    fig.suptitle("WENO5 Time for various implementations")
    ax.set_xlabel("Number of Entries", fontsize=13)
    ax.set_ylabel("Time (s)", fontsize=13)
    ax.legend(features)
    ax.tick_params(labelsize=10)
    ax.grid(color="b", alpha=0.25)
    fig.savefig("weno.png")

except Exception:
    pass
finally:
    ptr.release()
    shm.close()
    shm.unlink()
