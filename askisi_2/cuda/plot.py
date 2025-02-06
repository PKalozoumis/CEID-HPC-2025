import subprocess
from multiprocessing import shared_memory
import struct
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

N_tests = [100, 500, 1000, 1500, 2000, 2500]
#3000, 3500, 4000, 4500, 5000

#Each row is for a different N value
#Each column is for a different type of time (e.g. gpu time)
times = np.zeros((len(N_tests), 6), dtype=np.float64)

#Run cuda program multiple times
#Get back times
#=============================================================================
N = 1500

shm = shared_memory.SharedMemory(create=True, size=6*8)
ptr = memoryview(shm.buf)

try:
    for i, N in enumerate(N_tests):
        print(f"Running CUDA program with N = {N}\n" + "="*64 + "\n")

        subprocess.run(["./cuda", str(N), "2", shm.name])

        for j in range(6):
            times[i][j] = round(struct.unpack('d', ptr[j*8 : (j+1)*8])[0], 3)

except KeyboardInterrupt:
    print("\nReceived interrupt. Cleaning up shared memory...")
finally:
    ptr.release()
    shm.close()
    shm.unlink()

df = pd.DataFrame(times, columns=["CPU Initialization", "CPU", "GPU (multiple kernels)", "Comparison", "GPU (single kernel)", "Comparison_2"], index=N_tests)

print(df)

#Plot
#=============================================================================

df[["CPU", "GPU (multiple kernels)", "GPU (single kernel)"]].plot()
plt.show()