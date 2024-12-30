import subprocess
import argparse
import sys

parser = argparse.ArgumentParser(description='Grid search parallelization', allow_abbrev=False)
parser.add_argument("--serial", action="store_true")
parser.add_argument("-t", nargs=2, action="append")
args = parser.parse_args()

for arg in args.t:
    if arg[0] not in ["pool", "mpi"]:
        print(f"Invalid type \"{arg[0]}\". Must be \"pool\" or \"mpi\"", file=sys.stderr)
        sys.exit()

print()

if args.serial:
    subprocess.run(["python3", "gs.py"])
    print("===============================================\n")

for i, test in enumerate(args.t):
    try:
        if test[0] == "pool":
            subprocess.run(["python3", "gs_parallel_pool.py", test[1]])
        elif test[0] == "mpi":
            subprocess.run(["mpiexec", "python3", "./gs_parallel_mpi.py", "-n", test[1]])

        if i < len(args.t) - 1:
            print("===============================================\n")

    except KeyboardInterrupt:
        pass