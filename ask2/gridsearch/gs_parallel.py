import subprocess
import argparse
import sys
import os
import json

parser = argparse.ArgumentParser(description='Grid search parallelization', allow_abbrev=False)
parser.add_argument("f", nargs="?", action="store", type=str, default="tests.json")
args = parser.parse_args()

if not os.path.exists(args.f):
    print(f"File {args.f} does not exist", file=sys.stderr)
    sys.exit()

with open(args.f, "r") as f:
    tests = json.load(f)

print()

for i, test in enumerate(tests):

    test_type = test["type"]

    train_params = [
        "-ns", str(test.get("samples", 10000)),
        "-nf", str(test.get("features", 2)),
        "-v1", str(test.get("v1", 16)),
        "-v2", str(test.get("v2", 32)),
    ]

    try:
        if test_type == "serial":
            subprocess.run(["python3", "gs.py"] + train_params)
        elif test_type == "pool":
            subprocess.run(["python3", "gs_parallel_pool.py", str(test["processes"])] + train_params)
        elif test_type == "mpi":
            subprocess.run(["mpiexec", "-n", str(test["processes"]), "python3", "./gs_parallel_mpi.py"] + train_params)
        elif test_type == "futures":
            subprocess.run(["mpiexec", "-n", str(test["processes"]), "python3", "./gs_parallel_futures.py"] + train_params)
        else:
            print(f"Invalid type \"{test_type}\". Must be \"pool\", \"mpi\" or \"serial\"", file=sys.stderr)
            sys.exit()

        if i < len(tests) - 1:
            print("===================================================\n")
        else:
            print()

    except KeyboardInterrupt:
        pass