from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings
from mpi4py import MPI
from mpi4py.MPI import Intracomm
from collections import namedtuple
import pickle
import sys
import struct
import argparse

WORKTAG = 0
DIETAG = 1

warnings.filterwarnings("ignore", category=ConvergenceWarning)

Result = namedtuple('Result', ['index', 'l1', 'l2', 'l3', 'score'])

#============================================================================================

#Represents a set of parameters that should be tested by a Worker
class Work:

    def __init__(self, params: dict[str, any]):
        self.params = params

    @classmethod
    def deserialize(cls, bytes: bytes):
        return pickle.loads(bytes)

    def run(self) -> Result:
        return test_params(self.params)

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    def __str__(self) -> str:
        return f"{self.params['index']}: ({self.params['mlp_layer1']}, {self.params['mlp_layer2']}, {self.params['mlp_layer3']})"

    def __repr__(self) -> str:
        return self.__str__()

#============================================================================================    
#We create only one Master
#He contains a queue of the Works that need to be completed
#He distributes Work to Workers
#Keeps track of how many Workers are currently busy
#Once a Worker is done, we get back Result and give him more Work
class Master:

    def __init__(self, comm: Intracomm, parameter_grid):

        self.queue = [Work((params.update({"index": i}), params)[1]) for i, params in enumerate(parameter_grid)]
        self.comm = comm
        self.sent_requests = 0 #How many workers are currently executing work
        self.results = []

    #--------------------------------------------------------------------
    
    #Tells the master to add more work to the queue
    def submit(self, work: Work):
        self.queue.append(work)

    #--------------------------------------------------------------------
    
    #Return next request from the queue
    def _get_next_request(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            return None
        
    #--------------------------------------------------------------------
    
    #Called once at the beginning to distribute initial work to workers
    def distribute_work(self) -> "Master":
        
        for i in range(1, self.comm.Get_size()):
            self.send_work_to(i)

        return self
    
    #--------------------------------------------------------------------

    #Sends the next work object from the queue to a worker that just finished his work
    def send_work_to(self, rank):
        work = self._get_next_request()

        if work is None: return
        comm.isend(work, rank, WORKTAG)

        self.sent_requests += 1

    #--------------------------------------------------------------------

    #Waits for all the workers that currently have received work to finish
    #Once someone finishes, we retrieve Result
    #...and we give him more work, if there's work left in the queue
    def get_results(self) -> list[Result]:
        temp = self.sent_requests
        self.sent_requests = 0

        for _ in range(temp):
            status = MPI.Status()
            res = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()

            #print(f"Worker {worker} finished: {res}")
            self.results.append(res)

            self.send_work_to(worker)

        if self.sent_requests > 0:
            #print("Getting more results...")
            return self.get_results()
        else:
            return sorted(self.results, key=lambda x: x.index)
        
    #--------------------------------------------------------------------

    #💀
    def kill_workers(self):
        for i in range(size):
            self.comm.isend(None, i, DIETAG)

#============================================================================================

class Worker:

    def __init__(self, comm: Intracomm):
        self.comm = comm

    #--------------------------------------------------------------------

    def work(self):
        while(True):
            status = MPI.Status()
            work: Work = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if status.tag == DIETAG:
                return
            
            res = work.run()
            comm.send(res, 0)

#============================================================================================

#Perform the grid search for a specific point on the parameter grid
def test_params(p: dict[str, any]) -> Result:
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), max_iter=500)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)

    return Result(p["index"], l1, l2, l3, ac)

#============================================================================================


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get the values from the arguments
    parser = argparse.ArgumentParser(description='Grid search parallelization', allow_abbrev=False)
    parser.add_argument("-ns", action="store", type=int, default=10000, help="Number of samples")
    parser.add_argument("-nf", action="store", type=int, default=2, help="Number of features")
    parser.add_argument("-v1", action="store", type=int, default=16, help="Small value for neurons")
    parser.add_argument("-v2", action="store", type=int, default=32, help="Large value for neurons")
    args = parser.parse_args()

    if rank == 0:
        print(f"Using MPI master-worker model with {size} processes...\n\nSamples: {args.ns}\nFeatures: {args.nf}\nTesting values: [{args.v1}, {args.v2}]")
        sys.stdout.flush()

    #Set up parameters for Grid Search
    X, y = make_classification(n_samples=args.ns, random_state=42, n_features=args.nf, n_informative=args.nf, n_redundant=0, class_sep=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Only the root makes the parameter grid
    if rank == 0:
        params = [{'mlp_layer1': [args.v1, args.v2],
                'mlp_layer2': [args.v1, args.v2],
                'mlp_layer3': [args.v1, args.v2]}]

        pg = ParameterGrid(params)

    if rank == 0:
        # Perform the grid search with master worker
        t = MPI.Wtime()
        master = Master(comm, pg)
        results = master.distribute_work().get_results()
        t = MPI.Wtime() - t
        print(f"\nTime: {t:.02f}s")

        # Print and Save Results
        with open("mpi_time.bin", "wb") as f:
            f.write(struct.pack('f', t))

        print("\n(l1, l2, l3): score\n---------------------")

        for r in results:
            print(f"({r.l1}, {r.l2}, {r.l3}): {r.score:.4f}")

        print("\n===================================\n")

        sys.stdout.flush()

        master.kill_workers()
            
    else:
        Worker(comm).work()