from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from multiprocessing import Pool
from mpi4py import MPI
from mpi4py.MPI import Intracomm
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor

import argparse

'''
parser = argparse.ArgumentParser(description='Grid search parallelization', allow_abbrev=False)
parser.add_argument("numprocs", action="store", type=int)
#parser.add_argument("-type", action="store", type=str)
args = parser.parse_args()
'''

#============================================================================================

class Work:
    params: dict[str, any]

    def __init__(self, params: dict[str, any]):
        Work.params = params

    def run(self) -> float:
        return test_params(params)
    
class Master:

    def submit(self, work: Work):
        Work.queue.append(work)
    
    def __init__(self, comm: Intracomm, queue: list[Work]):
        self.queue: list[Work] = queue
        self.comm = comm
        self.sent_requests = 0
    
    def get_next_request(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            return None
        
    def send_work(self):
        
        for i in range(1, self.comm.Get_size()):
            work = self.get_next_request()
            if work is None: break

            comm.Isend(work, i, "WORKTAG")
            self.sent_requests += 1

    def get_results(self):
        for i in range(self.sent_requests):
            res = None
            comm.Recv(res, MPI.ANY_SOURCE, MPI.ANY_TAG)
        

class Workers:

    comm = None

    def __init__(self, comm: Intracomm):
        self.comm = comm

    def work(self):
        status = MPI.Status()
        work = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.tag == "DIETAG":
            return
        
        work.run

#============================================================================================

def test_params(p: dict[str, any]) -> float:
    print(p)
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)

    return ac
    

X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


params = [{'mlp_layer1': [16, 32],
           'mlp_layer2': [16, 32],
           'mlp_layer3': [16, 32]}]

pg = ParameterGrid(params)



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    Master(comm, pg).send_work()

    
'''

with Pool(processes=8) as pool:
    results = pool.map(test_params, pg)

for r in results:
    print(r)
'''
