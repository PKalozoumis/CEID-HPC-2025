from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# Δημιουργία δεδομένων
X, y = make_classification(n_samples=10000, random_state=42, n_features=2, 
                            n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Ορισμός παραμέτρων για την αναζήτηση
params = [{'mlp_layer1': [16, 32], 'mlp_layer2': [16, 32], 'mlp_layer3': [16, 32]}]
pg = list(ParameterGrid(params))

# Συνάρτηση για την εκτέλεση του μοντέλου
def evaluate_model(p):
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    model = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_pred, y_test)

# Master-Worker με MPI για διανομή των εργασιών
def mpi_parallel_execution():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master: Διανομή παραμέτρων στους άλλους κόμβους
        with MPIPoolExecutor(comm=comm) as executor:
            results = list(executor.map(evaluate_model, pg))

        # Εκτύπωση αποτελεσμάτων
        for i, acc in enumerate(results):
            print(f"Iteration {i}, Accuracy: {acc}")
    else:
        # Worker: Εκτέλεση του μοντέλου
        for param in pg:
            result = evaluate_model(param)
            comm.send(result, dest=0)

# Χρήση Multiprocessing Pool για εκτέλεση σε τοπικό επίπεδο (εντός του κόμβου)
def multiprocessing_parallel_execution():
    with Pool() as pool:
        results = pool.map(evaluate_model, pg)
    
    # Εκτύπωση αποτελεσμάτων
    for i, acc in enumerate(results):
        print(f"Iteration {i}, Accuracy: {acc}")

# Συνδυασμός όλων των μεθόδων (MPI Futures + Multiprocessing + Master-Worker)
def combined_parallel_execution():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Master κόμβος
    if rank == 0:
        with MPIPoolExecutor(comm=comm) as executor:
            # Επικοινωνία μέσω MPI για διανομή παραμέτρων
            results = list(executor.map(evaluate_model, pg))

        # Εκτύπωση αποτελεσμάτων
        for i, acc in enumerate(results):
            print(f"Iteration {i}, Accuracy: {acc}")

    # Worker κόμβοι
    else:
        # Χρησιμοποίηση multiprocessing για επεξεργασία παραμέτρων εντός κάθε worker
        with Pool() as pool:
            results = pool.map(evaluate_model, pg)
        
        # Αποστολή αποτελεσμάτων πίσω στον master
        for result in results:
            comm.send(result, dest=0)

# Καλέστε τη συνδυασμένη παραλληλία
if __name__ == "__main__":
    combined_parallel_execution()  # Χρησιμοποιούμε τον συνδυασμένο τρόπο
