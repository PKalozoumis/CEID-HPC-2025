#serial execution of grid search

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings
from collections import namedtuple
import time
import argparse

warnings.filterwarnings("ignore", category=ConvergenceWarning)

#======================================================================================================

# Serial Grid Search
def gs_serial(ns,nf,v1,v2):
    
    Result = namedtuple('Result', ['l1', 'l2', 'l3', 'score'])

    print(f"Serial execution...\n\nSamples: {ns}\nFeatures: {nf}\nTesting values: [{v1}, {v2}]")

    X, y = make_classification(n_samples=ns, random_state=42, n_features=nf, n_informative=nf, n_redundant=0, class_sep=0.8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    params = [{'mlp_layer1': [v1, v2],
            'mlp_layer2': [v1, v2],
            'mlp_layer3': [v1, v2]}]

    pg = ParameterGrid(params)

    t = time.time()

    results = []

    # Perform the grid search by traversing the parameter grid
    for p in pg:
        l1 = p['mlp_layer1']
        l2 = p['mlp_layer2']
        l3 = p['mlp_layer3']
        m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), max_iter=500)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        ac = accuracy_score(y_pred, y_test)
        results.append(Result(l1, l2, l3, ac))

    print(f"\nTime: {time.time() - t:.02f}s")

    print("\n(l1, l2, l3): score\n---------------------")

    for r in results:
        print(f"({r.l1}, {r.l2}, {r.l3}): {r.score:.4f}")
    
    print("\n===================================\n")


    return time.time() - t

#======================================================================================================

if __name__ == "__main__":

    # Get the values from the arguments
    parser = argparse.ArgumentParser(description='Grid search parallelization', allow_abbrev=False)

    parser.add_argument("-ns", action="store", type=int, default=20000, help="Number of samples")
    parser.add_argument("-nf", action="store", type=int, default=10, help="Number of features")
    parser.add_argument("-v1", action="store", type=int, default=16, help="Small value for neurons")
    parser.add_argument("-v2", action="store", type=int, default=32, help="Large value for neurons")

    args = parser.parse_args()

    gs_serial(args.ns, args.nf, args.v1, args.v2)