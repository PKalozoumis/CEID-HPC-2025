from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from collections import namedtuple
import time
import argparse

parser = argparse.ArgumentParser(description='Grid search parallelization', allow_abbrev=False)
parser.add_argument("-ns", action="store", type=int, default=10000, help="Number of samples")
parser.add_argument("-nf", action="store", type=int, default=2, help="Number of features")
parser.add_argument("-v1", action="store", type=int, default=16, help="Small value for neurons")
parser.add_argument("-v2", action="store", type=int, default=32, help="Large value for neurons")
args = parser.parse_args()

Result = namedtuple('Result', ['l1', 'l2', 'l3', 'score'])

print(f"Serial execution...\n\nSamples: {args.ns}\nFeatures: {args.nf}\nTesting values: [{args.v1}, {args.v2}]")

X, y = make_classification(n_samples=args.ns, random_state=42, n_features=args.nf, n_informative=args.nf, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


params = [{'mlp_layer1': [args.v1, args.v2],
           'mlp_layer2': [args.v1, args.v2],
           'mlp_layer3': [args.v1, args.v2]}]

pg = ParameterGrid(params)

t = time.time()

results = []

for i, p in enumerate(pg):
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

print()
