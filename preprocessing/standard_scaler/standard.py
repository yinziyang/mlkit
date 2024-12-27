from sklearn.preprocessing import StandardScaler
import numpy as np

# Test Matrix 1: Simple sparse matrix
X1 = np.array([
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
])

# Test Matrix 2: More complex sparse matrix
X2 = np.array([
    [1, 0, 2, 0],
    [0, 3, 0, 4],
    [5, 0, 6, 0],
    [0, 7, 0, 8],
])

# Test Matrix 3: Extremely sparse matrix
X3 = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 0, 0, 3, 0],
])

def test_scaler(X, name="Matrix"):
    print(f"\n=== Testing {name} ===")
    
    # Case 1: with_mean=False, with_std=False
    scaler = StandardScaler(with_mean=False, with_std=False)
    transformed = scaler.fit_transform(X)
    print(f"\nCase 1: {name} with_mean=False, with_std=False")
    print(transformed)
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)

    # Case 2: with_mean=True, with_std=False
    scaler = StandardScaler(with_mean=True, with_std=False)
    transformed = scaler.fit_transform(X)
    print(f"\nCase 2: {name} with_mean=True, with_std=False")
    print(transformed)
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)

    # Case 3: with_mean=False, with_std=True
    scaler = StandardScaler(with_mean=False, with_std=True)
    transformed = scaler.fit_transform(X)
    print(f"\nCase 3: {name} with_mean=False, with_std=True")
    print(transformed)
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)

    # Case 4: with_mean=True, with_std=True
    scaler = StandardScaler(with_mean=True, with_std=True)
    transformed = scaler.fit_transform(X)
    print(f"\nCase 4: {name} with_mean=True, with_std=True")
    print(transformed)
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)

# Run tests for all matrices
test_scaler(X1, "Simple Sparse Matrix")
test_scaler(X2, "Complex Sparse Matrix")
test_scaler(X3, "Extremely Sparse Matrix")