from sklearn.decomposition import PCA
import numpy as np

# 测试数据
X = np.array([
    [6, 5, 4, 3, 8, 2, 9],
    [5, 1, 10, 2, 3, 8, 7],
    [5, 14, 2, 3, 6, 3, 2]
])

def test_pca(n_components):
    print(f"\n=== Testing PCA with n_components={n_components} ===")
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X)
    
    print("\nTransformed data:")
    print(transformed)
    print("\nComponents:")
    print(pca.components_)
    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("\nTotal explained variance ratio:")
    print(sum(pca.explained_variance_ratio_))
    print("\nMean:")
    print(pca.mean_)
    
    # 还原数据
    reconstructed = pca.inverse_transform(transformed)
    print("\nReconstructed data:")
    print(reconstructed)
    print("\nReconstruction error:")
    print(np.mean((X - reconstructed) ** 2))

# 测试不同的n_components
test_pca(None)  # 使用所有组件
test_pca(0.95)  # 保留95%的方差
test_pca(0.5)   # 保留50%的方差
