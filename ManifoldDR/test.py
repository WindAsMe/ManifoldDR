import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import umap

# Digitsで試す
digits = load_digits()

# umapで2次元に削減
reducer = umap.UMAP(n_components=4)
reducer.fit(digits.data)
embedding = reducer.transform(digits.data)
print(embedding.shape)
embedding = reducer.inverse_transform(embedding)
print(embedding.shape)