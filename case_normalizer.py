import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CaseNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.Series(X).apply(lambda x: x.lower()).values
    
normalizer = CaseNormalizer()

X = np.array(['Implementing', 'a', 'Custom', 'Transformer',' From', 'SCIKIT-LEARN'])
print(normalizer.transform(X))