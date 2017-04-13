import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

class TforTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, norm=None, sublinear_tf=False):
        self._norm = norm
        self._sublinear_tf = sublinear_tf
        

    def fit(self, tf, y):
        n_samples, n_features = tf.shape

        # Masks for positive and negative samples
        pos_samples = sp.spdiags(y, 0, n_samples, n_samples)
        neg_samples = sp.spdiags(1-y, 0, n_samples, n_samples)

        # Extract positive and negative samples
        tf_pos = pos_samples*tf
        tf_neg = neg_samples*tf

        # tp: number of positive samples that contain given term
        # fp: number of positive samples that do not contain given term
        # fn: number of negative samples that contain given term
        # tn: number of negative samples that do not contain given term
        tp = np.bincount(tf_pos.indices, minlength=n_features)
        fp = np.sum(y)-tp
        fn = np.bincount(tf_neg.indices, minlength=n_features)
        tn = np.sum(1-y)-fn

        # Smooth document frequencies
        self._tp = tp + 1.0
        self._fp = fp + 1.0
        self._fn = fn + 1.0
        self._tn = tn + 1.0

        self._n_samples = n_samples
        self._n_features = n_features

        return self

    def transform(self, X):
        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        f = self._n_features
        k = np.log(2 + tp / fn)
        
        if self._sublinear_tf:
            X = TfidfTransformer(norm=None, use_idf=False, sublinear_tf=True).transform(X)
            
        X = X * sp.spdiags(k, 0, f, f)
        
        if self._norm:
            X = normalize(X, self._norm)

        return X