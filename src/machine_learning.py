import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class RandomForestClassifierWrapper:

    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                oob_score=False, n_jobs=None, random_state=None, verbose=0,
                warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.pca = None
        self.model = None

        def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
             max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
             oob_score=False, n_jobs=None, random_state=None, verbose=0,
             warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
            """
            Initialize the RandomForestClassifierWrapper.

            Parameters
            ----------
            n_estimators : int, default=100
                The number of trees in the forest.

            criterion : {'gini', 'entropy'}, default='gini'
                The function to measure the quality of a split.

            max_depth : int, default=None
                The maximum depth of the tree.

            min_samples_split : int, float, default=2
                The minimum number of samples required to split an internal node.

            min_samples_leaf : int, float, default=1
                The minimum number of samples required to be at a leaf node.

            min_weight_fraction_leaf : float, default=0.0
                The minimum weighted fraction of the sum total of weights (of all input samples) required to be at a leaf node.

            max_features : {'sqrt', 'log2'}, default='sqrt'
                The number of features to consider when looking for the best split.

            max_leaf_nodes : int, default=None
                Grow trees with max_leaf_nodes in best-first fashion.

            min_impurity_decrease : float, default=0.0
                A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

            bootstrap : bool, default=True
                Whether bootstrap samples are used when building trees.

            oob_score : bool, default=False
                Whether to use out-of-bag samples to estimate the generalization accuracy.

            n_jobs : int, default=None
                The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.

            random_state : int, RandomState instance, default=None
                Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.

            verbose : int, default=0
                Controls the verbosity when fitting and predicting.

            warm_start : bool, default=False
                When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.

            class_weight : dict, list of dict or 'balanced', default=None
                Weights associated with classes in the form {class_label: weight}.

            ccp_alpha : non-negative float, default=0.0
                Complexity parameter used for Minimal Cost-Complexity Pruning.

            max_samples : int or float, default=None
                If bootstrap is True, the number of samples to draw from X to train each base estimator. If None, then draw X.shape[0] samples.
            """
    def apply_pca(self, X, threshold=0.95):
        """
        Apply Principal Component Analysis (PCA) to the input data.

        :param X: Input data.
        :type X: np.ndarray

        :param threshold: Variance threshold for PCA.
        :type threshold: float, default=0.95

        :return: Transformed data after PCA.
        :rtype: np.ndarray
        """
        self.pca = PCA(random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        explained_variance_ratio_cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance_ratio_cumulative >= threshold) + 1
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        return X_pca

    def fit(self, X, y, apply_pca=True, pca_threshold=0.95):
        """
        Fit the RandomForestClassifier model to the training data.

        :param X: Training data.
        :type X: np.ndarray

        :param y: Target labels.
        :type y: np.ndarray

        :param apply_pca: Whether to apply PCA to the input data.
        :type apply_pca: bool, default=True

        :param pca_threshold: Variance threshold for PCA.
        :type pca_threshold: float, default=0.95
        """
        if np.unique(y).shape[0] != 3:
            raise ValueError("Invalid number of classes. Expected 3 classes, got {}.".format(np.unique(y).shape[0]))

        X_train_pca = self.apply_pca(X, threshold=pca_threshold)
        X_train_pca, X_val_pca, y_train, y_val = train_test_split(X_train_pca, y, test_size=0.2, random_state=self.random_state)
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.model.fit(X_train_pca, y_train)

    def apply_pca_test(self, X_test):
        """
        Apply Principal Component Analysis (PCA) to the test data.

        :param X_test: Test data.
        :type X_test: np.ndarray

        :return: Transformed test data after PCA.
        :rtype: np.ndarray
        """
        if self.pca is not None:
            X_test_pca = self.pca.transform(X_test)
            return X_test_pca
        else:
            raise ValueError("PCA has not been applied to the training data. Please fit the model first.")

    def predict(self, X_test):
        """
        Predict class labels for the test data.

        :param X_test: Test data.
        :type X_test: np.ndarray

        :return: Predicted class labels.
        :rtype: np.ndarray
        """
        X_test_pca = self.apply_pca_test(X_test)
        return self.model.predict(X_test_pca)

    def predict_proba(self, X_test):
        """
        Predict class probabilities for the test data.

        :param X_test: Test data.
        :type X_test: np.ndarray

        :return: Predicted class probabilities.
        :rtype: np.ndarray
        """
        X_test_pca = self.apply_pca_test(X_test)
        return self.model.predict_proba(X_test_pca)

    def evaluate(self, y_pred, y_true):
        """
        Evaluate the model's performance using a confusion matrix.

        :param y_pred: Predicted class labels.
        :type y_pred: np.ndarray

        :param y_true: True class labels.
        :type y_true: np.ndarray

        :return: Confusion matrix.
        :rtype: np.ndarray
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm
