import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
sys.path.append("./src/")
from machine_learning import RandomForestClassifierWrapper



class TestRandomForestClassifierWrapper(unittest.TestCase):

    def setUp(self):
        """ Setup method to create a sample dataset """
        self.X, self.y = make_classification(
                            n_samples=1000,  
                            n_features=3000,
                            n_classes=3,
                            n_clusters_per_class=1,
                            n_informative=300,
                            n_redundant=0,
                            random_state=42)

    def test_initialization_defaults(self):
        """Test the initialization of the RandomForestClassifierWrapper with default values"""
        clf = RandomForestClassifierWrapper()
        self.assertEqual(clf.n_estimators, 100)
        self.assertIsNone(clf.model)
        self.assertIsNone(clf.pca)

    def test_initialization_custom_values(self):
        """Test the initialization of the RandomForestClassifierWrapper with custom values"""
        clf = RandomForestClassifierWrapper(
            n_estimators=50, criterion='entropy', max_depth=10, min_samples_split=5,
            min_samples_leaf=2, max_features='log2', random_state=42
        )
        self.assertEqual(clf.n_estimators, 50)
        self.assertEqual(clf.criterion, 'entropy')
        self.assertEqual(clf.max_depth, 10)
        self.assertEqual(clf.min_samples_split, 5)
        self.assertEqual(clf.min_samples_leaf, 2)
        self.assertEqual(clf.max_features, 'log2')
        self.assertEqual(clf.random_state, 42)
        self.assertIsNone(clf.model)
        self.assertIsNone(clf.pca)

    def test_apply_pca_defaults(self):
        """Test apply_pca method with default threshold"""
        clf = RandomForestClassifierWrapper()
        X_pca = clf.apply_pca(self.X)
        self.assertIsNotNone(clf.pca)
        self.assertEqual(X_pca.shape[0], self.X.shape[0])
        self.assertTrue(X_pca.shape[1] <= self.X.shape[1])

    def test_fit_invalid_num_classes(self):
        """Test fit method with invalid number of classes"""
        clf = RandomForestClassifierWrapper()
        with self.assertRaises(ValueError):
            clf.fit(self.X, np.array([0, 1, 2, 2]))  

    def test_fit_with_pca(self):
        """Test fit method with PCA"""
        clf = RandomForestClassifierWrapper()
        clf.fit(self.X, self.y)
        self.assertIsNotNone(clf.model)
        self.assertIsNotNone(clf.pca)


    def test_apply_pca_test_with_pca(self):
        """Test apply_pca_test method with PCA"""
        clf = RandomForestClassifierWrapper()
        X_pca = clf.apply_pca(self.X)  
        X_test_pca = clf.apply_pca_test(self.X)
        self.assertIsNotNone(clf.pca)
        self.assertEqual(X_test_pca.shape[1], X_pca.shape[1])

    def test_apply_pca_test_without_pca(self):
        """ Test apply_pca_test method without PCA"""
        clf = RandomForestClassifierWrapper()
        with self.assertRaises(ValueError):
            clf.apply_pca_test(self.X)  

    def test_predict(self):
        """ Test apply_pca_test method without PCA"""
        clf = RandomForestClassifierWrapper()
        clf.fit(self.X, self.y)
        y_pred = clf.predict(self.X)
        self.assertEqual(len(y_pred), self.X.shape[0])

    def test_predict_proba(self):
        """ Test apply_pca_test method without PCA"""
        clf = RandomForestClassifierWrapper()
        clf.fit(self.X, self.y)
        proba = clf.predict_proba(self.X)
        print(proba.shape[1])
        self.assertEqual(proba.shape[0], self.X.shape[0])
        self.assertEqual(proba.shape[1], 3) 

    def test_evaluate(self):
        """ Test the evaluate method """
        clf = RandomForestClassifierWrapper()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = clf.evaluate(y_pred = y_pred, y_true = y_test)
        self.assertEqual(np.array(cm).sum(axis=1).sum(), len(y_test))

if __name__ == '__main__':
    unittest.main()