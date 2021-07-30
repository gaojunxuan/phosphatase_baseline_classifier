import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.633109243697479
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=MLPClassifier(alpha=0.1, learning_rate_init=0.001)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=7, min_child_weight=16, n_estimators=100, n_jobs=1, subsample=0.55, verbosity=0)),
    LinearSVC(C=5.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
