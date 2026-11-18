from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

SEED = 42

MODELS = {
    "LogReg": ("linear", lambda: Pipeline([
        ("sc", StandardScaler()),
        ("m", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])),
    "KNN": ("instance-based", lambda: Pipeline([
        ("sc", StandardScaler()),
        ("m", KNeighborsClassifier(n_neighbors=15, weights="distance"))])),
    "RandomForest": ("bagging ensemble", lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=SEED)),
    "LightGBM": ("boosting", lambda: LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=SEED, verbose=-1)),
    "MLP": ("neural network", lambda: Pipeline([
        ("sc", StandardScaler()),
        ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                            early_stopping=True, random_state=SEED))])),
}
