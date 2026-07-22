import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class EEGClassifier:
    """Classifier wrapper for EEG BCI using SVM or LDA."""

    def __init__(self, method="svm", kernel="rbf"):
        """
        Args:
            method: "svm" or "lda"
            kernel: SVM kernel (only used if method="svm")
        """
        self.method = method.lower()
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

        if self.method == "svm":
            self.model = SVC(kernel=kernel, probability=True, C=1.0, gamma='scale')
        elif self.method == "lda":
            self.model = LinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'svm' or 'lda'.")

    def train(self, features, labels):
        """Train the classifier.

        Args:
            features: ndarray (n_trials, n_features)
            labels: ndarray (n_trials,)

        Returns:
            dict with training accuracy and cross-val scores
        """
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)
        self.is_fitted = True

        train_acc = self.model.score(scaled, labels)

        # Cross-validation
        cv = StratifiedKFold(n_splits=min(5, len(labels)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, scaled, labels, cv=cv, scoring='accuracy')

        results = {
            "train_accuracy": train_acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores
        }

        print(f"[Classifier] {self.method.upper()} | Train Acc: {train_acc:.4f} | CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        return results

    def predict(self, features):
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        scaled = self.scaler.transform(features)
        return self.model.predict(scaled)

    def predict_proba(self, features):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        scaled = self.scaler.transform(features)
        return self.model.predict_proba(scaled)

    def score(self, features, labels):
        """Compute accuracy."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        scaled = self.scaler.transform(features)
        return self.model.score(scaled, labels)
