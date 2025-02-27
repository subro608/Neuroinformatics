import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class EEGSVM:
    """
    SVM model for EEG classification with hyperparameter optimization.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', class_weight=None):
        """
        Initialize the SVM model.
        
        Args:
            kernel (str): Kernel type (e.g., 'rbf', 'linear', 'poly')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            class_weight (dict or str): Class weights
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
        
        self.param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'svm__kernel': ['rbf', 'linear', 'poly'],
        }
        
    def create_pipeline(self):
        """
        Create a pipeline with preprocessing and SVM model.
        
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', self.model)
        ])
        
        return pipeline
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5, n_jobs=-1, verbose=1):
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        pipeline = self.create_pipeline()
        
        grid_search = GridSearchCV(
            pipeline,
            self.param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring='accuracy'
        )
        
        print("Performing grid search for SVM hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_.named_steps['svm']
        
        return grid_search
    
    def train(self, X_train, y_train):
        """
        Train the SVM model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            
        Returns:
            EEGSVM: Self for method chaining
        """
        print("Training SVM model...")
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        train_pred = self.pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X (numpy.ndarray): Features
            
        Returns:
            numpy.ndarray: Predictions
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (numpy.ndarray): Features
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        class_names = ['A', 'C', 'F']
        cm = confusion_matrix(y_test, y_pred)
        
        # Class-specific metrics
        class_metrics = {}
        for i, cls in enumerate(class_names):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            # Calculate metrics
            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            cls_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            class_metrics[cls] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'specificity': cls_specificity
            }
        
        # Binary classification accuracies
        # A vs C
        a_vs_c_mask = np.where(y_test != 2)[0]
        a_vs_c_acc = accuracy_score(y_test[a_vs_c_mask], y_pred[a_vs_c_mask])
        
        # C vs F
        c_vs_f_mask = np.where(y_test != 0)[0]
        c_vs_f_acc = accuracy_score(y_test[c_vs_f_mask], y_pred[c_vs_f_mask])
        
        # A vs F
        a_vs_f_mask = np.where(y_test != 1)[0]
        a_vs_f_acc = accuracy_score(y_test[a_vs_f_mask], y_pred[a_vs_f_mask])
        
        # Create results dictionary
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_metrics': class_metrics,
            'binary_accuracies': {
                'A_vs_C': a_vs_c_acc,
                'C_vs_F': c_vs_f_acc,
                'A_vs_F': a_vs_f_acc
            }
        }
        
        return results


# Example usage
if __name__ == "__main__":
    # Create a simple example
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.choice([0, 1, 2], size=100)
    
    # Create and train model
    svm_model = EEGSVM(kernel='rbf', C=1.0)
    svm_model.train(X, y)
    
    # Evaluate model
    results = svm_model.evaluate(X, y)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")