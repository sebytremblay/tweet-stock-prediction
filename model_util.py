import matplotlib.pyplot as plt
import pickle
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def load_model(X_train, y_train, model_path, model_type, force_reload=False, random_state=42, folds=5):
    """Load a model from a file.

    Args:
        X_train (scipy.sparse._csr.csr_matrix): The training features.
        y_train (numpy.ndarray): The training target variable.
        model_path (str): The path to the model file.
        model_type (str): The type of model to load.
        force_reload (bool): Whether to force the model to be retrained.
        random_state (int): The random state for the model.
        folds (int): The number of cross-validation folds.
    """
    try:
        # Force the model to be retrained
        if force_reload:
            raise ValueError('Forcing model reload.')
        
        # Load the model if it exists
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            print(f'{model_type} model loaded.')
            return model
    except (FileNotFoundError, ValueError):
        # Get the correct model
        if model_type == 'RIDGE':
            model = Ridge(random_state=random_state)
            param_grid = {'alpha': np.logspace(-6, 6, 13)}
        elif model_type == 'LASSO':
            model = Lasso(random_state=random_state)
            param_grid = {'alpha': np.logspace(-6, 6, 13)}
        elif model_type == 'RANDOM FOREST':
            model = RandomForestRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt', 'log2']
            }
        else:
            raise ValueError('Invalid model type.')
        
        # Train the model
        print(f"Training {model_type} model...")
        grid = GridSearchCV(model, param_grid=param_grid, cv=folds, verbose=2, n_jobs = -1, 
                            scoring='neg_mean_squared_error',
                            error_score='raise')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        
        # Save the model
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        print(f'{model_type} model trained and saved.')
        return model

def plot_metric(title, models, metrics, color, xlabel, ylabel):
    """Plot a metric for multiple models.

    Args:
        title (str): The title of the plot.
        models (list): The list of model names.
        metrics (list): The list of metric values.
        color (str): The color of the bars.
        xlabel (str): The x-axis label.
        ylabel (str): The y-axis label.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(models, metrics, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate a model on the test set.

    Args:
        model: The model to evaluate.
        X_test: The test features.
        y_test: The test target variable.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2