from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import numpy as np

def prepare_data(delai, rain_Index, years_in_data):
    """
    Prépare les données pour l'entraînement et le test.
    """
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(delai, groups=years_in_data):
        X_train, X_test = delai[train_idx].values, delai[test_idx].values
        y_train, y_test = rain_Index[train_idx].values, rain_Index[test_idx].values
        
        # Convertir en 2D si nécessaire
        X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
        X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test
        
        yield X_train, X_test, y_train, y_test
        

def get_best_model(X_train, y_train, param_grid):
    """
    Recherche le meilleur modèle Ridge via GridSearchCV.
    
    Parameters:
        X_train: array-like
            Les données d'entraînement.
        y_train: array-like
            La cible d'entraînement.
        param_grid: dict
            Le dictionnaire des hyperparamètres à tester.
    
    Returns:
        best_model: estimator
            Le meilleur modèle entraîné.
        best_params: dict
            Les meilleurs hyperparamètres trouvés.
    """
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    model = Ridge()
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
    
def ridge_evaluate(model, X_test, y_test):
    """
    Evalue le modèle sur le jeu de test.
    
    Parameters:
        model: estimator
            Le modèle entraîné.
        X_test: array-like
            Les données de test.
        y_test: array-like
            La cible de test.
    
    Returns:
        y_pred: array
            Les prédictions du modèle.
        mae: float
            L'erreur absolue moyenne.
        corr: float
            Le coefficient de corrélation.
        rmse: float
            La racine de l'erreur quadratique moyenne.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, mae, corr, rmse

def run_ridge(delais, delai_names, rain_Index):
    """
    Exécute la recherche d'hyperparamètres et l'évaluation du modèle pour chaque retard.
    
    Parameters:
        delais: list of array-like
            La liste des jeux de données de retards.
        delai_names: list of str
            Les noms correspondants aux retards.
        rain_Index: xarray.DataArray ou similaire
            La variable cible avec une coordonnée 'time'.
    
    Returns:
        model_scores_mae: list of list of float
            Les MAE pour chaque découpage par année et par retard.
        model_scores_corr: list of list of float
            Les coefficients de corrélation pour chaque découpage.
        model_scores_rmse: list of list of float
            Les RMSE pour chaque découpage.
        predictions_by_delay: list of list of array
            Les prédictions du modèle pour chaque découpage.
    """
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    # Extraction des années depuis la coordonnée 'time'
    years_in_data = rain_Index.coords['time'].dt.year.values
    
    model_scores_mae, model_scores_corr, model_scores_rmse, predictions_by_delay = [], [], [], []
    
    # Boucle sur chaque type de retard
    for delai, name in zip(delais, delai_names):
        year_scores_mae, year_scores_corr, year_scores_rmse, predictions_for_delay = [], [], [], []
        
        # Découpage par groupe (année)
        for X_train, X_test, y_train, y_test in prepare_data(delai, rain_Index, years_in_data):
            best_model, best_params = get_best_model(X_train, y_train, param_grid)
            y_pred, mae, corr, rmse = ridge_evaluate(best_model, X_test, y_test)
            
            predictions_for_delay.append(y_pred)
            year_scores_mae.append(mae)
            year_scores_corr.append(corr)
            year_scores_rmse.append(rmse)
        model_scores_mae.append(year_scores_mae)
        model_scores_corr.append(year_scores_corr)
        model_scores_rmse.append(year_scores_rmse)
        predictions_by_delay.append(predictions_for_delay)
    
    return model_scores_mae, model_scores_corr, model_scores_rmse, predictions_by_delay
        

