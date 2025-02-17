import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

def prepare_data(delai, rain_Index, years_in_data):
    """
    Prépare les données pour l'entraînement et le test.
    """
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(delai, groups=years_in_data):
        X_train, X_test = delai[train_idx].values, delai[test_idx].values
        y_train, y_test = rain_Index[train_idx].values, rain_Index[test_idx].values

        # Reshape pour s'assurer d'avoir une matrice 2D (samples, features)
        X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
        X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test
        
        yield X_train, X_test, y_train, y_test

def get_best_ridge(X_train, y_train, param_grid):
    """
    Trouve les meilleurs hyperparamètres pour un modèle Ridge.
    """
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    model = Ridge()
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def ridge_evaluate(model, X_test, y_test):
    """
    Évalue le modèle Ridge sur l'ensemble de test.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else np.nan  # Corrélation non définie pour 1 valeur

    return y_pred, mae, corr, rmse

def run_ridge(delais, delai_names, rain_Index, Rain_Index):
    """
    Applique Ridge avec les hyperparamètres optimaux trouvés avec `rain_Index` sur `Rain_Index` (2D spatial).
    """
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    years_in_data = rain_Index.coords['time'].dt.year.values
    time_dim = rain_Index.sizes["time"]  # 662
    years_dim = len(np.unique(years_in_data))  # 38
    space_dim = Rain_Index.sizes["space"]  # 14000
    delay_dim = len(delais)  # 6

    # Initialisation des matrices de stockage
    model_scores_mae = np.full((delay_dim, years_dim, space_dim), np.nan)
    model_scores_corr = np.full((delay_dim, years_dim, space_dim), np.nan)
    model_scores_rmse = np.full((delay_dim, years_dim, space_dim), np.nan)
    predictions_by_delay = np.full((delay_dim, time_dim, space_dim), np.nan)

    best_params_by_delai = {}

    # Phase 1 : Trouver les meilleurs hyperparamètres avec rain_Index (1D time)
    for i, (delai, name) in enumerate(zip(delais, delai_names)):
        best_params = None
        for X_train, X_test, y_train, y_test in prepare_data(delai, rain_Index, years_in_data):
            _, best_params = get_best_ridge(X_train, y_train, param_grid)
        
        best_params_by_delai[name] = best_params  # Stockage des meilleurs paramètres pour chaque délai

    # Phase 2 : Appliquer sur Rain_Index (2D space)
    space_points = Rain_Index.coords['space'].values  # 14000 points spatiaux

    for i, (delai, name) in enumerate(zip(delais, delai_names)):
        best_params = best_params_by_delai[name]

        for j, space in enumerate(space_points):
            sub_Rain = Rain_Index.sel(space=space)  # Extraction du point spatial

            for train_idx, test_idx in LeaveOneGroupOut().split(delai, groups=years_in_data):
                X_train, X_test = delai[train_idx].values, delai[test_idx].values
                y_train, y_test = sub_Rain[train_idx].values, sub_Rain[test_idx].values

                X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
                X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test
                
                model = Ridge(**best_params)
                model.fit(X_train, y_train)
                y_pred, mae, corr, rmse = ridge_evaluate(model, X_test, y_test)
                
                # Stocker les scores
                year_test = np.unique(years_in_data[test_idx])  # Années utilisées pour test
                for y, test_year in enumerate(year_test):
                    year_idx = np.where(np.unique(years_in_data) == test_year)[0][0]  # Trouver l'index dans les 38 années
                    model_scores_mae[i, year_idx, j] = mae
                    model_scores_corr[i, year_idx, j] = corr
                    model_scores_rmse[i, year_idx, j] = rmse
                
                # Stocker les prédictions
                predictions_by_delay[i, test_idx, j] = y_pred  # Stockage pour les timestamps correspondants
    
    return model_scores_mae, model_scores_corr, model_scores_rmse, predictions_by_delay

