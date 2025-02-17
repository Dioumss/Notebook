import numpy as np
import xarray as xr
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

MODEL_FILE = "random_forest_models.pkl"  # Nom unique du fichier de stockage

def save_models(models_dict):
    """Sauvegarde tous les modèles dans un seul fichier .pkl."""
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(models_dict, f)

def load_models():
    """Charge tous les modèles à partir du fichier .pkl."""
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}  # Retourne un dictionnaire vide si aucun fichier trouvé


def run_random_forest(delais, delai_names, rain_Index, Rain_Index):
    """
    Applique Random Forest avec les hyperparamètres optimaux trouvés avec `rain_Index` sur `Rain_Index` (2D spatial).
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    years_in_data = rain_Index.coords['time'].dt.year.values
    time_dim = rain_Index.sizes["time"]
    years_dim = len(np.unique(years_in_data))
    space_dim = Rain_Index.sizes["space"]
    delay_dim = len(delais)

    # Matrices de stockage des résultats
    model_scores_mae = np.full((delay_dim, years_dim, space_dim), np.nan)
    model_scores_corr = np.full((delay_dim, years_dim, space_dim), np.nan)
    model_scores_rmse = np.full((delay_dim, years_dim, space_dim), np.nan)
    predictions_by_delay = np.full((delay_dim, time_dim, space_dim), np.nan)

    best_params_by_delai = {}
    models_dict = load_models()  # Charger tous les modèles existants

    # Phase 1 : Trouver les meilleurs hyperparamètres avec rain_Index
    for i, (delai, name) in enumerate(zip(delais, delai_names)):
        best_params = None
        for X_train, X_test, y_train, y_test in prepare_data(delai, rain_Index, years_in_data):
            _, best_params = get_best_random_forest(X_train, y_train, param_grid)
        
        best_params_by_delai[name] = best_params  # Stockage des meilleurs paramètres pour chaque délai

    # Phase 2 : Appliquer sur Rain_Index (2D spatial)
    space_points = Rain_Index.coords['space'].values

    for i, (delai, name) in enumerate(zip(delais, delai_names)):
        best_params = best_params_by_delai[name]

        for j, space in enumerate(space_points):
            sub_Rain = Rain_Index.sel(space=space)  # Extraction du point spatial

            key = (name, space)
            if key in models_dict:
                model = models_dict[key]  # Charger le modèle existant
            else:
                model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                
                for train_idx, test_idx in LeaveOneGroupOut().split(delai, groups=years_in_data):
                    X_train, X_test = delai[train_idx].values, delai[test_idx].values
                    y_train, y_test = sub_Rain[train_idx].values, sub_Rain[test_idx].values

                    X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
                    X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test

                    model.fit(X_train, y_train)
                
                models_dict[key] = model  # Sauvegarde du modèle dans le dictionnaire

            # Évaluation du modèle
            y_pred, mae, corr, rmse = random_forest_evaluate(model, X_test, y_test)

            year_test = np.unique(years_in_data[test_idx])
            for y, test_year in enumerate(year_test):
                year_idx = np.where(np.unique(years_in_data) == test_year)[0][0]
                model_scores_mae[i, year_idx, j] = mae
                model_scores_corr[i, year_idx, j] = corr
                model_scores_rmse[i, year_idx, j] = rmse

            # Stockage des prédictions
            predictions_by_delay[i, test_idx, j] = y_pred

    save_models(models_dict)  # Sauvegarde globale des modèles

    return model_scores_mae, model_scores_corr, model_scores_rmse, predictions_by_delay

