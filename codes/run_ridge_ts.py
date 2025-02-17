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
            best_model, best_params = get_best_ridge(X_train, y_train, param_grid)
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
        
