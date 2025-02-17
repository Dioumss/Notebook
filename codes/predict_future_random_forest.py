#Pour faire des previsions dans le future, vous pouvez donner en entrain name qui corresponde aux delais, #space correspondant a len(lat*lon) et X_future tes donnees qui correspondent ici aux signaux #intrasaisonniere.
def predict_future(name, space, X_future):
    """
    Utilise un modèle sauvegardé pour faire une prévision sur de nouvelles données.
    """
    model = load_model(name, space)
    if model is None:
        print(f"Aucun modèle trouvé pour {name} à l'espace {space}.")
        return None
    return model.predict(X_future)
