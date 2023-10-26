import pandas as pd


def scoring_function(model_name, time, silh, distortion, db_score):
    """
    Agrège les différentes metrics appliqué au modèle de ML pour permettre de les comparer et choisir le meilleur modèle

    Arguments:
        modele_name : nom du modèle de clustering
        time : temps de traitement du modèle
        silh : silhouette score (permet d'évaluer la qualité du clustering, la cohérence intra-cluster)
        distortion : coefficient de distortion (mesure la variation totale à l'intérieur des clusters)
        db_score : davies-bouldin (compare la distance moyenne entre les points d'un cluster 
                                    avec la distance entre les centres de clusters)

    Return : 
        array() : ajoute les valeurs à un tableau
    """
    scoring_tab = {'Model_name': [model_name], 'Time': [time], 'Silhouette_score' : [silh], 'Coef_distortion' : [distortion], 'Davies-bouldin' : [db_score]}
    scoring_tab = pd.DataFrame(scoring_tab)

    return scoring_tab

