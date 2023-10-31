import pandas as pd


def CreateScoreTab():
    global_scoring = {'Model_name': None, 'Time': None,
                      'Silhouette_score': None,
                      'Coef_distortion': None, 'Davies-bouldin': None}
    global_scoring = pd.DataFrame([global_scoring], index=[0])
    global_scoring.drop(global_scoring.index[0], inplace=True)
    global_scoring_csv = '../Data/0.global_scoring.csv'
    global_scoring.to_csv(global_scoring_csv, index=True)
    return


def AllScore_function(scoring_function):
    """
    Agrège les différentes metrics appliqué au modèle de ML
    pour permettre de les comparer et choisir le meilleur modèle

    Arguments:
        scoring_function : dataframe à agréger au dataframe global

    Return :
    """
    df = pd.read_csv('../Data/0.global_scoring.csv', index_col=0)
    df = df.reset_index(drop=True)
    scoring_function = scoring_function.reset_index(drop=True)
    df = pd.concat([df, scoring_function], ignore_index=True)
    df_csv = '../Data/0.global_scoring.csv'
    df.to_csv(df_csv, index=False)
    return


def scoring_function(model_name, time, silh, distortion, db_score):
    """
    Agrège les différentes metrics appliqué au modèle de ML
    pour permettre de les comparer et choisir le meilleur modèle

    Arguments:
        modele_name : nom du modèle de clustering
        time : temps de traitement du modèle
        silh : silhouette score (permet d'évaluer
            la qualité du clustering, la cohérence intra-cluster)
        distortion : coefficient de distortion
            (mesure la variation totale à l'intérieur des clusters)
        db_score : davies-bouldin (compare la distance
            moyenne entre les points d'un cluster
            avec la distance entre les centres de clusters)

    Return :
        array() : ajoute les valeurs à un tableau
    """
    scoring_tab = {'Model_name': [model_name], 'Time': [time],
                   'Silhouette_score': [silh], 'Coef_distortion': [distortion],
                   'Davies-bouldin': [db_score]}
    scoring_tab = pd.DataFrame(scoring_tab)
    return scoring_tab
