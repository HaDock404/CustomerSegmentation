import pandas as pd

#global_scoring = {'Time': None, 'Silhouette_score': None, 'Coef_distortion': None, 'Davies-bouldin': None}
#global_scoring = pd.DataFrame([global_scoring], index=[0])
#global_scoring.drop(global_scoring.index[0], inplace=True)
#global_scoring_csv = '../Data/0.global_scoring.csv'
#global_scoring.to_csv(global_scoring_csv, index=False)

def AllScore_function(scoring_function):
    """
    Agrège les différentes metrics appliqué au modèle de ML pour permettre de les comparer et choisir le meilleur modèle

    Arguments:
        scoring_function : dataframe à agréger au dataframe global

    Return : 
        
    """
    df = pd.read_csv('../Data/0.global_scoring.csv')
    df = df.reset_index(drop=True)
    scoring_function = scoring_function.reset_index(drop=True)
    df = pd.concat([df, scoring_function], ignore_index=True)
    df_csv = '../Data/0.global_scoring.csv'
    df.to_csv(df_csv, index=False)
    return 