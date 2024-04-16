import read_wine_data
import wine_classifier
import pandas as pd
import keras
from keras.losses import mean_squared_error
import autoencoder
from sklearn.preprocessing import StandardScaler

def reduce_features_dim(features, encoder=autoencoder.autoencoder):
      
      for n_bottleneck in range(2, 6):
            n_layers = [8, 6, 4] if n_bottleneck < 4 else [8, 7, 6]
            encoder_model, training_model = encoder(n_input=11, 
                                                        n_bottleneck=n_bottleneck, 
                                                        n_layers=n_layers)
            
            training_model.fit(features.values, features.values,
                            epochs=50,
                            batch_size=32,
                            shuffle=True)

            data = encoder_model.predict(features)
            df = pd.DataFrame(data, columns=[f'dim{n}' for n in range(data.shape[1])])
            df.to_csv(f'data/dimensionality_reduced_wine_{n_bottleneck}.csv', index=False)



if __name__ == '__main__':
    

    df = read_wine_data.read('winequality-red.csv')
    df_features = read_wine_data.read('data/dimensionality_reduced_wine_2.csv')

    #Defining features and labels
    labels = df['quality']
    features = df[[col for col in df.columns if col != 'quality']]
    reduce_features_dim(features)

    #Scaling orignial features
    sc = StandardScaler()
    features = sc.fit_transform(features)

    #Instantiating classifiers
    classifier = wine_classifier.WineClassifier('data', 'xgb')
    classifier_original = wine_classifier.WineClassifier('data', 'xgb')

    #Fitting classifier on two dimensionalities
    classifier.train(features=df_features, labels=labels, scaling='Reduced_Dim')
    classifier_original.train(features=features, labels=labels, scaling='Min/Max')

    print(f'Classifier Metadata: {classifier.metadata}')
    print(f'Classifier Original Metadata: {classifier_original.metadata}')


    #NOTE, when predicting encoder bottleneck outputs (reduced data), pass in original features not scaled features