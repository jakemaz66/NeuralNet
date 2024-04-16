import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import numpy
import os
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#Camel Case because it's a class
class WineClassifier():

    def __init__(self, model_dir, model):
        """
        Constructor

        Args:
        model is the type of classifier we are loading
        model_dir is the directory in which to save the model
        """

        #Initialize model in __init__
        self.model = self._initialize_xgb_model()
        self.metadata = {}
        self.model_dir = model_dir

    def train(self, features: pd.DataFrame, labels: pd.Series, scaling:str):
        """
        Train the classifier on the data

        Args:
        features: The feature columns of the dataset
        labels: The column that is to be predicted
    
        """

        #Test size can go down as training data goes up
        features, features_test, labels, labels_test = train_test_split(features, labels, test_size=0.2)


        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels_test = le.transform(labels_test)

        #Guranteeing labels are floating point
        self.model.fit(features, labels)

        self.metadata['training_date'] = datetime.datetime.now().strftime('%Y%m%d')
        self.metadata['training_rows'] = len(labels)

        self.metadata['accuracy'] = self.assess(features_test, labels_test)
        self.metadata['scaling'] = scaling
     

    
    def predict(self, features: pd.DataFrame, proba: bool = False) -> numpy.ndarray:
        """
        Model predicts on the test_data

        Args:
        features: the features of the dataset
        proba: whether to return probabilities

        Returns:
        numpy array 
        """
        if len(self.metadata) == 0:
            raise ValueError("Model must be trained first")

        if proba:
             return self.model.predict_proba(features)[:, 0]
        return self.model.predict(features)
    
    def save(self, file_name: str, overwrite: bool = False):
        """
        Save to location path on hard drive

        Args:
        file_name: the name of the file to save
        overwrite: a boolean to check for permission to overwrite an existing file_name
        """

        if len(self.metadata) == 0:
            raise ValueError("Model must be trained before saving")
        
        now = datetime.datetime.now().strftime('%Y%m%d')
        
        if file_name[:6] != now:
            filename = f'{now}_{file_name}'

        #Check for correct file suffix
        if os.path.splitext(filename)[1] != '.json':
            file_name = file_name + '.json'
        
        #Pickle is dangerous because it depends on correct versioning
            
        path = os.path.join(self.model_dir, file_name)
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'
            
        if not overwrite and (os.path.exists(path) or 
                              os.path.exists(metadata_path)):
            raise FileExistsError('Cannot overwrite existing file')
                            
        self.model.save_model(path)

        with open(metadata_path, 'w') as fo:
            json.dump(self.metadata, fo)


    def load(self, file_name):
        """
        load in a model filename with associated metadata from model_dir

        Args:
        file_name: name of the model file
        """

        path = os.path.join(self.model_dir, file_name)
        metadata_path = os.path.splittext(path)[0] + '_metadata.json'

        self.model.load_model(path)
        with open(metadata_path) as fo:
            json.dump(self.metadata, fo)
        

    def assess(self, features, labels) -> float:
        """
        Returns the accuracy of the model

        Args:
        features: The feature columns of the dataset
        labels: The column that is to be predicted

        Returns:
        Accuracy of model chosen metric
        
        """
        

        pred_labels = self.predict(features)
        return accuracy_score(labels, pred_labels)

        
    
    def _initialize_xgb_model(self):
        """Create a new xgbclassifier"""
        return xgb.XGBClassifier()
    