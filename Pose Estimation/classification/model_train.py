import pandas as pd
import pickle # Object serialization.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 

def load_dataset(csv_data):
    df = pd.read_csv(csv_data)
    features = df.drop('class', axis=1) 
    target_value = df['class']          
    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.2, random_state=1337)

    return x_train, x_test, y_train, y_test

def evaluate_model(fit_models, x_test, y_test):
    print('\nEvaluate model accuracy:')
    # Evaluate and Serialize Model.
    for key_algo, value_pipeline in fit_models.items():
        y_pred = value_pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)*100
        print(f'Classify algorithm: {key_algo}, Accuracy: {accuracy}%')

if __name__ == '__main__':
    
    dataset_csv_file = './dataset/coords_dataset.csv'
    model_weights = './model_weights/weights_body_language.pkl'

    x_train = load_dataset(csv_data=dataset_csv_file)[0]
    y_train = load_dataset(csv_data=dataset_csv_file)[2]
    x_test = load_dataset(csv_data=dataset_csv_file)[1]
    y_test = load_dataset(csv_data=dataset_csv_file)[3]
    
    pipelines = {
        'lr' : make_pipeline(StandardScaler(), LogisticRegression()),
        'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
    }

    fit_models = {}
    print('Model is Training ....')
    for key_algo, value_pipeline in pipelines.items():
        model = value_pipeline.fit(x_train, y_train)
        fit_models[key_algo] = model
    print('Training done.')

    rf_predict = fit_models['rf'].predict(x_test)

    # Save model weights.
    with open(model_weights, 'wb') as f:
        pickle.dump(fit_models['rf'], f)
    print('\nSave model done.')
    
    evaluate_model(fit_models, x_test, y_test)
  