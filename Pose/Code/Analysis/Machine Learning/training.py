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
    columns_removed = [
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 
            'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
            'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11',
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
            'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31',
            'v32', 'v33']

    df.drop(columns_removed, axis = 'columns')
    target_value = df['class']          
    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.2, random_state=1337, shuffle=True)

    return x_train, x_test, y_train, y_test

def evaluate_model(fit_models, x_test, y_test):
    print('\nEvaluate model accuracy:')
    # Evaluate and Serialize Model.
    for key_algo, value_pipeline in fit_models.items():
        y_pred = value_pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)*100
        print(f'Classify algorithm: {key_algo}, Accuracy: {accuracy}%')

if __name__ == '__main__':
    
    dataset_csv_file = './dataset/coords_dataset_classification.csv'
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
  