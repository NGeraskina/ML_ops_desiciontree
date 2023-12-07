import joblib
import pandas as pd
from train import prepare_data

def predict(X):
    filename = 'finalized_model.sav'
    loaded_model = joblib.load(filename)
    result = loaded_model.predict(X)
    return result


if __name__ == '__main__':
    # X = pd.DataFrame(dict(zip(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug'], [23, 'F', 'HIGH', 'HIGH', 25.355, 'drugY'])), index = [0])
    X, y = prepare_data()
    print(X[-2:])
    print(predict(X[-10:]))

