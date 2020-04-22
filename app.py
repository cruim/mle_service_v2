from fastapi import FastAPI, HTTPException
import numpy as np
from catboost import CatBoostClassifier
import pickle
from pytorch_model import Net
import pandas as pd
import torch
from torch.autograd import Variable
from validate import ModelSchema


app = FastAPI(title='Test')


CATBOOST_MODEL = CatBoostClassifier().load_model(fname="models/catboost_model")
GRADIENT_BOOSTING_CLASSIFIER_MODEL = pickle.load(open("models/gradient_boosting_classifier_model.dat", "rb"))
PYTORCH_MODEL = Net()
MODEL_MAPPING = {"001": CATBOOST_MODEL, "002": GRADIENT_BOOSTING_CLASSIFIER_MODEL, "003": PYTORCH_MODEL}



@app.get(path='/', response_description='Test')
async def get():
    try:
        phrase = 'test42'
    except IndexError:
        raise HTTPException(404, "Phrase list is empty")
    return phrase


@app.post(path='/api/predict', response_description="Predict about passenger surviving")
async def post_predict(input: ModelSchema):
    return predict(input.dict())


def prepare_data(input: dict) -> dict:
    del input['name']
    input['sex'] = np.where(input['sex'] == 'female', 1, 0).item(0)
    input['age'] = convert_passenger_age(input)
    input['embarked'] = convert_passenger_embarked(input)
    input['fare'] = convert_passenger_fare(input)
    return format_keys(input)


def convert_passenger_fare(input: dict) -> int:
    fare = int(input['fare'])
    if fare <= 17:
        return 0
    elif 17 < fare <= 30:
        return 1
    elif 30 < fare <= 100:
        return 2
    else:
        return 3


def convert_passenger_embarked(input: dict) -> int:
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    return embarked_map[input['embarked']]


def convert_passenger_age(input: dict) -> int:
    age = int(input['age'])
    if age <= 15:
        return 0  # Дети
    elif 15 < age <= 25:
        return 1  # Молодые
    elif 25 < age <= 35:
        return 2  # Взрослые
    elif 35 < age <= 48:
        return 3  # Средний возраст
    else:
        return 4  # Пожилые


# Приведение ключей к формату модели
def format_keys(input: dict) -> dict:
    mapping = {'pclass': 'Pclass',
               'sex': 'Sex',
               'sibsp': 'SibSp',
               'parch': 'Parch',
               'embarked': 'Embarked',
               'fare': 'Fare',
               'age': 'Age'}
    for key, value in mapping.items():
        input[value] = input.pop(key)
    return input


def predict(input: dict) -> list:
    result = []
    df = pd.DataFrame.from_dict([prepare_data(input['data'])], orient='columns')
    for mod in input['models']:
        model = MODEL_MAPPING.get(mod, False)
        if model and mod == '003':
            model.load_state_dict(torch.load('models/torch.pth'))
            X_test = df.iloc[:, :].values
            with torch.no_grad():
                t_ = model(Variable(torch.FloatTensor(X_test.astype(int)), requires_grad=False))
                value = torch.max(t_, 0)
                result.append({"model_id": mod, "value": value[1].data.numpy().item(0), "result_code": 0})
        elif model:
            result.append({"model_id": mod, "value": model.predict(df).item(0), "result_code": 0})
        else:
            result.append({"model_id": mod, "error": "Model not found", "result_code": 1})
    return result
