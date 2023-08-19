from itertools import product
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from _load_data import *
from _engineer_features import *
from _build_model import *

import uvicorn
from fastapi import FastAPI
app = FastAPI()


def _get_top_n_menus(df_predicted, df_menu, user_id, n=10):
    """Return top n menu based on rating.

    Args:
        df_predicted (DataFrame): _description_
        df_menu (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    menu_cols = ['menu_id', 'menu_name', 'brand',
                 'menu_category', 'menu_category_detail']
    
    predicted_menus = (
        df_predicted
        .query('user_id == @user_id')
        .head(n)
        .merge(df_menu, on='menu_id')
        .loc[:, menu_cols]
        .to_dict(orient='records')
        )
    
    if len(predicted_menus) > 0:
        response = {
            'user_id': user_id,
            'predicted_menus': predicted_menus,
        }
    else:
        response = {
            'user_id': user_id,
            'predicted_menus': 'Insufficient data to make prediction.',
        }    
    
    return response


@app.get("/api/v1/rec/user/{user_id}")
def get_top_n_menus(user_id, n=10):
    return _get_top_n_menus(df_predicted, df_menu, user_id, n=n)


if __name__ == "__main__":
    df_user, df_menu, df_sales, df_bridge = load_data()
    df_user, df_menu, df_sales, df_bridge, df_rating = engineer_features(
        df_user, df_menu, df_sales, df_bridge
        )
    
    # Caching
    try:
        df_predicted = pd.read_excel('data/df_predicted.xlsx')
    except FileNotFoundError:
        (df_user_modeling, df_menu_modeling,
        input_user, input_menu, y) = prepare_data(df_user, df_menu, df_rating)
        model = build_model(input_user, input_menu, y)
        df_predicted = return_prediction(model,
                                         df_user_modeling, df_menu_modeling)
        df_predicted.to_excel('data/df_predicted.xlsx', index=False)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    