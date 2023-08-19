from itertools import product
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from _engineer_features import *


TIME_UNIT = pd.to_timedelta(1, unit='W')


def prepare_data(df_user, df_menu, df_rating):    
    """Convert the DataFrames into numpy arrays for modeling.

    Args:
        df_user (DataFrame): _description_
        df_menu (DataFrame): _description_
        df_rating (DataFrame): _description_

    Returns:
        Tuple of numpy arrays: _description_
    """
    user_cols = [
    'user_id',
    'user_tier_level',
    'user_gender',
    'time_since_created',
    'avg_trx_interval',
    ]
    
    df_user_modeling = (
        df_user
        .loc[:, user_cols]
        .set_index('user_id')
        .assign(
            user_gender_M=lambda x: x['user_gender'].map({'M': 1, 'F': 0}),
            user_tier_level=lambda x: x['user_tier_level'].cat.codes,
            time_since_created=lambda x: x['time_since_created'] / TIME_UNIT,
            avg_trx_interval=lambda x: x['avg_trx_interval'] / TIME_UNIT,
            )
        .drop('user_gender', axis=1)
        )

    menu_cols = [
        'menu_id',
        'brand',
        'menu_category',
        'menu_category_detail',
        # 'menu_name',
        'last_sold',
        'avg_sales_interval',
    ]

    df_menu_modeling = (pd.get_dummies(
        df_menu
        .loc[:, menu_cols]
        .set_index('menu_id')
        )
        .assign(
            last_sold=lambda x: x['last_sold'] / TIME_UNIT,
            avg_sales_interval=lambda x: x['avg_sales_interval'] / TIME_UNIT,
            )
    )
    
    input_user = (
        df_rating
        .set_index('user_id')
        .join(df_user_modeling)
        .drop(columns=['menu_id', 'rating'])
        .to_numpy()
        )

    input_menu = (
        df_rating
        .set_index('menu_id')
        .join(df_menu_modeling)
        .drop(columns=['user_id', 'rating'])
        .to_numpy()
        )

    y = df_rating['rating'].to_numpy()
    
    return df_user_modeling, df_menu_modeling, input_user, input_menu, y
    

def build_model(input_user: np.ndarray, input_menu: np.ndarray, y: np.ndarray):    
    (user_train, user_test,
    menu_train, menu_test,
    y_train, y_test) = train_test_split(input_user,
                                        input_menu,
                                        y,
                                        test_size=0.2,
                                        random_state=42)
    
    num_outputs = 32
    tf.random.set_seed(42)
    
    num_user_features = input_user.shape[1]
    num_menu_features = input_menu.shape[1]

    user_NN = tf.keras.models.Sequential([     
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_outputs),  
    ])

    menu_NN = tf.keras.models.Sequential([     
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_outputs),  
    ])

    # create the user input and point to the base network
    input_user = tf.keras.layers.Input(shape=(num_user_features))
    vu = user_NN(input_user)
    # vu = tf.linalg.l2_normalize(vu, axis=1)

    # create the menu input and point to the base network
    input_menu = tf.keras.layers.Input(shape=(num_menu_features))
    vm = menu_NN(input_menu)
    # vm = tf.linalg.l2_normalize(vm, axis=1)

    # compute the dot product of the two vectors vu and vm
    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    # specify the inputs and output of the model
    model = tf.keras.Model([input_user, input_menu], output)
    
    cost_fn = tf.keras.losses.Huber()
    model.compile(optimizer='adam',
                loss=cost_fn)

    model.fit([user_train, menu_train], y_train, epochs=30)
    
    print(model.evaluate([user_test, menu_test], y_test))
    
    return model


def return_prediction(model, df_user_modeling, df_menu_modeling):
    df_user_menu_id = pd.DataFrame(
        product(df_user_modeling.index, df_menu_modeling.index),
        columns=['user_id', 'menu_id'],
        )

    final_input_user = (
        df_user_menu_id
        .merge(df_user_modeling, on='user_id')
        .iloc[:,2:]
        .to_numpy()
        )

    final_input_menu = (
        df_user_menu_id
        .merge(df_menu_modeling, on='menu_id')
        .iloc[:,2:]
        .to_numpy()
        )

    y_pred = model.predict([final_input_user, final_input_menu])
    
    df_predicted = (
        pd.concat(
            [df_user_menu_id, pd.DataFrame(y_pred, columns=['rating'])],
            axis=1,
            )
        .sort_values(by=['user_id', 'rating'], ascending=False)
        .query('rating > 0')
        )
    
    return df_predicted

    
if __name__ == '__main__':
    df_user, df_menu, df_sales, df_bridge = load_data()
    df_user, df_menu, df_sales, df_bridge, df_rating = engineer_features(df_user, df_menu, df_sales, df_bridge)
    input_user, input_menu, y = prepare_data(df_user, df_menu, df_rating)
    model = build_model(input_user, input_menu, y)