import numpy as np
import pandas as pd
from _load_data import *


def engineer_features(df_user, df_menu, df_sales, df_bridge):
    """Engineer potential features for the model.

    Returns:
        Tuple of DataFrames: DataFrames of users, menu, sales, bridge,
        and rating.
    """    
    # avg_trx_interval
    df_trx_interval = (
        df_sales
        .merge(
            df_bridge[['sales_id','user_id']].drop_duplicates(),
            on='sales_id',
            how='left',
            )
        .sort_values(['user_id','trx_date_detail'])
        .assign(
            # Calculate the interval between transactions
            avg_trx_interval=lambda x: (x
                                        .groupby('user_id')['trx_date_detail']
                                        .diff()),
            )
        .groupby('user_id')
        .agg({'avg_trx_interval': ['mean']})
        .droplevel(1, axis=1)
        .reset_index()
        )
    ## Impute missing time with mean
    df_user = (
        df_user
        .merge(df_trx_interval, on='user_id')
        )
    
    df_user = (
        df_user
        .merge((df_user
                .groupby(['user_tier_level', 'user_gender'])
                .agg({'avg_trx_interval': ['mean']})
                .reset_index()
                .droplevel(1, axis=1)),
                on=['user_tier_level', 'user_gender'],
                suffixes=['','_mean'],
                )
        .assign(avg_trx_interval=lambda x: (
                x['avg_trx_interval']
                .fillna(x['avg_trx_interval_mean'])
                ))
        .drop('avg_trx_interval_mean', axis=1)
        )
    
    # Rating as a function of quantity and time since last purchase
    df_rating = (
        df_bridge
        .merge(df_user, on='user_id')
        .merge(df_sales, on='sales_id')
        .groupby(['user_id', 'menu_id'])
        .agg({'quantity': 'sum',
                # Time since last order
                'trx_date_detail': lambda x: NOW - x.max()})
        .reset_index()
        .merge(df_user, on='user_id')
        .assign(
                rating=(
                    # Do for each user
                    lambda x: x.apply(
                            lambda y: (
                                y['quantity']
                                # Time since last purchase is short
                                if y['trx_date_detail']<y['avg_trx_interval']
                                # or quantity is more than 1
                                        or y['quantity']>1
                                else -y['quantity']
                                ),
                            axis=1
                            )
                    )
                )
        .loc[:, ['user_id', 'menu_id', 'rating']]
        )


    # last_sold
    df_menu = (
        df_menu.merge(
            df_sales
            .merge(
                df_bridge[['sales_id','menu_id']].drop_duplicates(),
                on='sales_id',
                how='left',
                )
            .groupby(['menu_id'])
            .agg({'trx_date_detail': lambda x: NOW - x.max()})
            .reset_index()
            .rename(columns={'trx_date_detail': 'last_sold'}),
            on='menu_id',
            )
        )
    
    df_sales_interval = (
        df_sales
        .merge(
            df_bridge[['sales_id','menu_id']].drop_duplicates(),
            on='sales_id',
            how='left',
            )
        .sort_values(['menu_id','trx_date_detail'])
        .assign(
            # Calculate the interval between transactions
            avg_sales_interval=lambda x: x.groupby('menu_id')['trx_date_detail'].diff(),
            )
        .groupby('menu_id')
        .agg({'avg_sales_interval': ['mean']})
        .droplevel(1, axis=1)
        .reset_index()
        )
    
    df_menu = (
        df_menu
        .merge(df_sales_interval, on='menu_id')
        )
    
    df_menu = (
        df_menu
        .merge((df_menu
                .groupby(['brand'])
                .agg({'avg_sales_interval': ['mean']})
                .reset_index()
                .droplevel(1, axis=1)),
                on=['brand'],
                suffixes=['','_mean'],
                )
        .assign(avg_sales_interval=lambda x: (
                x['avg_sales_interval']
                .fillna(x['avg_sales_interval_mean'])
                ))
        .drop('avg_sales_interval_mean', axis=1)
        )
    
    return df_user, df_menu, df_sales, df_bridge, df_rating


if __name__ == '__main__':
    df_user, df_menu, df_sales, df_bridge, df_rating = engineer_features(*load_data())