import numpy as np
import pandas as pd

# See Sales for detail
# NOW = pd.Timestamp.now()
NOW = pd.Timestamp(2023,4,1)


def load_data(path: str = 'data/data.csv'):
    """Load and clean the data.

    Args:
        path (str, optional): Path to the data. Defaults to 'data/data.csv'.

    Returns:
        Tuple of dataframes: Dataframe of users, menu, sales, and connector
        between the three (bridge).
    """
    df = (
        pd.read_csv(path)
        # .assign(
        #     trx_date=lambda x: pd.to_datetime(x['trx_date']),
        #     trx_date_detail=lambda x: pd.to_datetime(x['trx_date_detail']),
        # )
    )
    
    cols_user = ['user_id', 'user_created_at', 'user_tier_level', 'user_gender']
    df_user = (
        df.loc[:, cols_user]
        .drop_duplicates(subset=['user_id'])
        .reset_index(drop=True)
        .assign(user_created_at=lambda x: pd.to_datetime(x['user_created_at']),
                time_since_created=lambda x: (NOW - x['user_created_at']),
                user_tier_level=lambda x: pd.Categorical(
                    x['user_tier_level'],
                    # See below for the order of the categories
                    categories=['Basic', 'Silver', 'Gold', 'Diamond', 'Black'],
                    ordered=True
                    ),
                user_gender=lambda x: x['user_gender'].astype('category'),
                )
        )
    
    cols_menu = ['menu_id', 'concept', 'brand', 'menu_category',
       'menu_category_detail', 'menu_name']
    excluded_category_details = {
        "BEVERAGE MODIFIER",
        "FOOD MODIFIER",
        "FOOD ADDITIONAL",
        "BEVERAGE ADDITIONAL",
        }
    excluded_categories = {
        "OTHER",
        }
    menu_category_detail_mapper = {
        'APPETIZERS': 'APPETIZER',
        'APPETIZER': 'APPETIZER',
        'BEER': 'BEER',
        'BEER PROMO': 'BEER PROMO',
        'BEVERAGE PROMO': 'BEVERAGE PROMO',
        'BLENDED': 'BLENDED',
        'CHAMPAGNE': 'CHAMPAGNE',
        'CIGARETTE': 'CIGARETTE',
        'COCKTAIL BEER': 'COCKTAIL',
        'CLASSIC COCKTAIL': 'COCKTAIL',
        'SIGNATURE COCKTAIL': 'COCKTAIL',
        'SPECIAL COCKTAIL': 'COCKTAIL',
        'COGNAC': 'COGNAC',
        'DESSERT': 'DESSERT',
        'CAKE': 'DESSERT',
        'DESSERTS': 'DESSERT',
        'SIGNATURE FLAVOURED': 'DRINKS',
        'BOTTLED DRINKS': 'DRINKS',
        'MILK & CHOCOLATE': 'DRINKS',
        'FOOD PROMO': 'FOOD PROMO',
        'GIN': 'GIN',
        'KIDS MEAL': 'KIDS MEAL',
        'LIQUOR': 'LIQUOR',
        'VODKA': 'LIQUOR',
        'TEQUILA': 'LIQUOR',
        'SAKE & SOJU': 'LIQUOR',
        'APERITIF & DIGESTIF': 'LIQUOR',
        'FREE FLOW GIN & TONIC': 'LIQUOR',
        'RICE & NOODLE': 'MAIN',
        'RICE': 'MAIN',
        'PRIME BEEF CUTS': 'MAIN',
        'ALL DAY BREAKFAST': 'MAIN',
        'MAINS': 'MAIN',
        'JOSPER GRILL': 'MAIN',
        'LAND & SEA': 'MAIN',
        'MAIN': 'MAIN',
        'TEISHOKU': 'MAIN',
        'FROM THE WOK': 'MAIN',
        'SEJUICEME': 'MOCKTAIL & JUICE',
        'MOCKTAIL & JUICE': 'MOCKTAIL & JUICE',
        'MOCKTAIL & JUICES': 'MOCKTAIL & JUICE',
        'DJAMU DJAMU BY DJOURNAL': 'MOCKTAIL & JUICE',
        'MOCKTAILS': 'MOCKTAIL & JUICE',
        'MOCKTAIL': 'MOCKTAIL & JUICE',
        'SASHIMI': 'NIGIRI, SUSHI & SASHIMI',
        'NIGIRI': 'NIGIRI, SUSHI & SASHIMI',
        'SUSHI ROLL': 'NIGIRI, SUSHI & SASHIMI',
        'PASTA': 'PASTA',
        'PASTRY & SANDWICH': 'PASTRY & SANDWICH',
        'NY STYLE GIANT PIZZA': 'PIZZA',
        'CLASSIC CRUST PIZZA': 'PIZZA',
        'THIN CRUST PIZZA': 'PIZZA',
        'PIZZA': 'PIZZA',
        'DRY RAMEN': 'RAMEN',
        'PREMIUM TOKYO BELLY RAMEN': 'RAMEN',
        'RAMEN': 'RAMEN',
        'SIGNATURE BROTH RAMEN': 'RAMEN',
        'CLASSIC TORI PAITAN RAMEN': 'RAMEN',
        'RUM': 'RUM',
        'SET MENU': 'SET MENU',
        'SNACK BAR': 'SIDE DISH',
        'GRAB N GO': 'SIDE DISH',
        'SIDES': 'SIDE DISH',
        'LITE BITES': 'SIDE DISH',
        'SOUP & SALAD': 'SOUP & SALAD',
        'SALAD': 'SOUP & SALAD',
        'SPECIALS MENU': 'SPECIAL MENU',
        'SPECIAL MENU': 'SPECIAL MENU',
        'SPIRIT PROMO': 'SPIRIT PROMO',
        'TEA & COFFEE': 'TEA & COFFEE',
        'FRESHLY BREWED BY DJOURNAL': 'TEA & COFFEE',
        'BREW AT HOME': 'TEA & COFFEE',
        'KOPI NUSANTARA': 'TEA & COFFEE',
        'KOPI BATAVIA BY DJOURNAL': 'TEA & COFFEE',
        'HANDBREWED COFFEE': 'TEA & COFFEE',
        'FRESHLY BREWED BY DJOURNAL ': 'TEA & COFFEE',
        'ESPRESSO BASED': 'TEA & COFFEE',
        'TEA': 'TEA & COFFEE',
        'WATER & SOFT DRINKS': 'WATER & SOFT DRINKS',
        'WATER & SOFTDRINKS': 'WATER & SOFT DRINKS',
        'WATER & SOFTIES': 'WATER & SOFT DRINKS',
        'SINGLE MALT': 'WHISK(E)Y',
        'WHISK(E)Y': 'WHISK(E)Y',
        'WHITE WINE': 'WINE',
        'WINE': 'WINE',
        'RED WINE': 'WINE',
        'SWEET WINE': 'WINE',
        'WINE PROMO': 'WINE PROMO',
        }
    df_menu = (
        df.loc[:, cols_menu]
        .drop_duplicates(subset=['menu_id'])
        .reset_index(drop=True)
        .assign(
            concept=lambda x: x['concept'].astype('category'),
            brand=lambda x: x['brand'].astype('category'),
            menu_category=lambda x: x['menu_category'].astype('category'),
            menu_name=lambda x: (x['menu_name']
                                .str.lower()
                                .str.strip()
                                .str.replace('"', "")
                                .str.replace("'", "")),
            )
        .query('menu_category_detail not in @excluded_category_details and menu_category not in @excluded_categories')
        .assign(menu_category_detail = lambda x: x.menu_category_detail.map(menu_category_detail_mapper))
        .reset_index(drop=True)
        )
    
    cols_sales = ['sales_id','trx_date','trx_date_detail', 'outlet','district',
              'city']
    df_sales = (
        df.loc[:, cols_sales]
        .drop_duplicates(subset=['sales_id'])
        .reset_index(drop=True)
        .assign(
            trx_date=lambda x: pd.to_datetime(x['trx_date']),
            trx_year=lambda x: x['trx_date'].dt.year,
            trx_month=lambda x: x['trx_date'].dt.month,
            trx_day=lambda x: x['trx_date'].dt.day,
            trx_dayofweek=lambda x: x['trx_date'].dt.dayofweek,
            trx_hour=lambda x: pd.to_datetime(x['trx_date_detail']).dt.hour,
            trx_date_detail=lambda x: (pd.to_datetime(x['trx_date_detail'])
                                    .dt.tz_convert(None)),
            district=lambda x: x['district'].astype('category'),
            city=lambda x: x['city'].astype('category'),
            mall=lambda x: (x['outlet']
                            .str.split(', ')
                            .apply(lambda x: x[1])
                            .astype('category')),
            outlet=lambda x: (x['outlet']
                            .str.split(', ')
                            .apply(lambda x: x[0])
                            .astype('category')),
            )
        # .drop(columns=['trx_date_detail'])
        )
    
    cols_bridge = ['sales_id','user_id','menu_id','menu_type','quantity']
    df_bridge = (
        df.loc[:, cols_bridge]
        )

    return df_user, df_menu, df_sales, df_bridge


if __name__ == '__main__':
    df_user, df_menu, df_sales, df_bridge = load_data()