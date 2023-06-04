
# example of mapping_dict: {'high': 3, 'medium': 2, 'low': 1}
def map_order_column(df, column_name, mapping_dict):
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].map(mapping_dict)
    return df_copy


