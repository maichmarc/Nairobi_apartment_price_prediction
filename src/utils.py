def location_fill(df1, df2, col_a, col_b):
    location_list = list(df2[col_b])
    list_location = [
        next((item1 for item1 in location_list if item1 in item2), item2) 
        for item2 in list(df1[col_a])
    ]
    df3 = df1.copy()
    df3[col_b] = list_location
    return df3