import geopandas as gpd

def clip_points(input_gdf,boundary_gdf):
    '''
    HR Addition: Adding clip function to clip point features to Boston boundaries, replacing the need to manually identify erroneous locations.
    Returns a clipped geodataframe.

    input_gdf (geodataframe): Input geodataframe to be clipped by df_boston_boundary. This should already be in ESPG 4326.
    '''
    clipped_df = gpd.clip(input_gdf,boundary_gdf,keep_geom_type=True)
    clipped_df.reset_index(drop=True,inplace=True)

    dropped_points = input_gdf.loc[~(input_gdf.index.isin(clipped_df.index))]

    print(f'Dropped {dropped_points.shape[0]} points outside of Boston, out of {input_gdf.shape[0]} total points.\nReturning {clipped_df.shape[0]} remaining points.')
    
    return clipped_df