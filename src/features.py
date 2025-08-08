from pathlib import Path
from .config import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED, CRS_WGS84
from .utils import setup_logging, ensure_dir
import argparse


def features():
    # Packages

    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely import wkt
    import re
    import matplotlib.pyplot as plt
    from collections import Counter
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler 

    # Census blocks data and shared variables

    # Data path
    DATA_PATH = r"C:\Users\40009389\Documents\Rodents\data"
    DATA_PATH_2 = r"C:\Users\40009382\Documents\Capstone_repo\data"

    # Coordinate reference systems
    CRS_2249 = "EPSG:2249"
    CRS_4326 = "EPSG:4326"  # lat-lon
    CRS_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

    # Census blocks
    df_blocks = pd.read_csv(DATA_PATH + "\census_blocks.csv")
    df_blocks.insert(0, "geometry", df_blocks['st_astext'].apply(wkt.loads))
    df_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry', crs=CRS_2249)
    df_blocks.to_crs(CRS_4326, inplace=True)


    # Parks data
    df_open = gpd.read_file(DATA_PATH + r"\sidewalk_park_waste\PRK_Open_Space.shp")
    df_open.to_crs(CRS_4326, inplace=True)

    # Park area within block

    intersections = [df_open.intersection(row.geometry) for i, row in df_blocks.iterrows()]  # For each census block, get intersection with each park (EMPTY if no overlap)
    intersections = [i[~i.is_empty].reset_index(drop=True).to_crs(CRS_32619) for i in intersections]  # For each census block, subset to non-empty polygons
    park_areas = [i.area.sum() for i in intersections]  # Get total park area in this census block (in meters^2)
    df_blocks["park_area"] = park_areas

    # Distance to nearest park

    # Convert geometries to projected CRS for measuring distance
    df_open_proj = df_open.to_crs(CRS_32619)
    df_blocks_proj = df_blocks.to_crs(CRS_32619)

    # Get distances from each block to each park, and take the minimum distance
    park_dists = [df_open_proj.distance(row.geometry).min() for i, row in df_blocks_proj.iterrows()]
    df_blocks["park_dist"] = park_dists

    # Types of parks?
    counts = df_open.TypeLong.value_counts().sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(counts.index, counts.values)
    plt.title("Distribution of Park Types")
    plt.xlabel("Counts")
    plt.show()

    # Waste
    df_waste = gpd.read_file(DATA_PATH + r"\sidewalk_park_waste\WasteReceptacles.shp")
    df_waste.to_crs(CRS_4326, inplace=True)

    # Neighborhoods
    df_nbr = pd.read_csv(DATA_PATH + r"\neighborhood_data.csv")
    df_nbr.insert(0, "geometry", df_nbr['st_astext'].apply(wkt.loads))
    df_nbr = gpd.GeoDataFrame(df_nbr, geometry='geometry', crs=CRS_2249)
    df_nbr.to_crs(CRS_4326, inplace=True)

    # Plot
    fig, ax = plt.subplots()
    df_blocks.plot(ax=ax)
    df_waste.plot(ax=ax, color="red", markersize=2)
    plt.title("Waste Receptacle Locations")
    plt.show()

    # Find which neighborhoods don't have waste receptacles data

    # Fix invalid geometries
    invalid_nbr = df_nbr[~df_nbr['geometry'].is_valid]
    df_nbr['geometry'] = df_nbr['geometry'].buffer(0) #This helps deal with self intersections and improper orientations

    # Find names of neighborhoods with 0 receptacles
    intersections_nbr = [df_waste.intersects(row.geometry) for i, row in df_nbr.iterrows()]  # For each neighborhood, get intersection with each waste receptacle location
    n_bins_nbr = np.array([i.sum() for i in intersections_nbr]) #sum number of bins in each neighborhood
    nbr_missing = df_nbr.iloc[np.where(n_bins_nbr == 0)[0]].Name.values #get neighborhood names with 0 receptacles
    nbr_missing = np.append(nbr_missing, ['Hyde Park', 'South Boston', 'West Roxbury']) #add Hyde Park and South Boston

    # Get nbr name for each block
    nbr_block_intersections = [df_nbr.intersection(row.geometry) for i, row in df_blocks.iterrows()]  # For each census block, get corresponding neighborhood
    nbr_block_intersections = [i[~i.is_empty] for i in nbr_block_intersections]  # For each census block, subset to non-empty polygons
    nbr_idx = np.array([i.index[0] for i in nbr_block_intersections])
    nbr_names = [df_nbr.iloc[nbr_id].Name for nbr_id in nbr_idx]
    df_blocks["neighborhood"] = nbr_names

    #boolean for blocks whose neighborhoods are among ones with missing waste data
    boolean_missing = [nbr_name in nbr_missing for nbr_name in nbr_names]

    #Number of waste receptacles in each block

    intersections = [df_waste.intersects(row.geometry) for i, row in df_blocks.iterrows()]  # For each census block, get intersection with each waste receptacle location
    n_bins = [i.sum() for i in intersections] #sum number of bins in each block

    #assign None for blocks with 0 bins and whose whole neighborhood had 0 bins, 0 to other blocks with 0 bins (true zeros)
    n_bins_final = [None if n_bins[i] == 0 and boolean_missing[i] else 0 if n_bins[i] == 0 and not boolean_missing[i] else n_bins[i] for i in range(len(n_bins))]
    df_blocks["n_waste_bins"] = n_bins_final

    print("Locations missing for", round(n_bins_final.count(None)*100/len(n_bins_final),2), "% of the data")

    #Distance to nearest waste receptacle

    # Convert geometries to projected CRS for measuring distance
    df_waste_proj = df_waste.to_crs(CRS_32619)
    df_blocks_proj = df_blocks.to_crs(CRS_32619)

    # Get distances from each block to each waste bin, and take the minimum distance
    waste_dists = [
        0 if df_waste_proj.contains(row.geometry).any() 
        else df_waste_proj.distance(row.geometry).min()
        for i, row in df_blocks_proj.iterrows()
    ]

    #Impute Nones where necessary
    waste_dists = [None if n_bins_final[i] is None else waste_dists[i] for i in range(len(waste_dists))]
    df_blocks["waste_dist"] = waste_dists

    # Read Census Data
    # Dataframe with geometry
    df_census_geometry = pd.read_csv(DATA_PATH + r"\census_tract_geometry.csv")
    df_census_geometry.insert(0, "geometry", df_census_geometry['st_astext'].apply(wkt.loads))
    df_census_geometry = gpd.GeoDataFrame(df_census_geometry, geometry='geometry', crs=CRS_2249)
    df_census_geometry.to_crs(CRS_32619, inplace=True)

    # Dataframe with relevant data
    df_census_info = pd.read_csv(DATA_PATH + r"\census_tract_access_boston.csv")
    df_census_info = df_census_info.iloc[1:, :]

    # Merge datasets & clean relevant variables

    # Convert relevant data to numeric
    # Geocode
    df_census_info.GEOCODE = pd.to_numeric(df_census_info.GEOCODE)
    df_census_geometry.geoid20 = pd.to_numeric(df_census_geometry.geoid20)
    # Tract code
    df_census_info.TRACT = pd.to_numeric(df_census_info.TRACT)
    df_census_geometry.tractce20 = pd.to_numeric(df_census_geometry.tractce20)
    # Population / #houses
    df_census_info.P0020001 = pd.to_numeric(df_census_info.P0020001)
    df_census_info.H0010001 = pd.to_numeric(df_census_info.H0010001)

    # Get area of tract geometry (m^2)
    df_census_geometry['area'] = df_census_geometry.geometry.area

    # Merge on GEOID
    df_census = df_census_info.merge(df_census_geometry, how="right", left_on="GEOCODE", right_on="geoid20")
    df_census = gpd.GeoDataFrame(df_census, geometry="geometry")
    df_census.to_crs(CRS_4326, inplace=True)

    # Impute population / variables for tract 981502 (Boston has no data because entire population was assigned to Roxbury)

    # Get populations / houses from neighboring tracts (510, 511.01)
    pops = df_census[df_census.tractce20.isin([51000, 51101])].P0020001
    bldgs = df_census[df_census.tractce20.isin([51000, 51101])].H0010001

    # Assign avg population to 981502
    idx_981502 = df_census[df_census.tractce20 == 981502].index[0]
    df_census.iloc[idx_981502, 11] = pops.mean()
    df_census.iloc[idx_981502, 39] = bldgs.mean()

    # Get population and building density (# objects / land area per census tract)
    df_census["pop_density"] = df_census.P0020001 / df_census.aland20
    df_census["bldg_density"] = df_census.H0010001 / df_census.aland20

    # Assign blocks to tracts by finding largest intersection between block and tract
 
    intersections = [df_census.intersection(row.geometry) for i, row in df_blocks.iterrows()]   # For each census block, get intersection with each park (EMPTY if no overlap)
    intersections = [i.to_crs(CRS_32619) for i in intersections]  # Change to projected CRS to get area
    idx = [np.argmax(i.area) for i in intersections]  # Get index of tract with max area for this block

    # Get corresponding population/building density for each block
    pop_density = [df_census.pop_density[idx[i]] for i, row in df_blocks.iterrows()]
    bldg_density =[df_census.bldg_density[idx[i]] for i, row in df_blocks.iterrows()]

    # Save
    df_blocks['pop_density'] = pop_density
    df_blocks['bldg_density'] = bldg_density

    # Parcel Data

    df_parcel_raw = gpd.read_file(DATA_PATH + r"\parcel_data\parcel_data.shp")
    df_parcel_raw.to_crs(CRS_4326, inplace=True)

    # # Showing that all condo unit parcels intersect condo main parcels
    # # Get condo unit and condo main parcels
    # res_condo = df_parcel_raw[df_parcel_raw.LU_DESC=="RESIDENTIAL CONDO"]
    # main_condo = df_parcel_raw[df_parcel_raw.LU_DESC=="CONDO MAIN"]

    # # For each condo unit, get number of intersections with condo mains
    # res_main_num_intersects = [sum(main_condo.intersects(x)) for x in res_condo.geometry]

    # # Print results
    # Counter(res_main_num_intersects)

    # Subset to relevant columns & rows

    # Remove rows that are condo main (because all condo mains are made up of RESIDENTIAL CONDOs)
    df_parcel = df_parcel_raw[df_parcel_raw.LU_DESC != "CONDO MAIN"].reset_index(drop=True)

    df_parcel = df_parcel[["PID", "NUM_BLDGS", "LU", "RES_FLOOR", "GROSS_AREA", "LAND_SF", "BLDG_VALUE", "TOTAL_VALU", "YR_BUILT", "YR_REMODEL", "STRUCTURE_", "INT_WALL", 
               "EXT_FNISHE", "OVERALL_CO", "HEAT_TYPE", "geometry"]]

    df_parcel.columns = ["PID", "NUM_BLDGS", "LU", "RES_FLOOR", "GROSS_AREA", "LAND_SF", "BLDG_VALUE", "TOTAL_VALUE", "YR_BUILT", "YR_REMODEL", "STRUCTURE_CLASS", "INT_WALL", 
               "EXT_FINISHED", "OVERALL_COND", "HEAT_TYPE", "geometry"]

    # Clean / create some columns

    # Get parcel area
    df_parcel["AREA"] = df_parcel.to_crs(CRS_32619).geometry.area

    # BLDG_VALUE
    # Make numeric
    df_parcel.BLDG_VALUE = pd.to_numeric([re.sub(r"[^\d]", "", x) for x in df_parcel.BLDG_VALUE])


    # YR_BUILT
    # Fix typos
    idx_built = np.where(df_parcel.YR_BUILT > 2024)[0]
    df_parcel.loc[idx_built[0], "YR_BUILT"] = 2019


    # YR_REMODEL
    # Fix typos
    idx_remodel_big = np.where(df_parcel.YR_REMODEL > 2024)[0]  # After 2024
    df_parcel.loc[idx_remodel_big[0], "YR_REMODEL"] = 2021
    df_parcel.loc[idx_remodel_big[1], "YR_REMODEL"] = 2021
    df_parcel.loc[idx_remodel_big[2], "YR_REMODEL"] = 2022

    idx_remodel_small = np.where((df_parcel.YR_REMODEL <1700) & (df_parcel.YR_REMODEL > 0))[0]  # Before 1700 (first house built)
    df_parcel.loc[idx_remodel_small[0], "YR_REMODEL"] = 2010
    df_parcel.loc[idx_remodel_small[1], "YR_REMODEL"] = 2021
    df_parcel.loc[idx_remodel_small[2], "YR_REMODEL"] = 2010


    # OVERALL_COND
    # Convert some types to standard format
    df_parcel.OVERALL_COND = [None if c is None else re.sub("EX - Excellent", "E - Excellent", c) for c in df_parcel.OVERALL_COND]
    df_parcel.OVERALL_COND = [None if c is None else re.sub("AVG - Default - Average", "A - Average", c) for c in df_parcel.OVERALL_COND]


    # Make Boston College building value None (to remove outlier)
    idx = df_parcel.index[df_parcel.PID=='2102473001'][0]
    df_parcel.loc[idx, 'BLDG_VALUE'] = None

    # # Do the parcels cover Boston? (Yes)

    # fig, ax = plt.subplots(figsize=(5,5))
    # df_blocks.plot(ax=ax, color="red")
    # df_parcel.plot(ax=ax, color="blue")
    # plt.title("Census Blocks (Red) with Parcel Coverage (Blue)")
    # plt.show()

    # LU (Land Use) (Sorting into fewer categories)

    # Group into smaller categories (residential, commercial, r/c, etc.)

    mapping = {'A': 'R',    # R - Residential
               'R1': 'R', 
               'R2': 'R',
               'R3': 'R',
               'R4': 'R', 
               'RL - RL': 'R',
               'CD': 'CO',  # CO - Condos
               'CM': 'CO',
               'CP': 'CO',
               'C': 'C',    # C - Commercial
               'CC': 'C',
               'CL': 'C',
               'RC': 'RC',  # RC - Mixed residential/commercial
               'I': 'I',    # I - Industrial
               'AH': 'AH',  # AH - Agricultural
               'E': 'E',    # E - Tax Exempt
               'EA': 'E'}

    # Assign new categories
    LU_new = df_parcel['LU'].replace(mapping)
    df_parcel['LU_new'] = LU_new

    # One-hot-encode
    one_hot_encoded = pd.get_dummies(df_parcel['LU_new'], prefix='LU')

    # Add to dataframe
    df_parcel = pd.concat([df_parcel, one_hot_encoded], axis=1)


    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    counts = df_parcel.LU.value_counts().sort_values()
    ax1.barh(counts.index, counts.values)
    ax1.set_title("Distribution of LU")
    ax1.set_xlabel("Counts")

    counts = df_parcel.LU_new.value_counts().sort_values()
    ax2.barh(counts.index, counts.values)
    ax2.set_title("Distribution of LU Clustered")
    ax2.set_xlabel("Counts")
    plt.show()

    # RES_FLOOR (Number of residential floors) (Maybe not so useful)

    # Plot distribution
    plt.figure(figsize=(4,3))
    plt.hist(df_parcel.RES_FLOOR, bins=100)
    plt.title("Distribution of Residential Floors")
    plt.show()

    # Description
    print("Total observations:", df_parcel.shape[0])
    print("Number of NA:", sum(pd.isna(df_parcel.RES_FLOOR)))
    print("\nSummary:")
    df_parcel.RES_FLOOR.describe()


    # BLDG_VALUE (Building Value) (Lots of 0's?)

    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
    ax1.hist(df_parcel.BLDG_VALUE, bins=30)
    ax1.set_title("Building Value")
    tmp = df_parcel[(df_parcel.BLDG_VALUE!=0) & (df_parcel.BLDG_VALUE < 10000000)]
    ax2.hist(tmp.BLDG_VALUE, bins=30)
    ax2.set_title("Nonzero Building Value Below 1,000,000")
    plt.show()

    # Description
    print("Total observations:", df_parcel.shape[0])
    print("Number of NA:", sum(pd.isna(df_parcel.BLDG_VALUE)))
    print(f"Number of 0: {sum(df_parcel.BLDG_VALUE==0)} ({sum(df_parcel.BLDG_VALUE==0)/df_parcel.shape[0]})")
    print("\nSummary:")
    df_parcel.BLDG_VALUE.describe()

    # Replace 0 with None
    # df_parcel.loc[:, 'BLDG_VALUE'] = [None if x==0 else x for x in df_parcel.BLDG_VALUE]

    # YR_BUILT, YR_REMODEL (Calculating age and years since remodeled)

    # Calculate age
    ages = [None if yr==0 else 2024-yr for yr in df_parcel.YR_BUILT]
    df_parcel["AGE"] = ages


    # Calculate age since remodel (taking YR_BUILT as YR_REMODEL if YR_REMODEL=0)
    yr_remodel = df_parcel.YR_REMODEL
    yr_built = df_parcel.YR_BUILT

    yr_remodel = [yr_built[i] if yr_remodel[i]==0 else yr_remodel[i] for i in range(len(yr_remodel))]
    yrs_since_remodel = [None if yr==0 else 2024-yr for yr in yr_remodel]
    df_parcel["YRS_SINCE_REMODEL"] = yrs_since_remodel

    # STRUCTURE_CLASS: (Building material) (90% is None)

    # Plots
    counts = df_parcel.STRUCTURE_CLASS.value_counts().sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(counts.index, counts.values)
    plt.title("Distribution of Structure Classes")
    plt.xlabel("Counts")
    plt.show()

    # Description
    print("Total observations:", df_parcel.shape[0])
    print(f"Number of NA: {sum(pd.isna(df_parcel.STRUCTURE_CLASS))} ({round(sum(pd.isna(df_parcel.STRUCTURE_CLASS))/df_parcel.shape[0], 3)})")

    # INT_WALL (Interior wall condition) (Most are Normal)

    # Plots
    counts = df_parcel.INT_WALL.value_counts().sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(counts.index, counts.values)
    plt.title("Distribution of Interior Wall Condition")
    plt.xlabel("Counts")
    plt.show()

    # Description
    print("Total observations:", df_parcel.shape[0])
    print(f"Number of NA: {sum(pd.isna(df_parcel.INT_WALL))} ({round(sum(pd.isna(df_parcel.INT_WALL))/df_parcel.shape[0], 3)})")

    # EXT_FINISHED (Exterior siding material) (Sorting into fewer categories)

    # Group into smaller categories (stone, metal, wood, etc.)

    # Define groupings
    mapping = {'01 - Brick': 'BRICK_STONE',
                '02 - Stone': 'BRICK_STONE',
                '03 - Poured Concr': 'BRICK_STONE',
                '04 - Precast Concr': 'BRICK_STONE',
                '05 - Concr & Glass': 'BRICK_STONE',
                '06 - Metal/Glass': 'METAL_GLASS',
                '07 - Stone/Marble': 'BRICK_STONE',
                '08 - Stucco': 'BRICK_STONE',
                '09 - Wood Siding': 'WOOD',
                '10 - Alum/Vinyl': 'VINYL',
                '11 - Metal Siding': 'METAL_GLASS',
                '12 - Conc Block': 'BRICK_STONE',
                '13 - Br Sill/Sash': 'BRICK_STONE',
                '14 - Hollow Tile': 'METAL_GLASS',
                '15 - Corrug Siding': 'METAL_GLASS',
                'A - Asbestos': 'ASBESTOS',
                'B - Brick/Stone': 'BRICK_STONE',
                'C - Cement Board': 'BRICK_STONE',
                'F - Frame/Clapbrd': 'WOOD',
                'G - Glass': 'METAL_GLASS',
                'K - Concrete': 'BRICK_STONE',
                'M - Vinyl': 'VINYL',
                'O - Other': 'OTHER',
                'P - Asphalt': 'BRICK_STONE',
                'S - Stucco': 'BRICK_STONE',
                'U - Alum Siding': 'METAL_GLASS',
                'V - Brck/Stn Venr': 'BRICK_STONE',
                'W - Wood Shake': 'WOOD'
    
    }
    # Assign new categories
    df_parcel['EXT_new'] = df_parcel['EXT_FINISHED'].replace(mapping)
    # One-hot-encode
    one_hot_encoded = pd.get_dummies(df_parcel['EXT_new'], prefix='EXT')
    # Add to dataframe
    df_parcel = pd.concat([df_parcel, one_hot_encoded], axis=1)


    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    counts = df_parcel.EXT_FINISHED.value_counts().sort_values()
    ax1.barh(counts.index, counts.values)
    ax1.set_title("Distribution of Exterior Siding Material")
    ax1.set_xlabel("Counts")

    counts = df_parcel.EXT_new.value_counts().sort_values()
    ax2.barh(counts.index, counts.values)
    ax2.set_title("Distribution of Exterior Siding Material, Regrouped")
    ax2.set_xlabel("Counts")
    plt.show()

    # Descriptions
    print("Total observations:", df_parcel.shape[0])
    print(f"Number of NA: {sum(pd.isna(df_parcel.EXT_FINISHED))} ({round(sum(pd.isna(df_parcel.EXT_FINISHED))/df_parcel.shape[0], 3)})")


    # OVERALL_COND (Overall parcel condition) (Cleaning, making scores from condition strings)

    # Converting to score
    def get_score_from_condition(condition):
        '''
        Convert string condition into score
        '''
        if condition is None:
            return None

        if condition == "US - Unsound":
            return 0
    
        if condition == "VP - Very Poor":
            return 1
    
        if condition == "P - Poor":
            return 2
    
        if condition == "F - Fair":
            return 3
    
        if condition == "A - Average":
            return 4
    
        if condition == "G - Good":
            return 5
    
        if condition == "VG - Very Good":
            return 6
    
        if condition == "E - Excellent":
            return 7
    
        raise ValueError("Condition not recognized") 

    overall_condition_scores = [get_score_from_condition(c) for c in df_parcel.OVERALL_COND]
    df_parcel["OVERALL_COND_SCORE"] = overall_condition_scores


    # Plots
    counts = df_parcel.OVERALL_COND.value_counts().sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(counts.index, counts.values)
    plt.title("Distribution of Overall Parcel Condition")
    plt.xlabel("Counts")
    plt.show()

    # Description
    print("Total observations:", df_parcel.shape[0])
    print(f"Number of NA: {sum(pd.isna(df_parcel.OVERALL_COND))} ({round(sum(pd.isna(df_parcel.OVERALL_COND))/df_parcel.shape[0], 3)})")

    # HEAT_TYPE (Type of heating system) (Not sure if useful)

    # Plots
    counts = df_parcel.HEAT_TYPE.value_counts().sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(counts.index, counts.values)
    plt.title("Distribution of Heating System Type")
    plt.xlabel("Counts")
    plt.show()

    # Description
    print("Total observations:", df_parcel.shape[0])
    print(f"Number of NA: {sum(pd.isna(df_parcel.HEAT_TYPE))} ({round(sum(pd.isna(df_parcel.HEAT_TYPE))/df_parcel.shape[0], 3)})")

    # Subset df_parcel to relevant columns

    df_parcel = df_parcel.loc[:, [
        'PID',
        'NUM_BLDGS',
        'BLDG_VALUE',
        'AGE',
        'YRS_SINCE_REMODEL',
        'OVERALL_COND_SCORE',
        'LU_AH',
        'LU_C',
        'LU_CO',
        'LU_E',
        'LU_I',
        'LU_R',
        'LU_RC',
        'EXT_ASBESTOS',
        'EXT_BRICK_STONE',
        'EXT_METAL_GLASS',
        'EXT_OTHER',
        'EXT_WOOD',
        'EXT_VINYL',
        'AREA',
        'geometry'
    ]]

    # Join with df_blocks on where intersections occur
    df_parcel_blocks = gpd.sjoin(df_parcel, df_blocks[['gid', 'geometry']], how="left", predicate="intersects")

    # Reduce rows such that there is only one block per parcel

    def get_biggest_gid(group):
        '''
        To be used as df.groupby('PID').apply(get_biggest_id)
        Given a parcel-block dataframe with only one parcel ID, return the block's gid with the largest intersection
        '''
    
        # Store the resulting gid
        result = {}
    
        # If there is only one intersecting block, just return current gid
        if group.shape[0] == 1:
            result['gid'] = group.gid.values[0]
            return pd.Series(result)
    
        # Get geometry of this parcel
        geom = group.geometry.values[0]
    
        # Get blocks that intersect with this parcel
        this_df_blocks = df_blocks[df_blocks.gid.isin(group.gid)].reset_index(drop=True)
    
        # Get gid of maximum-area intersection
        idx = np.argmax(this_df_blocks.intersection(geom).to_crs(CRS_32619).area)
        this_gid = this_df_blocks.gid[idx]

        result['gid'] = this_gid
        return pd.Series(result)

    tqdm.pandas()
    df_parcel_gids = df_parcel_blocks.groupby('PID').progress_apply(get_biggest_gid, include_groups=False).reset_index()
    df_parcel_blocks = df_parcel_blocks.merge(df_parcel_gids, on=['PID', 'gid'], how='inner')

    # Define how to combine parcel data per block (sum, mean, min, etc.)

    def combine_functions(group):
        result = {}
        result['num_bldgs'] = np.sum(group['NUM_BLDGS'])
        result['bldg_value_mean'] = np.mean(group['BLDG_VALUE'])
        result['bldg_value_min'] = np.min(group['BLDG_VALUE'])
        result['age_mean'] = np.mean(group['AGE'])
        result['age_max'] = np.max(group['AGE'])
        result['yrs_since_remodel_mean'] = np.mean(group['YRS_SINCE_REMODEL'])
        result['yrs_since_remodel_max'] = np.max(group['YRS_SINCE_REMODEL'])
        result['overall_cond_score_mean'] = np.mean(group['OVERALL_COND_SCORE'])
        result['overall_cond_score_min'] = np.min(group['OVERALL_COND_SCORE'])
    
        LU_area = sum(group[group.iloc[:, 6:13].any(axis=1)].AREA)
        EXT_area = sum(group[group.iloc[:, 13:18].any(axis=1)].AREA)
    
        if LU_area > 0:
            result['LU_AH'] = sum(group[group.LU_AH].AREA) / LU_area
            result['LU_C'] = sum(group[group.LU_C].AREA) / LU_area
            result['LU_CO'] = sum(group[group.LU_CO].AREA) / LU_area
            result['LU_E'] = sum(group[group.LU_E].AREA) / LU_area
            result['LU_I'] = sum(group[group.LU_I].AREA) / LU_area
            result['LU_R'] = sum(group[group.LU_R].AREA) / LU_area
            result['LU_RC'] = sum(group[group.LU_RC].AREA) / LU_area
        else:
            result['LU_AH'] = 0
            result['LU_C'] = 0
            result['LU_CO'] = 0
            result['LU_E'] = 0
            result['LU_I'] = 0
            result['LU_R'] = 0
            result['LU_RC'] = 0
    
        if EXT_area > 0:
            result['EXT_asbestos'] = sum(group[group.EXT_ASBESTOS].AREA) / EXT_area
            result['EXT_brick_stone'] = sum(group[group.EXT_BRICK_STONE].AREA) / EXT_area
            result['EXT_metal_glass'] = sum(group[group.EXT_METAL_GLASS].AREA) / EXT_area
            result['EXT_other'] = sum(group[group.EXT_OTHER].AREA) / EXT_area
            result['EXT_wood'] = sum(group[group.EXT_WOOD].AREA) / EXT_area
            result['EXT_vinyl'] = sum(group[group.EXT_VINYL].AREA) / EXT_area
        else:
            result['EXT_asbestos'] = 0
            result['EXT_brick_stone'] = 0
            result['EXT_metal_glass'] = 0
            result['EXT_other'] = 0
            result['EXT_wood'] = 0
            result['EXT_vinyl'] = 0
    
        return pd.Series(result)

    df_parcel_blocks = df_parcel_blocks.groupby('gid').apply(combine_functions, include_groups=False).reset_index()
    df_blocks = df_blocks.merge(df_parcel_blocks, how="left", on="gid")

    fig, ax = plt.subplots()

    df_blocks.plot(ax=ax, color="blue")
    df_blocks[pd.isna(df_blocks.LU_CO)].plot(ax=ax, color="red")
    plt.show()

    # What to do to fix this lonely block? It has parcels, but doesn't have the largest intersection with any of those parcels, so it doesn't have any information
    # Should we not force only one block per parcel?
    # Should we put in 0 for counts?

    # SCL Data
    df_scl = pd.read_csv(DATA_PATH + r"\restaurant_scl.csv")
    df_scl.insert(0, "geometry", df_scl['st_astext'].apply(wkt.loads))
    df_scl = gpd.GeoDataFrame(df_scl, geometry='geometry', crs=CRS_4326)

    # Active Food Establishment License Data
    df_active_food_licenses = pd.read_csv(DATA_PATH + r"\ips_active_food_establishment_licenses.csv")
    df_active_food_licenses = gpd.GeoDataFrame(df_active_food_licenses, geometry=gpd.points_from_xy(df_active_food_licenses.longitude, df_active_food_licenses.latitude), crs=CRS_4326)
    df_active_food_licenses = df_active_food_licenses[df_active_food_licenses.longitude > -73]

    # Sanity plot

    fig, ax = plt.subplots(figsize=(5,5))
    df_blocks.plot(ax=ax)
    df_scl.plot(ax=ax, color="red", markersize=2)
    plt.title("Site Cleanliness Licenses (Restaurants)")
    plt.show()

    # Number of licenses
    # Get number of SCL licenses per block
    n_scl = df_blocks.geometry.apply(lambda g: sum(df_scl.intersects(g)))  # For each block, get number of intersections with SCL licenses
    df_blocks['n_scl'] = n_scl


    # Distance to nearest license
    # Get distance to nearest SCL license per block
    # Convert geometries to projected CRS for measuring distance
    df_scl_proj = df_scl.to_crs(CRS_32619)

    # Get distances from each block to each receptacle, and take the minimum distance
    scl_dists = df_blocks_proj.geometry.apply(lambda g: min(df_scl_proj.distance(g)))
    df_blocks["scl_dist"] = scl_dists

    # Number of establishments
    # Get number of food establishment licenses per block
    n_restaurants = df_blocks.geometry.apply(lambda g: sum(df_active_food_licenses.intersects(g)))  # For each block, get number of intersections with food licenses
    df_blocks['n_restaurants'] = n_restaurants


    # Distance to nearest license
    # Get distance to nearest SCL license per block
    # Convert geometries to projected CRS for measuring distance
    df_active_food_licenses_proj = df_active_food_licenses.to_crs(CRS_32619)

    # Get distances from each block to each receptacle, and take the minimum distance
    restaurant_dists = df_blocks_proj.geometry.apply(lambda g: min(df_active_food_licenses_proj.distance(g)))
    df_blocks["restaurant_dist"] = restaurant_dists

    # Averaging?
    df_blocks["n_scl_restaurants_mean"] = (df_blocks.n_scl + df_blocks.n_restaurants)/2
    df_blocks["scl_restaurant_dist_mean"] = (df_blocks.scl_dist + df_blocks.restaurant_dist)/2

    # Make Bar Harbor restaurant distance None (uninformative)
    idx = df_blocks.restaurant_dist.idxmax()
    df_blocks.loc[idx, 'restaurant_dist'] = None

    # Extract feature names
    feature_names = [
        'park_area', 'park_dist', 
        'n_waste_bins', 'waste_dist', 
        'pop_density', 'bldg_density',
        'bldg_value_mean', 'bldg_value_min',
        'age_mean', 'age_max', 'yrs_since_remodel_mean', 'yrs_since_remodel_max', 
        'overall_cond_score_mean', 'overall_cond_score_min', 
        'LU_AH', 'LU_C', 'LU_CO', 'LU_E', 'LU_I','LU_R', 'LU_RC', 
        'EXT_asbestos', 'EXT_brick_stone', 'EXT_metal_glass','EXT_other', 'EXT_wood', 'EXT_vinyl',
        'n_restaurants', 'restaurant_dist']

    # Subset dataset
    df_final = df_blocks[feature_names] 

    # Define standard scaler 
    scaler = StandardScaler() 

    # Transform data 
    df_sampling = pd.DataFrame(scaler.fit_transform(df_final))
    df_sampling.columns = feature_names
    df_sampling["st_astext"] = df_blocks.st_astext

    # Save
    df_final.to_csv('./features.csv', index=False)
    df_sampling.to_csv('./sampling.csv',index=False)


    plotting_cols = feature_names

    for col in plotting_cols:
        fig, ax = plt.subplots(figsize=(5,5))
        df_blocks.plot(ax=ax, column=col, cmap='Reds', edgecolor='black', linewidth=0.05, legend=True)
        ax.set_title(col)
        plt.show()

    block_gid = df_blocks.loc[df_blocks.LU_RC.idxmax(), 'gid']
    parcel_pids = df_parcel_gids[df_parcel_gids.gid==block_gid].PID
    df_parcel_raw[df_parcel_raw.PID.isin(parcel_pids)]
    block_gid

    df_blocks.n_scl.corr(df_blocks.n_restaurants)
    # plt.scatter(df_blocks.n_scl, df_blocks.n_restaurants)

    # Plots of SCL and active food licenses
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

    df_blocks.plot(ax=ax1)
    df_scl.plot(ax=ax1, color="red", markersize=2)
    ax1.set_title("Active Site Cleanliness Licenses")

    df_blocks.plot(ax=ax2)
    df_active_food_licenses.plot(ax=ax2, color="orange", markersize=2)
    ax2.set_title("Active Food Licenses")

    df_blocks.plot(ax=ax3)
    df_scl.plot(ax=ax3, color="red", markersize=2, alpha=0.5)
    df_active_food_licenses.plot(ax=ax3, color="orange", markersize=2, alpha=0.5)
    ax3.set_title("All Licenses")
    plt.show()


    # Number of SCL and Restaurants
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    df_blocks.plot(ax=ax1, column='n_scl', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)
    ax1.set_title("n_scl")

    df_blocks.plot(ax=ax2, column='n_restaurants', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)
    ax2.set_title("n_restaurants")

    df_blocks.plot(ax=ax3, column='n_scl_restaurants_mean', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)
    ax3.set_title("Mean of n_scl and n_restaurants")
    plt.show()


    # Distance to SCL and Restaurants
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    df_blocks.plot(ax=ax1, column='scl_dist', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)
    ax1.set_title("scl_dist")

    df_blocks.plot(ax=ax2, column='restaurant_dist', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)
    ax2.set_title("restaurant_dist")

    df_blocks.plot(ax=ax3, column='scl_restaurant_dist_mean', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)
    ax3.set_title("Mean of scl_dist and restaurant_dist")
    plt.show()


    # 1164 geometries overlap
    df_scl_active = gpd.sjoin(df_scl[['geometry', 'businessname']], df_active_food_licenses[['geometry', 'businessname', 'address']], how="inner", predicate="intersects")
    df_scl_active.shape

    # Plots of SCL and active food licenses
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

    df_blocks.plot(ax=ax1)
    df_scl_active.plot(ax=ax1, color="black", markersize=2)
    ax1.set_title("Overlap")

    df_blocks.plot(ax=ax2)
    df_scl[~df_scl.index.isin(df_scl_active.index)].plot(ax=ax2, color="red", markersize=2)
    ax2.set_title("Non-Overlap SCL")

    df_blocks.plot(ax=ax3)
    df_active_food_licenses[~df_active_food_licenses.index.isin(df_scl_active.index_right)].plot(ax=ax3, color="orange", markersize=2)
    ax3.set_title("Non-Overlap Active Food Licenses")
    plt.show()

    df_parcel.plot(column='BLDG_VALUE', cmap="Reds", edgecolor='black', linewidth=0.05, legend=True)

    LU_new

    df_parcel['LU_new'] = LU_new

    df_parcel[['LU_new', 'BLDG_VALUE']].groupby('LU_new').apply(np.mean)

    df_parcel[['LU_new', 'BLDG_VALUE']].groupby('LU_new').apply(lambda x: sum(x.BLDG_VALUE==0))

    df_parcel_raw[df_parcel_raw.BLDG_VALUE==0]


    df_parcel_raw[df_parcel_raw.LU_DESC == 'CONDO MAIN'].BLDG_VALUE





# Backward compatibility: call into original function if present
def extract_features(**kwargs):
    """
    Public entry point for this step. Accepts keyword args to customize behavior.
    """
    logger = setup_logging()
    logger.info("Starting extract_features()")
    if "features" in globals():
        return features(**kwargs) if callable(globals().get("features")) else None
    # Fallback: nothing to run
    logger.warning("Original function 'features()' not found; nothing executed.")
    return None

def _cli():
    parser = argparse.ArgumentParser(description="extract_features step")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to data dir")
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR), help="Path to figures dir")
    args = parser.parse_args()
    ensure_dir(Path(args.figures_dir))
    return extract_features()

if __name__ == "__main__":
    _cli()
