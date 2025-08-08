from pathlib import Path
from .config import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED, CRS_WGS84
from .utils import setup_logging, ensure_dir
import argparse

def sidewalks():
    # Packages
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import re
    import pandas as pd
    import numpy as np
    from collections import Counter
    from shapely import wkt
    import math
    from tqdm import tqdm

    # Data path
    DATA_PATH = r"C:\Users\40009389\Documents\Rodents\data"

    # Coordinate reference systems
    CRS_2249 = "EPSG:2249"
    CRS_4326 = "EPSG:4326"  # lat-lon
    CRS_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19
    CRS_3857 = "EPSG:3857"  # projected, df_raw default

    # Read data

    # Sidewalks & complaints
    df_sidewalk_raw = gpd.read_file(DATA_PATH + r"\sidewalk_park_waste\BostonSidewalk.shp")
    df_complaint_raw = gpd.read_file(DATA_PATH + r"\sidewalk_park_waste\ONS_311_Sidewalks_11_20_23.shp")
    df_raw = gpd.read_file(DATA_PATH + r"\sidewalk_merged.gdb")  # Credit to Jeff Kaplan for merging these two datasets
    # Lat/Lon
    df_sidewalk_raw.to_crs(CRS_4326, inplace=True)
    df_complaint_raw.to_crs(CRS_4326, inplace=True)
    df_raw.to_crs(CRS_4326, inplace=True)


    # Blocks & tracts
    df_blocks = pd.read_csv(DATA_PATH + "\census_blocks.csv")
    df_blocks.insert(0, "geometry", df_blocks['st_astext'].apply(wkt.loads))
    df_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry', crs=CRS_2249)
    df_blocks.to_crs(CRS_4326, inplace=True)

    df_tracts = pd.read_csv(DATA_PATH + "\census_tract_geometry.csv")
    df_tracts.insert(0, "geometry", df_tracts['st_astext'].apply(wkt.loads))
    df_tracts = gpd.GeoDataFrame(df_tracts, geometry='geometry', crs=CRS_2249)
    df_tracts.to_crs(CRS_4326, inplace=True)


    # Copy/subset data

    cols = [
        # SCIs
        'SCI_2023',
        'New_SCI',
        'SCI_2023_2024',
        'MeasureSCI',
    
        # Sidewalk Information
        'Join_Count',
        'MATERIAL',
        'SWK_WIDTH',
        'PARENT',
        'SIDE',
        'Neighborhood',
        'DISTRICT',
        'Notes',
        'Notes_1',
        'InspectDate',
        'Recon_Date',
        'Rpr_Yr',
        'HIGHWAY',
    
        # Complaint Information
        'Priority',
    
        # IDs
        'GlobalID',
        'OBJECTID',
        'OBJECTID_1',
        'TARGET_FID',
        'SWK_ID',
        'SEG_ID',
        'CG_ID',
    
        # Geometry
        'Shape_Length',
        'Shape_Area',
        'geometry'
    ]

    # cols = df_raw.columns

    # Subset to relevant columns; make new ID
    df = df_raw[cols]
    df.reset_index(names='SID', inplace=True)

    # Make copy of raw df (to avoid reading in every time)
    df_sidewalk = df_sidewalk_raw.copy()
    df_complaint = df_complaint_raw.copy()

    # What column can we use to join df_complaint with df?

    for col in df_complaint.columns:
        print(f"{col}: \t{len(df_complaint[col].unique())} / {df_complaint.shape[0]}")

    # Let's try OBJECTID
    Counter(df_complaint.OBJECTID).most_common()

    # The OBJECTIDs that are repeated are
    # 16648: Both in Roxbury with SWK_ID=17862 
    # DEVON1_0
    # WARRE8_1912

    # 19536: Both in Jamaica Plain with SWK_ID=20704
    # SPALD1
    # SOUTH5

    # 0: Both in East Boston
    # Proposed new sidewalk, let's remove these since there will be no corresponding sidewalk

    df_complaint = df_complaint[df_complaint.OBJECTID!=0]

    # Get complaints that have no matching sidewalk
    df_complaint_missing = df_complaint[~df_complaint.OBJECTID.isin(df.OBJECTID_1)]
    OBJECTID_missing = list(df_complaint_missing.OBJECTID)

    # Find nearest sidewalk per missing complaint
    idx1 = []
    idx2 = []
    for i, row in tqdm(df_complaint_missing.to_crs(CRS_3857).iterrows()):
        # idx.append([row.geometry.distance(df.to_crs(CRS_3857).geometry).idxmin()])
        this_idx1 = list(row.geometry.distance(df.to_crs(CRS_3857).geometry).sort_values()[:3].index)
        idx1.append(this_idx1)
        this_idx2 = (abs(df.iloc[this_idx1].to_crs(CRS_3857).geometry.length - row.geometry.length)).idxmin()
        idx2.append(this_idx2)

    idx2

    # Visualize (~4 minutes)
    fig, axs = plt.subplots(8, 3, figsize=(20,60))

    for i in tqdm(range(len(OBJECTID_missing))):
        row = df_complaint_missing.iloc[[i]]
    
        # Determine which grid to put plot in
        ax = axs[math.floor(i/3), i%3]
    
        # Get lon and lat to zoom in on sidewalk
        lon = row.geometry.representative_point().x.values[0]
        lat = row.geometry.representative_point().y.values[0]
    
        # Plot
        # ax.scatter([1,2,3], [1,2,3])
        df.plot(ax=ax)
        row.plot(ax=ax, edgecolor="red", facecolor="red")
        df.iloc[[idx2[i]]].plot(ax=ax, edgecolor="lime", facecolor="lime")
        ax.set_xlim(lon-0.0025, lon+0.0025)
        ax.set_ylim(lat-0.0025, lat+0.0025)
        ax.set_title(f"{i}: OBJECTID {OBJECTID_missing[i]}")
    plt.show()

    # Sanity check: # unique complaint geometries
    print(f"number of complaint rows: {df_complaint.shape[0]}")
    print(f"number of unique complaint geometries: {len(df_complaint.geometry.unique())}")
    print(f"number of complaints in merged dataset: {sum(~pd.isna(df.OBJECTID_1))}")

    # Clean/engineer data

    # SCI_2023
    # Drop row where SCI is out of bounds
    df = df.drop(df[df.SCI_2023 > 100].index)

    # Priority
    # Remove only ONS complaints
    df.loc[:, 'Priority'] = df.Priority.replace('ONS', None)

    # MATERIAL
    # Replace Cc with CC
    df.loc[:, 'MATERIAL'] = df.MATERIAL.str.replace('Cc', 'CC')

    # SWK_WIDTH
    # Fix non-digit characters, convert to numeric
    df.loc[:, 'SWK_WIDTH'] = pd.to_numeric(df.SWK_WIDTH.str.replace('O', '0').replace(r"[^\d.]", '', regex=True))

    # SIDE
    # Replace non-standard formats with None
    df.loc[:, 'SIDE'] = df.SIDE.replace(r"( |BR)", None, regex=True)

    # DISTRICT
    # Replace 'THE NORTH END' with 'NORTH END'
    df.loc[:, 'DISTRICT'] = df.DISTRICT.str.replace('THE NORTH END', 'NORTH END')

    # SCI_bin
    # Sort sidewalks into severity bins (credit to Jeff Kaplan)
    # Score over 80: Do nothing
    # 50-80 Localized repair
    # Under 50: full reconstruction
    df.insert(0, "SCI_bin", ["Do Nothing" if s > 80 else "Full Reconstruction" if s < 50 else "Localized Repair" for s in df.SCI_2023])

    # Difference between SCI columns
    # 'SCI_2023', 'New_SCI', 'SCI_2023_2024', 'MeasureSCI'

    SCIs = ['SCI_2023', 'New_SCI', 'SCI_2023_2024', 'MeasureSCI']

    # Histograms
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for i, col in enumerate(SCIs):
        axs[i].hist(df[col])
        axs[i].set_title(col)
        axs[i].set_xlabel(col)
    axs[0].set_ylabel('Count')
    plt.show()


    # Scatterplots
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i, col_i in enumerate(SCIs):
        for j, col_j in enumerate(SCIs):
            axs[i,j].scatter(df[col_j], df[col_i], s=2)
            axs[i,0].set_ylabel(col_i)
            axs[0,j].set_title(col_j)
    plt.show()

    # Complaint sources: 311 vs ONS
    c = dict(Counter(df_raw[~pd.isna(df_raw.Priority)].Priority))
    plt.figure(figsize=(3,3))
    plt.bar(c.keys(), c.values())
    plt.title("Distribution of Complaint Source ('Priority')")
    plt.xlabel("Priority")
    plt.ylabel("Freq")
    plt.show()

    # SCI bin

    # Barplot
    # Get and reorder counts
    c = dict(Counter(df.SCI_bin))
    c_items = list(c.items())
    full_reconstruction = c_items.pop()
    c_items = [full_reconstruction] + c_items
    c = dict(c_items)

    # Plot
    plt.figure(figsize=(5,3))
    plt.bar(c.keys(), c.values(), width=0.6, color=["red", "yellow", "green"])
    plt.title("Sidewalk SCI Severity")
    plt.xlabel("SCI Score Severity")
    plt.ylabel("Freq")
    plt.show()


    # Maps
    # True sidewalk status
    colors = ["red" if s=="Full Reconstruction" else "yellow" if s=="Localized Repair" else "green" for s in df.SCI_bin]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    df.plot(ax=ax1, edgecolor=colors)
    ax1.set_title("Sidewalk Scores")

    # Complaints
    df.plot(ax=ax2)
    df_complaint.plot(ax=ax2, edgecolor="red")
    ax2.set_title("Complaints")

    plt.show()

    # Complaints

    fig, ax = plt.subplots(figsize=(10,10))

    # Get sidewalks needing repair
    df_bad_sidewalks = df[df.SCI_bin == "Full Reconstruction"]

    # Merge to tract
    df_sidewalk_tract = gpd.sjoin(df_bad_sidewalks, df_tracts[['gid', 'geometry']], how="left", predicate="intersects")

    # Reduce rows such that there is only one tract per sidewalk

    def get_biggest_gid(group):
        '''
        To be used as df.groupby('SID').apply(get_biggest_id)
        Given a sidewalk-tract dataframe with only one sidewalk ID, return the tract's gid with the largest intersection
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
    df_sidewalk_gids = df_sidewalk_tract.groupby('SID').progress_apply(get_biggest_gid, include_groups=False).reset_index()
    df_sidewalk_tract = df_sidewalk_tract.merge(df_sidewalk_gids, on=['SID', 'gid'], how='inner')

    # Get proportion of how many bad sidewalks are reported
    df_tract_metric1 = df_sidewalk_tract.groupby('gid')['Join_Count'].apply(lambda group: np.mean(group)).reset_index(name='prop_reported')

    # Trying 0+1/total+1 for tracts with 0 complaints
    def combine_functions(group):
        result = {}
    
        if sum(group.Join_Count)==0:
            result['prop_reported'] = 1/(group.shape[0]+1)
    
        else:
            result['prop_reported'] = np.mean(group.Join_Count)
    
        return pd.Series(result)

    df_tract_metric2 = df_sidewalk_tract.groupby('gid').apply(combine_functions, include_groups=False).reset_index()

    # Trying 0+1/total+1 for tracts with 0 complaints
    def combine_functions(group):
        result = {}
        result['prop_reported'] = (sum(group.Join_Count)+1)/(group.shape[0]+1)
        return pd.Series(result)

    df_tract_metric3 = df_sidewalk_tract.groupby('gid').apply(combine_functions, include_groups=False).reset_index()




    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))

    ax1.hist(df_tract_metric1.prop_reported, bins=50)
    ax1.set_title("Proportion of Bad Sidewalks Reported")

    ax2.hist(df_tract_metric2.prop_reported, bins=50)
    ax2.set_title("Add 1 to Tracts w/o Complaints")

    ax3.hist(df_tract_metric3.prop_reported, bins=50)
    ax3.set_title("Add 1 to All Tracts")

    plt.show()

    def combine_functions(group):
        result = {}
        result['complaints'] = np.sum(group.Join_Count)
        result['total'] = group.shape[0]
        return pd.Series(result)


    df_sidewalk_tract.groupby('gid').apply(combine_functions).reset_index()

    df_test = df_tracts[['geometry', 'gid']]
    df_test = pd.merge(df_test, df_sidewalk_tract.groupby('gid').apply(combine_functions, include_groups=False).reset_index(), on='gid')
    df_test = pd.merge(df_test, df_tract_metric2, on='gid')
    df_test = pd.merge(df_test, df_tract_metric3, on='gid')
    df_test.columns = ['geometry', 'gid', 'complaints', 'total', 'm2', 'm3']

    df_test['total2'] = df_test.complaints/df_test.m2
    df_test['total3'] = df_test.complaints/df_test.m3
    df_test

    plt.hist(df_test.total, alpha=0.6, label='total', bins=30)
    plt.hist(df_test.total2, alpha=0.6, label='total2', bins=30)
    plt.hist(df_test.total3, alpha=0.6, label='total3', bins=30)
    plt.legend()
    plt.show()

    df_test

    diffs = [abs(row.total2 - row.total3) for i, row in df_test.iterrows()]
    plt.hist(diffs)

    # What if we make a model that predicts #problems given tract and #complaints?
    # No, we need more data than just one row per tract

    #





# Backward compatibility: call into original function if present
def process_sidewalks(**kwargs):
    """
    Public entry point for this step. Accepts keyword args to customize behavior.
    """
    logger = setup_logging()
    logger.info("Starting process_sidewalks()")
    if "sidewalks" in globals():
        return sidewalks(**kwargs) if callable(globals().get("sidewalks")) else None
    # Fallback: nothing to run
    logger.warning("Original function 'sidewalks()' not found; nothing executed.")
    return None

def _cli():
    parser = argparse.ArgumentParser(description="process_sidewalks step")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to data dir")
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR), help="Path to figures dir")
    args = parser.parse_args()
    ensure_dir(Path(args.figures_dir))
    return process_sidewalks()

if __name__ == "__main__":
    _cli()
