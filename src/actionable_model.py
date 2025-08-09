# Packages & Global Variables

# General
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# Geospatial
import geopandas as gpd
from shapely import wkt

# Plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Preprocessing
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler 

# Modeling & Evaluation
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import shap

from get_data_db import get_data_db


# Data path
DATA_PATH = r"..\data"

# Coordinate reference systems
CRS_2249 = "EPSG:2249"
CRS_4326 = "EPSG:4326"  # lat-lon
CRS_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

from datetime import datetime
# Save
date_string = datetime.now().strftime('%Y%m%d')

# Helper functions to preprocess data

def add_season_year(df_in, date_col='date'):
    '''
    Add season and year column to given dataframe, given date column
    '''
    df = df_in.copy()
    
    # Mapping from month to season
    mapping = {1: 'winter',
                2: 'winter',
                3: 'spring',
                4: 'spring',
                5: 'spring',
                6: 'summer',
                7: 'summer',
                8: 'summer',
                9: 'fall',
                10: 'fall',
                11: 'fall', 
                12: 'winter'}

    # Get season and year
    df['season'] = df[date_col].dt.month.replace(mapping)
    df['year'] = df[date_col].dt.year
    
    return df


def add_gids(df_in):
    '''
    Add block group and tract GIDs to each point in given dataframe
    '''
    df = df_in.copy()
    df = gpd.sjoin(df, df_block_groups[['geometry', 'gid']]).drop('index_right', axis=1)
    df = gpd.sjoin(df, df_tracts[['geometry', 'tract_gid']]).drop('index_right', axis=1)
    return df


def get_season_cutoffs(dates):
    '''
    Create a list of dates that separates season from season (dec/jan/feb, mar/apr/may, jun/jul/aug, sep/oct/nov) throughout the data
    e.g. If data spans 2023, then return 03-01-2023, 06-01-2023, 09-01-2023, 12-01-2023
    '''
    # Get max and min dates
    date_max = dates.max()
    date_min = dates.min()

    # Get season boundaries
    season_cutoffs = [pd.to_datetime(f"{yr}-{m}-01") for yr in range(date_min.year-1, date_max.year+1) for m in ['3', '6', '9', '12']]
    season_cutoffs.reverse()
    
    return season_cutoffs


def add_season_groups(df_in, season_cutoffs):
    '''
    Assign season groups to rows based on season cutoffs and row's date
    '''
    # Dataframe to keep track of dates and corresponding season group
    df = df_in.copy()
    df['season_group'] = 0
    
    # Update season group if past season cutoff, for each season cutoff
    for i, cutoff in enumerate(season_cutoffs):
        df.season_group = df.apply(lambda row: i if row.date<cutoff else row.season_group, axis=1)
    
    # Return original df with new season group column
    return df


def subset_date(df, date_cutoff):
    '''
    Drop rows from df that are before the date cutoff or in the future
    '''
    return df[(df.date >= date_cutoff) & (df.date <= pd.Timestamp.today())].reset_index(drop=True)

# Data

def read_data():
    '''
    Read all data, including
    - df_tracts = census tract geometries
    - df_block_groups = census block group geometries
    - df_neighborhoods = neighborhood geometries
    - df_features = features by census block group
    - df_reporting_metrics = reporting bias metrics by census tract
    - df_violations = Code Enforcement violations
    - df_survey123 = proactive inspections
        - df_sampling = sampled points from Summer 2024
        - df_survey123_unsuccessful = proactive inspections without evidence of rodent activity
    - df_complaints = 311 rodent complaints
    '''
    
    # Initialize data as global variables
    global df_tracts
    global df_block_groups
    global df_neighborhoods
    global df_features
    global df_reporting_metrics
    global df_violations
    global df_survey123
    global df_survey123_unsuccessful
    global df_sampling
    global df_complaints
    
    ########################################################################################################################################################################################
    # Census tracts (geometries from Civis)
    df_tracts = get_data_db(
        query='census_tract_geometry.sql',
        col_names=['gid','objectid','statefp20','countyfp20','tractce20','geoid20','name20','namelsad20','mtfcc20','funcstat20','aland20','awater20','intptlat20','intptlon20','shape_star','shape_stle','geom','st_astext'],
        save_file=False
    )
    df_tracts.insert(0, "geometry", df_tracts['st_astext'].apply(wkt.loads))
    df_tracts = gpd.GeoDataFrame(df_tracts, geometry='geometry', crs=CRS_2249)
    df_tracts.rename(columns={'gid': 'tract_gid'}, inplace=True)  # Rename gid to tract_gid to avoid confusion with block gid
    df_tracts.to_crs(CRS_4326, inplace=True)
    df_tracts_proj = df_tracts.to_crs(CRS_32619)
    
    ########################################################################################################################################################################################
    # Census block groups (geometries from Civis)
    df_block_groups = get_data_db(
        query='census_block_groups.sql',
        col_names = ['gid','objectid','statefp20','countyfp20','tractce20','blkgrpce20','geoid20','namelsad20','mtfcc20','funcstat20','aland20','awater20','intptlat20','intptlon20','shape_star','shape_stle','geom','st_astext'],
        save_file=False
    )
    df_block_groups.insert(0, "geometry", df_block_groups['st_astext'].apply(wkt.loads))
    df_block_groups = gpd.GeoDataFrame(df_block_groups, geometry='geometry', crs=CRS_2249)
    df_block_groups = df_block_groups[['geometry', 'gid','geoid20']]
    df_block_groups.to_crs(CRS_4326, inplace=True)

    ########################################################################################################################################################################################
    # Neighborhoods (geometries from Civis)
    # df_neighborhoods = pd.read_csv(DATA_PATH + '/neighborhood_data.csv')
    df_neighborhoods = get_data_db(
        query='neighborhood_data.sql',
        col_names = ['OBJECTID','name','acres','neighborhood_id','sqmiles','geom_multipolygon_2249','geom_multipolygon_4326','_ingest_datetime','st_astext'],
        save_file=False
    )
    df_neighborhoods.insert(0, "geometry", df_neighborhoods['st_astext'].apply(wkt.loads))
    df_neighborhoods = gpd.GeoDataFrame(df_neighborhoods, geometry='geometry', crs=CRS_2249)
    df_neighborhoods.to_crs(CRS_4326, inplace=True)
    # Fix geometries
    df_neighborhoods['geometry'] = df_neighborhoods['geometry'].buffer(0)

    ########################################################################################################################################################################################
    # Features (from feature engineering)
    df_features = pd.read_csv(f'../data/output/features_{date_string}.csv')
    df_features['geometry'] = df_features['geometry'].apply(wkt.loads)
    df_features = gpd.GeoDataFrame(df_features, geometry='geometry', crs=CRS_4326)
    df_features.loc[pd.isna(df_features.avg_n_trash_days), ['avg_n_trash_days', 'n_unique_trash_days']] = 0  # Replace missing trash days with 0

    ########################################################################################################################################################################################
    # Reporting bias (from debiasing analysis)
    df_reporting_metrics = pd.read_csv("../data/reporting_metrics.csv")
    df_reporting_metrics['geometry'] = df_reporting_metrics['geometry'].apply(wkt.loads)
    df_reporting_metrics = gpd.GeoDataFrame(df_reporting_metrics, geometry='geometry', crs=CRS_4326)
    df_reporting_metrics.rename(columns={'gid': 'tract_gid'}, inplace=True)  # Rename gid to tract_gid to avoid confusion with block gid

    ########################################################################################################################################################################################
    # Violations (from BostonMaps https://boston.maps.arcgis.com/home/item.html?id=33c6e5e6fe1145c0bb2ccb35dc53f7f5)
    df_violations = gpd.read_file(DATA_PATH + r"\rodent_violations\rodent_violations.shp")
    df_violations.to_crs(CRS_4326, inplace = True)

    ########################################################################################################################################################################################
    # Survey123 (from BostonMaps https://survey123.arcgis.com/surveys/4fbe77c098f7432dbbdab5ff048eff79/data?extent=-157.5000,-46.6796,157.5000,46.6796)
    df_survey123 = pd.read_csv(DATA_PATH + r"\survey_123_updated_20240920.csv")
    df_survey123 = gpd.GeoDataFrame(df_survey123, geometry=gpd.points_from_xy(df_survey123.x, df_survey123.y), crs= CRS_4326)
    # Subset to inspections with valid geometries in Boston
    df_survey123 = df_survey123[df_survey123.intersects(df_block_groups.geometry.unary_union)]

    # Extract sampling inspections
    df_sampling = df_survey123[df_survey123['Complaint-Based, Proactive, Smoke Test, or BWSC Project?']=='Sampling_']
    df_survey123 = df_survey123[df_survey123['Complaint-Based, Proactive, Smoke Test, or BWSC Project?']!='Sampling_']

    # Extract successful vs unsuccessful inspections
    conditions = (df_survey123['Type of Baiting'].isna()) & (df_survey123['General Baiting'].isna()) & (df_survey123['Bait Added (number)'].isna()) & (df_survey123['Total Bait Left (number)'].isna())
    df_survey123_unsuccessful = df_survey123[conditions]
    df_survey123 = df_survey123[~conditions]

    ########################################################################################################################################################################################
    # Complaints (from Civis)
    # df_complaints = pd.read_csv(DATA_PATH + r"\311_rodent_complaints.csv")
    df_complaints = get_data_db(
        query='311_rodent_complaints.sql',
        col_names=['case_enquiry_id','case_reference','created_by_id','open_dt','closed_dt','sla_target_dt','sla_suspended_dt','priority','severity','case_status','closure_reason','closed_by_id','case_description','case_title','subject','reason','type','queue','case_x','case_y','location','objecttype','propid','parcel_num','stid','fire_district','pwd_district','city_council_district','police_district','neighborhood','neighborhood_services_district','ward','vip','precinct','land_usage','us_representative_district','state_senatorial_district','location_street_number','location_street_name','location_city','location_zipcode','location_x','location_y','channel_type','reporter_first_name','reporter_last_name','reporter_addressnum','reporter_addressline1','reporter_addressline2','reporter_city','reporter_state','reporter_zipcode','reporter_phone_number','reporter_phone_type','reporter_email_address','reporter_email_type','source','geom_4326','st_astext'],
        save_file=False
    )

    df_complaints.insert(0, "geometry", df_complaints['st_astext'].apply(wkt.loads))
    df_complaints = gpd.GeoDataFrame(df_complaints, geometry = 'geometry', crs = CRS_4326)
    df_complaints.drop(columns=['geom_4326', 'st_astext'], inplace = True)  # Drop old geom columns
    

def clean_and_engineer_data(date_cutoff = '2023-06-01'):
    '''
    Feature engineer spatial and temporal variables into data; 
    Remove data points from the future and before the date cutoff, and data points outside of Boston
    Create dataframe of counts for each block group for each season group
    '''
    
    global df_tracts
    global df_block_groups
    global df_boston
    global df_neighborhoods
    global df_features
    global df_reporting_metrics
    global df_violations
    global df_survey123
    global df_survey123_unsuccessful
    global df_sampling
    global df_complaints
    global df_counts
    
    ########################################################################################################################################################################################
    # df_block_groups: tract_gid
    # Assign blocks groups to tracts by finding largest intersection between block group and tract
    intersections = [df_tracts.intersection(row.geometry) for i, row in df_block_groups.iterrows()]   # For each census block, get intersection with each tract (EMPTY if no overlap)
    intersections = [i.to_crs(CRS_32619) for i in intersections]  # Change to projected CRS to get area of intersections
    df_block_groups['tract_gid'] = [df_tracts.tract_gid[np.argmax(i.area)] for i in intersections]  # Get index of tract with max area for this block
    
    ########################################################################################################################################################################################
    # df_boston: one dataframe with entire Boston geometry excluding harbor islands for plotting
    harbor_islands_idx = df_block_groups.geometry.representative_point().x.idxmax()
    df_boston = gpd.GeoDataFrame({'geometry': [df_block_groups.drop(harbor_islands_idx).unary_union]}, geometry='geometry', crs=CRS_4326)
    
    ########################################################################################################################################################################################
    # df_neighborhoods: coords
    # Get representative point for plotting purposes
    df_neighborhoods['coords'] = df_neighborhoods.geometry.representative_point().apply(lambda x: x.coords[0])
    # Drop Harbor Islands for plotting purposes
    df_neighborhoods = df_neighborhoods[df_neighborhoods.name!='Harbor Islands']
    
    ########################################################################################################################################################################################
    # df_features: tract_gid
    df_features = df_features.merge(df_block_groups[['gid', 'tract_gid']], on='gid')
    
    ########################################################################################################################################################################################
    # df_reporting_metrics: p2_overreporting
    # Make new probability distribution from m2 (should do in sidewalks.ipynb)
    df_reporting_metrics['p2_overreporting'] = df_reporting_metrics.m2.apply(lambda m: m / sum(df_reporting_metrics.m2))
    
    ########################################################################################################################################################################################
    # df_violations: date, season, year, block group gid
    # Rename date column
    df_violations.rename(columns={'casedttm':'date'}, inplace=True)
    df_violations['date'] = pd.to_datetime(df_violations['date'])
    df_violations = subset_date(df_violations, date_cutoff)
    # Add season, year, and gids
    df_violations = add_season_year(df_violations)
    df_violations = add_gids(df_violations)
    
    ########################################################################################################################################################################################
    # df_survey123: date, season, year, block group gid
    # Create date column
    df_survey123['date'] = pd.to_datetime(df_survey123['Current Date (MM/DD/YYYY)'], format='%m/%d/%Y %I:%M:%S %p').dt.strftime('%Y-%m-%d')
    df_survey123['date'] = pd.to_datetime(df_survey123['date'])
    df_survey123 = subset_date(df_survey123,date_cutoff)
    # Add season, year, and gids
    df_survey123 = add_season_year(df_survey123)
    df_survey123 = add_gids(df_survey123)
    
    # Create date column
    df_survey123_unsuccessful['date'] = pd.to_datetime(df_survey123_unsuccessful['Current Date (MM/DD/YYYY)'], format='%m/%d/%Y %I:%M:%S %p').dt.strftime('%Y-%m-%d')
    df_survey123_unsuccessful['date'] = pd.to_datetime(df_survey123_unsuccessful['date'])
    df_survey123_unsuccessful = subset_date(df_survey123_unsuccessful, date_cutoff)
    # Add season, year, and gids
    df_survey123_unsuccessful = add_season_year(df_survey123_unsuccessful)
    df_survey123_unsuccessful = add_gids(df_survey123_unsuccessful)
    
    ########################################################################################################################################################################################
    # df_sampling: date, season, year, block group gid, label
    # Create date column
    df_sampling['date'] = pd.to_datetime(df_sampling['Current Date (MM/DD/YYYY)'], format='%m/%d/%Y %I:%M:%S %p').dt.strftime('%Y-%m-%d')
    df_sampling['date'] = pd.to_datetime(df_sampling['date'])
    df_sampling = subset_date(df_sampling, date_cutoff)
    # Add season, year, and gids
    df_sampling = add_season_year(df_sampling)
    df_sampling = add_gids(df_sampling)
    # Get label
    df_sampling['label'] = df_sampling.Sampling.apply(lambda s: 0 if s=="No Rats" else 1)
    
    ########################################################################################################################################################################################
    # df_complaints: date, season, year, block group gid
    # Drop addresses outside Boston
    df_complaints = df_complaints[df_complaints.intersects(df_block_groups.geometry.unary_union)]
    # Create date column
    df_complaints['date'] = pd.to_datetime(df_complaints.open_dt.dt.date)
    df_complaints = subset_date(df_complaints, date_cutoff)
    # Add season, year, and gids
    df_complaints = add_season_year(df_complaints)
    df_complaints = add_gids(df_complaints)
    
    ########################################################################################################################################################################################
    # Season groups for violations, survey123, and complaints
    season_cutoffs = get_season_cutoffs(pd.concat([df_violations.date, df_survey123.date, df_complaints.date]))
    df_violations = add_season_groups(df_violations, season_cutoffs)
    df_survey123 = add_season_groups(df_survey123, season_cutoffs)
    df_survey123_unsuccessful = add_season_groups(df_survey123_unsuccessful, season_cutoffs)
    df_sampling = add_season_groups(df_sampling, season_cutoffs)
    df_complaints = add_season_groups(df_complaints, season_cutoffs)
    
    ########################################################################################################################################################################################
    # Counts for violations and complaints from past season
    # Types of complaints
    complaint_type_rodent = ['Rodent Activity', 'Pest Infestation - Residential', 'Rat Bite', 'Mice Infestation - Residential']
    complaint_type_trash = ['Overflowing or Un-kept Dumpster', 'Trash on Vacant Lot', 'Poor Conditions of Property', 'Pick up Dead Animal', 'Improper Storage of Trash (Barrels)', 'Unsanitary Conditions - Establishment', 'Unsanitary Conditions - Food']

    # Violations
    df_violations_counts = df_violations.groupby(['season_group', 'gid']).size().reset_index().rename(columns={0:'n_violations'})
    # Rodent complaints
    df_complaints_rodent_counts = df_complaints[df_complaints['type'].isin(complaint_type_rodent)].groupby(['season_group', 'gid']).size().reset_index().rename(columns={0:'n_complaints_rodent'})
    # Trash complaints
    df_complaints_trash_counts = df_complaints[df_complaints['type'].isin(complaint_type_trash)].groupby(['season_group', 'gid']).size().reset_index().rename(columns={0:'n_complaints_trash'})
    # Merge everything
    df_counts = df_violations_counts.merge(df_complaints_rodent_counts, on=['season_group', 'gid']).merge(df_complaints_trash_counts, on=['season_group', 'gid'])
    # Get next season (to apply last season's counts)
    df_counts.loc[:, 'next_season'] = df_counts.season_group.apply(lambda x: x-1)
    
    ########################################################################################################################################################################################
    # Sort and reset indices
    df_violations = df_violations.sort_values('date').reset_index(drop=True)
    df_survey123 = df_survey123.sort_values('date').reset_index(drop=True)
    df_survey123_unsuccessful = df_survey123_unsuccessful.sort_values('date').reset_index(drop=True)
    df_sampling = df_sampling.sort_values('date').reset_index(drop=True)
    df_complaints = df_complaints.sort_values('date').reset_index(drop=True)

def label_positives(df_in):
    '''
    Given dataframe of positive points, subset to relevant columns & add label 1
    '''
    df = df_in.copy()
    
    # Subset to relevant columns 
    df = df[['geometry', 'gid', 'tract_gid', 'season', 'year', 'season_group']]
    
    # Add label
    df['label'] = 1
    
    return df

# Train-test split data based on test season group
def train_test_split(df_in, test_season_group):
    df_test = df_in[df_in.season_group==test_season_group]
    df_train = df_in[df_in.season_group>test_season_group]
    
    return (df_train, df_test)

# Pseudosampling: Violations

def pseudosample_violations(df_in, ps_prop=0.8, seed=1):
    '''
    Given training-set dataframe of violations, pseudosample negatives by season.
    IN:
    - df_in: dataframe containing violations
    - ps_prop: float describing 'pseudosample proportion', i.e., how many negatives to sample as a proportion of positives
    - seed: integer for random seed
    OUT:
    - df: dataframe containing pseudosampled negatives
    '''
    # Create dataframe to hold negatives
    df = df_in.drop(df_in.index)
    
    # Set seed
    np.random.seed(seed)
    
    # Pseudosample for each season
    for season_group, group in df_violations_train.groupby('season_group'):
        # Get number of negatives to sample
        n_negatives = int(group.shape[0]*ps_prop)
        
        # Get tracts available for pseudosampling
        group_tracts = df_reporting_metrics[df_reporting_metrics.tract_gid.isin(group.tract_gid)]

        # Calculate pseudosampling probabilities according to m2
        group_tracts.loc[:, 'p2_overreporting'] = group_tracts.m2.apply(lambda m: m / sum(group_tracts.m2))
        
        # Pseudosample
        pseudo_tract_gids = list(np.random.choice(group_tracts.tract_gid, size=n_negatives, p=group_tracts.p2_overreporting))  # According to reporting bias metric
        # pseudo_tract_gids = list(np.random.choice(group_tracts.tract_gid, size=n_negatives))  # Randomly
        # For each tract, sample a random block group within the tract
        for tract_gid in pseudo_tract_gids:
            # Get random block in this tract
            pseudo_block_group_gid = np.random.choice(df_block_groups[df_block_groups.tract_gid==tract_gid].gid)  # Sample GID
            pseudo_block_group = df_block_groups[df_block_groups.gid==pseudo_block_group_gid][['geometry', 'gid', 'tract_gid']]  # Get block group with that GID
            # Add season, year, and label
            pseudo_block_group['season'] = group.season.values[0]
            pseudo_block_group['year'] = group.year.max()
            pseudo_block_group['season_group'] = season_group
            pseudo_block_group['label'] = 0
            # Save into dataframe
            df = pd.concat([df, pseudo_block_group],
                        ignore_index=True)
    
    return df

# Pseudosampling: Violations

def pseudosample_survey123(df_in, ps_prop=0.5, seed=2):
    '''
    Given training-set dataframe of proactive inspections, pseudosample negatives by season.
    IN:
    - df_in: dataframe containing proactive inspections
    - ps_prop: float describing 'pseudosample proportion', i.e., how many negatives to sample as a proportion of positives
    - seed: integer for random seed
    OUT:
    - df: dataframe containing pseudosampled negatives
    '''
    # Create dataframe to hold negatives
    df = df_in.drop(df_in.index)
    
    # Set seed
    np.random.seed(seed)

    # Pseudosample for each season
    for season_group, group in df_survey123_train.groupby('season_group'):
        # Get number of negatives to sample
        n_negatives = int(group.shape[0]*ps_prop)
        
        # Get Survey123 counts for each block group
        tmp = df_block_groups[['geometry', 'gid']]
        tmp.insert(0, 'n_survey123', [sum(row.geometry.intersects(group.geometry)) for i, row in df_block_groups.iterrows()])
        
        # Create probability distribution based on counts
        tmp.loc[:, 'p'] = tmp.n_survey123.apply(lambda n: n/sum(tmp.n_survey123))
        
        # Pseudosample block groups
        pseudo_block_group_gids = list(np.random.choice(tmp.gid, size=n_negatives, p=tmp.p))  # According to count of proactive inspections 
        # pseudo_block_group_gids = list(np.random.choice(tmp.gid, size=n_negatives))  # Randomly
        # Add each block group as entry to df_negatives
        for pseudo_block_group_gid in pseudo_block_group_gids:
            # Get block group
            pseudo_block_group = df_block_groups[df_block_groups.gid==pseudo_block_group_gid][['geometry', 'gid', 'tract_gid']]
            # Add season, year, and label
            pseudo_block_group.loc[:, 'season'] = group.season.values[0]
            pseudo_block_group.loc[:, 'year'] = group.year.max()
            pseudo_block_group.loc[:, 'season_group'] = season_group
            pseudo_block_group.loc[:, 'label'] = 0
            # Save
            df = pd.concat([df, pseudo_block_group],
                        ignore_index=True)
    
    return df

# Make train and test datasets

def apply_features(df):
    '''
    Make a final dataset from positive points, negative points, and features. Make categorical variables dummies.
    IN: 
    - df: dataframe of points, with columns geometry, gid, season, year, and label
    OUT:
    - df_train: dataframe of training points with all features as columns, joined on gid
    - df_test: dataframe of test points with all features as columns, joined on gid
    '''
    # Merge m on tract gid
    df = df.merge(df_reporting_metrics[['tract_gid', 'm2']], how='left', on='tract_gid')
    
    # Merge features on block group gid
    df = df.merge(df_features.drop('tract_gid', axis=1), on='gid', how='left')
    
    # Merge past season counts on season_group and gid
    df = df.merge(df_counts[['n_violations', 'n_complaints_rodent', 'n_complaints_trash', 'next_season', 'gid']], left_on=['season_group', 'gid'], right_on=['next_season', 'gid'], how='left')
    df['n_violations'] = df.n_violations.fillna(0)
    df['n_complaints_rodent'] = df.n_complaints_rodent.fillna(0)
    df['n_complaints_trash'] = df.n_complaints_trash.fillna(0)
    
    # Create dummy for seasons
    df_dummies = pd.get_dummies(df.season)
    df_dummies = df_dummies.reindex(columns=['spring', 'summer', 'fall', 'winter'], fill_value=0)
    df = pd.concat([df.drop('season', axis=1), df_dummies], axis=1)
    
    # Drop columns that aren't features or labels
    columns =  list(df_features.drop(['gid', 'tract_gid', 'geometry'], axis=1).columns) + ['m2', 'spring', 'summer', 'fall', 'winter', 'year', 'label', 'n_violations', 'n_complaints_rodent', 'n_complaints_trash', 'season_group']
    df = df[columns]
    
    return df

# Columns to drop due to collinearity / model performance

def drop_columns(df_in, drop_cols=['sewer_length', 'bldg_density', 'yrs_since_remodel_max', 'EXT_wood', 'EXT_asbestos', 'sewers_junction_count', 'sewers_access_points_count', 
                                        'sewer_length_per_m2', 'LU_CO', 'year', 'establishment_dist', 'n_establishments', 'n_violations', 'n_complaints_trash', 'n_complaints_rodent',
                                        'bldg_value_mean', 'bldg_value_min']):
    '''
    Drop given columns
    '''
    # Copy df
    df = df_in.copy()
    # Drop columns
    df.drop(drop_cols, axis=1, inplace=True)
    return df

def drop_nas(df_in):
    '''
    Drop rows that have any NAs
    '''
    df = df_in.copy()
    
    # Drop rows
    df = df[~pd.isna(df).any(axis=1)].reset_index(drop=True)
    
    return df

def split_x_y(df_in):
    '''
    Given dataframe, split into X (features) and y (label)
    '''
    X = df_in.drop('label', axis=1)
    y = df_in.label
    return (X, y)


def standardize(X_in, scaler):
    '''
    Standardize given features (X_in) according to scaler
    '''
    X = X_in.copy()
    cols = X.columns
    X = pd.DataFrame(scaler.transform(X), columns=cols)
    return X

# Define function for standardized evaluation

def evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=0.6, plots=True):
    '''
    Given model and threshold, evaluate model based on accuracy, precision, and AUC
    IN:
    - model: trained predictive model to evaluate
    - threshold: float describing prediction threshold
    - plots: boolean determining whether to show confusion matrix & ROC curve
    '''
    # Predict (binary)
    y_train_pred = np.array([1 if p>threshold else 0 for p in y_train_pred_proba])
    y_test_pred = np.array([1 if p>threshold else 0 for p in y_test_pred_proba])
    
    # Get metrics
    train_metrics = classification_report(y_train, y_train_pred, output_dict=True)
    test_metrics = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Print accuracy, precision, AUC
    print(f'''
\t\tTrain\tTest
Accuracy:\t{round(accuracy_score(y_train, y_train_pred), 2)}\t{round(accuracy_score(y_test, y_test_pred), 2)}
Precision:\t{round(train_metrics['1']['precision'], 2)}\t{round(test_metrics['1']['precision'], 2)}
AUC:\t\t{round(roc_auc_score(y_train, y_train_pred_proba), 2)}\t{round(roc_auc_score(y_test, y_test_pred_proba), 2)}''')
    
    # Plot confusion matrix and ROC
    if plots:
        # Confusion matrices
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        # Train
        conf_matrix_train = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(conf_matrix_train, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 14}, ax=axs[0])
        axs[0].set_xlabel('Predicted Labels', fontsize=14)
        axs[0].set_ylabel('True Labels', fontsize=14)
        axs[0].set_title('Confusion Matrix - Train', fontsize=16)
        axs[0].set_xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
        axs[0].set_yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
        # Test
        conf_matrix_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(conf_matrix_test, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 14}, ax=axs[1])
        axs[1].set_xlabel('Predicted Labels', fontsize=14)
        axs[1].set_ylabel('True Labels', fontsize=14)
        axs[1].set_title('Confusion Matrix - Test', fontsize=16)
        axs[1].set_xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
        axs[1].set_yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
        plt.show()
        
        # ROC
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        # Train
        auc_train = roc_auc_score(y_train, y_train_pred_proba)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
        axs[0].plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC Curve (AUC = {auc_train:.4f})')
        axs[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('ROC Curve - Training Set')
        axs[0].legend(loc='lower right')
        axs[0].grid(True, color='lightgray')
        # Test
        auc_test = roc_auc_score(y_test, y_test_pred_proba)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
        axs[1].plot(fpr_test, tpr_test, color='blue', lw=2, label=f'Test ROC Curve (AUC = {auc_test:.4f})')
        axs[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].set_title('ROC Curve - Test Set')
        axs[1].legend(loc='lower right')
        axs[1].grid(True)
        plt.show()
        
    return True

# Custom CV split
class CustomTrainTestSplit(BaseCrossValidator):
    def __init__(self, train_folds, test_folds):
        self.train_folds = train_folds
        self.test_folds = test_folds
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.train_folds)
    
    def split(self, X, y=None, groups=None):
        for train_indices, test_indices in zip(self.train_folds, self.test_folds):
            yield train_indices, test_indices

def create_custom_cv():
    # Create CV splits based on season group
    cv_splits = []
    train_season_groups = list(reversed(sorted(df_train.season_group.unique())))  # [5, 4, 3, 2]

    # For each validation period, get all periods before as training data
    for i in range(1, len(train_season_groups)):
        train_indices = pd.concat([df_train[df_train.season_group==p] for p in train_season_groups[:i]]).index
        test_indices = df_train[df_train.season_group==train_season_groups[i]].index
        cv_splits.append((train_indices, test_indices))

    # Get all train and test folds
    train_folds, test_folds = (list(zip(*cv_splits)))
    custom_cv = CustomTrainTestSplit(train_folds, test_folds)
    
    return custom_cv

# Define the model and parameters

def train_rf(custom_cv, seed=1):
    # Initialize model
    model_rf_cv = RandomForestClassifier(random_state=seed)

    # Grid search over parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    grid_search = GridSearchCV(estimator=model_rf_cv, param_grid=param_grid, scoring='accuracy', cv=custom_cv, verbose=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model from grid search
    model_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return model_rf, best_params

def feature_importance(model_rf, plots=True):
    # Get feature importances
    importances = model_rf.feature_importances_

    # Create a DataFrame for better visualization
    features = X_train.columns
    df_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_importance = df_importance.sort_values(by='Importance', ascending=False)

    if plots:
        # Plot the feature importances
        # print(importance_df)
        plt.figure(figsize=(9, 7))
        sns.barplot(x='Importance', y='Feature', data=df_importance)
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()
    
    return df_importance

def create_X_block_groups(drop_cols, scaler):
    '''
    Create dataset where each row is a census block group and each column is a feature. Used for projection of predictions onto block groups.
    '''
    # Get dataset for all block groups for current time period
    X_block_groups = df_block_groups[['geometry', 'gid', 'tract_gid']]
    X_block_groups.loc[:, 'season_group'] = 1
    X_block_groups.loc[:, 'season'] = 'summer'
    X_block_groups.loc[:, 'year'] = 2024
    X_block_groups.loc[:, 'label'] = None
    X_block_groups = apply_features(X_block_groups).drop(['label', 'season_group'], axis=1)
    # Prep like train/test data
    # Put GID back in so that we know which block groups we are predicting for after dropping NAs
    X_block_groups.loc[:, 'gid'] = df_block_groups.gid
    X_block_groups = drop_nas(drop_columns(X_block_groups, drop_cols))
    # Make note of block groups that we are predicting for
    X_block_groups_gids = X_block_groups.gid
    X_block_groups.drop('gid', axis=1, inplace=True)
    X_block_groups = standardize(X_block_groups, scaler)
    
    return X_block_groups, X_block_groups_gids

def predict_on_boston(model, threshold=0.6):
    # Predict on all block groups
    y_block_group_pred = model.predict_proba(X_block_groups)[:, 1]
    df_block_group_pred = pd.DataFrame({'gid': X_block_groups_gids, 'pred_proba': y_block_group_pred})
    df_block_group_pred = gpd.GeoDataFrame(df_block_group_pred.merge(df_block_groups[['gid', 'geometry']], on='gid'), geometry='geometry', crs=CRS_4326)
    df_block_group_pred.loc[:, 'pred'] = df_block_group_pred.pred_proba.apply(lambda p: 1 if p>threshold else 0)
    
    return df_block_group_pred

# Visualize for all of Boston
def project_to_boston(df_block_group_pred, plots=True):
    '''
    Use given model to predict onto all census block groups; visualize map if plots is True
    IN:
    - model: trained predictive model to map
    - threshold: float describing prediction threshold
    - plots: boolean determining whether to show confusion matrix & ROC curve
    OUT:
    - df_block_group_pred: dataframe of predictions on all census block groups
    '''

    # Visualize predictions
    if plots:
        fig, axs = plt.subplots(1, 2, figsize=(20,8))
        # Continuous
        df_boston.plot(ax=axs[0], edgecolor='black', color='gray')
        df_block_group_pred.plot(ax=axs[0], column='pred_proba', cmap='Reds', legend=True)
        axs[0].set_title(f"Actionable Predictions (Continuous)")
        axs[0].axis('off')
        # Binary
        df_boston.plot(ax=axs[1], edgecolor='black', color='gray')
        df_block_group_pred.plot(ax=axs[1], column='pred', cmap='Reds', legend=True)
        axs[1].set_title(f"Actionable Predictions (Binary)")
        axs[1].axis('off')
        plt.show()
        

def get_uncovered_block_groups(df_block_groups_preds, plots=True):
    # Complaints in this period
    df_c = df_complaints[df_complaints.season_group==test_season_group]
    # Survey123 in this period
    df_s123 = df_survey123[df_survey123.season_group==test_season_group]
    # GIDS without complaint or s123
    gids_no_visits = df_block_groups[~df_block_groups.gid.isin(list(df_c.gid.values) + list(df_s123.gid.values))].gid

    # How many block groups?
    print(f"{df_block_groups_preds[(df_block_groups_preds.pred==1) & (df_block_groups_preds.gid.isin(gids_no_visits))].shape[0]} block groups without complaints were identified as having a high risk of rodent activity")
    
    if plots:
        # Plot
        fig, ax = plt.subplots(figsize=(8,8))
        # Boston
        df_boston.plot(ax=ax, edgecolor='black', color='gray')
        # Predictions
        df_block_groups_preds.plot(ax=ax, column='pred', cmap='Reds')
        df_block_groups_preds[df_block_groups_preds.pred==1].plot(ax=ax, color='peachpuff')
        # Predictions without visits
        df_block_groups_preds[(df_block_groups_preds.pred==1) & (df_block_groups_preds.gid.isin(gids_no_visits))].plot(ax=ax, color='#FB4D42')

        # Legend
        legend_darkred = Patch(color="#FB4D42", label="Predicted w/ NO Complaint")
        legend_red = Patch(color="peachpuff", label="Predicted w/ Complaint")

        ax.legend(handles=[legend_darkred, legend_red], loc="lower right", prop={'size': 12})
        ax.axis('off')

        plt.show()

pred_threshold = 0.6  # Threshold for binary prediction

#############################################################################################################################################################################
# 0) Read and prep data
read_data()
clean_and_engineer_data()

print("0) Finished reading and cleaning data")


#############################################################################################################################################################################
# 1) Get biased positives
df_violations = label_positives(df_violations)
df_survey123 = label_positives(df_survey123)

print("1) Finished biased positives")


#############################################################################################################################################################################
# 2) Train-test split
# Test season group = most recent season group (Summer 2024)
test_season_group = df_violations.season_group.min()
print("Season", df_violations[df_violations.season_group==test_season_group].season.unique())
print("Year", df_violations[df_violations.season_group==test_season_group].year.unique())

# Train-test split ll data
df_violations_train, df_violations_test = train_test_split(df_violations, test_season_group)
df_survey123_train, df_survey123_test = train_test_split(df_survey123, test_season_group)
df_complaints_train, df_complaints_test = train_test_split(df_complaints, test_season_group)
df_sampling_train, df_sampling_test = train_test_split(df_sampling, test_season_group)

print("2) Finished train-test split")


#############################################################################################################################################################################
# 3) Pseudosample
# Initialize negatives dataset with same columns as positive
df_negatives_train = df_violations_train.drop(df_violations_train.index)
# Fill with pseudosampled violations & proactive inspections
df_negatives_train = pd.concat([pseudosample_violations(df_negatives_train, ps_prop=0.8), pseudosample_survey123(df_negatives_train)])

print("3) Finished pseudosampling")


#############################################################################################################################################################################
# 4) Apply features & Finalize datasets
# Concatenate all points into training and test datasets; apply features
pre_feature_columns = ['gid', 'tract_gid', 'season', 'year', 'label', 'season_group']  # Columns necessary to apply features to observations
# Train: violations, survey123, sampling, pseudonegatives
df_train = pd.concat([df_violations_train[pre_feature_columns], df_survey123_train[pre_feature_columns], df_sampling_train[pre_feature_columns], df_negatives_train[pre_feature_columns]], ignore_index=True)
df_train = apply_features(df_train)
# Test: violations, survey123, sampling
df_test = pd.concat([df_violations_test[pre_feature_columns], df_survey123_test[pre_feature_columns], df_sampling_test[pre_feature_columns]], ignore_index=True)
df_test = apply_features(df_test)

# Drop NAs and collinear cols
drop_cols = ['sewer_length', 'bldg_density', 'yrs_since_remodel_max', 'EXT_wood', 'EXT_asbestos', 'sewers_junction_count', 'sewers_access_points_count', 
            'sewer_length_per_m2', 'LU_CO', 'year', 'establishment_dist', 'n_establishments', 'n_violations', 'n_complaints_trash', 'n_complaints_rodent',
            'bldg_value_mean', 'bldg_value_min', 'n_waste_bins', 'waste_dist', 'sewers_condition_score']
df_train = drop_nas(drop_columns(df_train, drop_cols=drop_cols))
df_test = drop_nas(drop_columns(df_test, drop_cols=drop_cols))

# Split into X and y
X_train, y_train = split_x_y(df_train.drop('season_group', axis=1))
X_test, y_test = split_x_y(df_test.drop('season_group', axis=1))

# Standardize
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = standardize(X_train, scaler)
X_test = standardize(X_test, scaler)

print("4) Finished feature application")


#############################################################################################################################################################################
# 5) Train and evaluate model
custom_cv = create_custom_cv()  # Cross-validation by season
model_rf_cv, best_params = train_rf(custom_cv)  # Train model


y_train_pred_proba = model_rf_cv.predict_proba(X_train)[:, 1]
y_test_pred_proba = model_rf_cv.predict_proba(X_test)[:, 1]
evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=pred_threshold)
feature_importance(model_rf_cv)

print("5) Finished training model")


#############################################################################################################################################################################
# 6) Predict and project onto Boston

# Make features dataframe for blocks
X_block_groups, X_block_groups_gids = create_X_block_groups(drop_cols, scaler)

# Predict
df_block_groups_preds = predict_on_boston(model_rf_cv, threshold=pred_threshold)
print(df_block_groups_preds.pred.value_counts())

# Project onto Boston, compare with complaints
project_to_boston(df_block_groups_preds)
get_uncovered_block_groups(df_block_groups_preds)

print("6) Finished projecting to Boston")

from datetime import datetime
# Save
date_string = datetime.now().strftime('%Y%m%d')

df_block_groups_preds = df_block_groups_preds.merge(df_block_groups[['gid','geoid20']],how='left',on='gid')

df_block_groups_preds['geoid20'] = df_block_groups_preds.geoid20.astype(str)

# Save results
df_block_groups_preds.to_csv(f'../data/output/block_group_predictions_{date_string}.csv', index=False)

import os
if not os.path.exists(f'../data/block_group_predictions_{date_string}'):
    os.makedirs(f'../data/block_group_predictions_{date_string}')

df_block_groups_preds.to_file(f'../data/block_group_predictions_{date_string}/block_group_predictions_{date_string}.shp')

df_block_groups_preds.crs

# Extra visualizations
# Visualize block predictions with PANs

# Identify PANs
PANs = ['Downtown', 'Chinatown', 'North End', 'South End', 'Back Bay', 'Beacon Hill', 'Allston', 'Brighton', 'Dorchester', 'Roxbury']
df_neighborhoods['PAN'] = df_neighborhoods.name.apply(lambda n: n in PANs)
# Make dataframe with PANs as geometry
df_PANs = gpd.GeoDataFrame({'geometry': [df_neighborhoods[df_neighborhoods.PAN].unary_union]}, geometry='geometry', crs=CRS_4326)

# Visualize
fig, axs = plt.subplots(1, 3, figsize=(14, 5))

# Define order
ax_bin = axs[2]
ax_cont = axs[0]
ax_cont_pan = axs[1]

# Binary
ax_bin.set_title('Binary Predictions by Block Group')
df_boston.plot(ax=ax_bin, edgecolor='black', color='gray')
df_block_groups_preds.plot(ax=ax_bin, column='pred', cmap='Reds')

# Continuous
ax_cont.set_title('Continuous Predictions by Block Group')
df_boston.plot(ax=ax_cont, edgecolor='black', color='gray')
df_block_groups_preds.plot(ax=ax_cont, column='pred_proba', cmap='Reds')

# Continuous w/ PANs
ax_cont_pan.set_title('Continuous Predictions by Block Group, with PANs')
df_boston.plot(ax=ax_cont_pan, edgecolor='black', color='gray')
df_block_groups_preds.plot(ax=ax_cont_pan, column='pred_proba', cmap='Reds')
df_PANs.plot(ax=ax_cont_pan, facecolor='none', edgecolor='blue', linewidth=1.5)


ax_bin.axis('off')
ax_cont.axis('off')
ax_cont_pan.axis('off')

plt.subplots_adjust(wspace=0)

plt.show()

# # Plotting positives
fig, ax = plt.subplots()
df_block_groups.plot(ax=ax)
df_survey123.plot(ax=ax, color='red', markersize=2)
df_violations.plot(ax=ax, color="orange", markersize=2)
plt.title("Positive Points (Violations & Survey123)")
plt.show()

# # Visualize m with pseudosampled points

# Get random points in block for each observation
df_negatives_points = pseudosample_violations(df_negatives_train).geometry.sample_points(1)

# Plot
fig, ax = plt.subplots(figsize=(10,8))
# df_boston.plot(ax=ax, facecolor='none', edgecolor='black')
df_reporting_metrics.plot(ax=ax, column='m1', cmap='Reds', edgecolor='black', linewidth=0.1)
df_negatives_points.plot(ax=ax, color='blue', markersize=2)
# df_neighborhoods[df_neighborhoods.name=="Mattapan"].plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)  # Highlight Mattapan for presentation
plt.axis('off')
plt.show()




# Visualize pseudosampled points
fig, axs = plt.subplots(1, 3, figsize=(20,5))
gpd.GeoDataFrame(df_negatives_train.groupby('geometry').size().reset_index()).plot(ax=axs[0], column=0, cmap='Reds', legend=True)
df_reporting_metrics.plot(ax=axs[1], column='p2_overreporting', cmap='Reds', legend=True)
tmp = df_block_groups[['geometry']]
tmp = tmp.drop(tmp.geometry.representative_point().x.idxmax())
tmp.insert(0, 'n_survey123', [sum(row.geometry.intersects(df_survey123_train.geometry)) for i, row in tmp.iterrows()])
tmp.plot(ax=axs[2], column='n_survey123', cmap='Reds', legend=True)
axs[0].set_title("Pseudonegatives")
axs[1].set_title("P")
axs[2].set_title("Survey123")
plt.show()

# Train
model_logr = linear_model.LogisticRegression(random_state=42)
_ = model_logr.fit(X_train, y_train)

# Evaluate
y_train_pred_proba = model_logr.predict_proba(X_train)[:, 1]
y_test_pred_proba = model_logr.predict_proba(X_test)[:, 1]
evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=pred_threshold)

# Interpret w/ Coefficients

# Get coefficients and feature names
feature_coefs = model_logr.coef_[0]
feature_names = X_train.columns
# Sort
sorted_indices = np.argsort(np.abs(feature_coefs))
feature_coefs = feature_coefs[sorted_indices]
feature_names = np.array(feature_names)[sorted_indices]

plt.figure(figsize=(5, 10))
plt.barh(feature_names, feature_coefs, color='skyblue')
plt.xlabel('Coefficient Magnitude')
plt.title('Feature Importance - Logistic Regression')
plt.grid(True)
plt.show()

# Train
model_xgb = xgb.XGBClassifier(random_state=42)
_ = model_xgb.fit(X_train, y_train)

# Evaluate
y_train_pred_proba = model_xgb.predict_proba(X_train)[:, 1]
y_test_pred_proba = model_xgb.predict_proba(X_test)[:, 1]
evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=pred_threshold)

# Train & Find Best Parameters
model_xgb_cv = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'n_estimators': [150, 200, 250],
    # 'learning_rate': [0.01, 0.1],
    'max_depth': [5, 7, 10],
    # 'subsample': [0.3, 0.6, 0.8],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
}

custom_cv = create_custom_cv()
grid_search = GridSearchCV(estimator=model_xgb_cv, param_grid=param_grid, scoring='accuracy', cv=custom_cv, verbose=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
model_xgb_cv = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate
y_train_pred_proba = model_xgb_cv.predict_proba(X_train)[:, 1]
y_test_pred_proba = model_xgb_cv.predict_proba(X_test)[:, 1]
evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=pred_threshold)

# Train
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate
y_train_pred_proba = model_rf.predict_proba(X_train)[:, 1]
y_test_pred_proba = model_rf.predict_proba(X_test)[:, 1]
evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=pred_threshold)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Fix y train to be suitable for neural networks
y_train_nn = to_categorical(y_train)

# Define NN structure
model_nn = Sequential()
model_nn.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(y_train_nn.shape[1], activation='softmax'))

# Train
tf.random.set_seed(42)
model_nn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model_nn.fit(X_train, y_train_nn, 
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=0)

# Evaluate
y_train_pred_proba = model_nn.predict(X_train)[:, 1]
y_test_pred_proba = model_nn.predict(X_test)[:, 1]
evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=pred_threshold)

# Evaluate model like a NN
y_test_nn = to_categorical(y_test)
loss, accuracy = model_nn.evaluate(X_test, y_test_nn, verbose=2)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Plot train/validation accuracy/loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='best')

# Plot loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='best')

plt.show()

# Predict using logistic regression, xgb, and random forest

# Average probabilities to get predicted probability
y_train_pred = pd.DataFrame({'logr': model_logr.predict_proba(X_train)[:, 1],
                                'xgb': model_xgb_cv.predict_proba(X_train)[:, 1],
                                'rf': model_rf_cv.predict_proba(X_train)[:, 1]})
y_train_pred.loc[:, 'ensemble_prob'] = y_train_pred.apply(np.mean, axis=1)
y_train_pred.loc[:, 'ensemble_bin'] = y_train_pred.ensemble_prob.apply(lambda p: 1 if p>0.5 else 0)

y_test_pred = pd.DataFrame({'logr': model_logr.predict_proba(X_test)[:, 1],
                                'xgb': model_xgb_cv.predict_proba(X_test)[:, 1],
                                'rf': model_rf_cv.predict_proba(X_test)[:, 1]})
y_test_pred.loc[:, 'ensemble_prob'] = y_test_pred.apply(np.mean, axis=1)
y_test_pred.loc[:, 'ensemble_bin'] = y_test_pred.ensemble_prob.apply(lambda p: 1 if p>0.5 else 0)


# Evaluate
evaluate(y_train, y_train_pred.ensemble_prob, y_test, y_test_pred.ensemble_prob, threshold=pred_threshold)

# Evaluate different thresholds given a model

this_model = model_rf_cv

for t in np.arange(0.5, 1, 0.1):
    print(t)
    y_train_pred_proba = this_model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = this_model.predict_proba(X_test)[:, 1]
    evaluate(y_train, y_train_pred_proba, y_test, y_test_pred_proba, threshold=t, plots=False)
    print('\n\n')