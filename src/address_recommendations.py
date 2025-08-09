# Packages & Global Variables
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import math

# Data path
DATA_PATH = r"..\data"

# Coordinate reference systems
CRS_2249 = "EPSG:2249"
CRS_4326 = "EPSG:4326"  # lat-lon
CRS_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

from datetime import datetime
# Save
date_string = datetime.now().strftime('%Y%m%d')

# Data

# Predictions
df_preds = pd.read_csv(f'../data/output/block_group_predictions_{date_string}.csv')
df_preds['geometry'] = df_preds['geometry'].apply(wkt.loads)
df_preds = gpd.GeoDataFrame(df_preds, geometry='geometry', crs=CRS_4326)

# Boston
df_boston = gpd.GeoDataFrame({'geometry': [df_preds.unary_union]}, geometry='geometry', crs=CRS_4326)

# Census blocks
df_blocks_raw = gpd.read_file(DATA_PATH + r"\census_blocks\census_blocks.shp")

# df_blocks_raw = gpd.read_file("../data/census_blocks.csv")
# df_blocks_raw['geometry'] = df_blocks_raw['st_astext'].apply(wkt.loads)
# df_blocks_raw = gpd.GeoDataFrame(df_blocks_raw, geometry='geometry',crs=CRS_4326)

df_blocks = df_blocks_raw
df_blocks.to_crs(CRS_4326, inplace=True)

# Number of block groups per threshold

fig, axs = plt.subplots(1, 4, figsize=(25,7))

# Plot predictions per threshold
for i, t in enumerate([0.9, 0.8, 0.7, 0.6]):
    df_boston.plot(ax=axs[i], color='gray', edgecolor='black')
    df_preds.plot(ax=axs[i], color='seashell')
    df_preds[df_preds.pred_proba>=t].plot(ax=axs[i], color='red')
    axs[i].axis('off')
    axs[i].set_title(f"Threshold {t}: {df_preds[df_preds.pred_proba>=t].shape[0]} Blocks")

plt.show()

# Address per block
# Number of addresses to sample per block group = 0.5*n_blocks

df_blocks = df_blocks_raw
df_blocks.to_crs(CRS_4326, inplace=True)

# Get random address per block

# Get all addresses
df_addresses = pd.read_csv(DATA_PATH + r"/addresses.csv")
df_addresses['geometry_address'] = df_addresses['st_astext'].apply(wkt.loads)
df_addresses = gpd.GeoDataFrame(df_addresses, geometry='geometry_address', crs=CRS_4326)

# Clean zip code and full address
df_addresses.loc[:, 'zip_code'] = df_addresses.zip_code.astype(str).str.replace(".", "", regex=False)
df_addresses['address'] = df_addresses.apply(lambda row: f"{row.full_address}, Boston, MA {row.zip_code}", axis=1)
df_addresses = df_addresses[['geometry_address', 'sam_address_id', 'address']]

# Sample one random intersecting address per block
df_block_addresses = gpd.sjoin(df_blocks[['GEOID', 'geometry']], df_addresses[['sam_address_id', 'geometry_address']]).groupby('GEOID').apply(lambda group: group.sam_address_id.sample().values[0], include_groups=False).reset_index(name='sam_address_id')
gpd.sjoin(df_blocks[['GEOID', 'geometry']], df_addresses[['sam_address_id', 'geometry_address']])

# Assign address to block
df_blocks = df_blocks.merge(df_block_addresses, on='GEOID').merge(df_addresses, on='sam_address_id')

output20241017 = df_blocks.drop(columns='geometry')
output20241017.to_csv(DATA_PATH + r"/output/random_address_census_blocks_20241017.csv",index=False)
output20241017

# Assign block group gid to block based on intersection w/ representative point

# Get representative point per block
df_blocks_rep_point = df_blocks[['GEOID', 'geometry']]
df_blocks_rep_point.loc[:, 'geometry'] = df_blocks_rep_point.geometry.representative_point()

# Assign block group gid based on intersection
block_gids = gpd.sjoin(df_blocks_rep_point, df_preds[['geometry', 'gid']])[['GEOID', 'gid']]
df_blocks = df_blocks.merge(block_gids, on='GEOID')

# Subset rep points to blocks that have intersection with df_preds
df_blocks_rep_point = df_blocks_rep_point[df_blocks_rep_point.GEOID.isin(df_blocks['GEOID'])]

# Plot results

fig, axs = plt.subplots(1, 2, figsize=(15,10))

# True GIDs
df_preds.plot(ax=axs[0], column='gid')
axs[0].set_title("Block Groups: GIDs")

# Assigned GIDs
df_blocks.plot(ax=axs[1], column='gid')
axs[1].set_title("Blocks: Assigned GIDs")

for i in range(len(axs)):
    axs[i].axis('off')

plt.show()

# Block rep.point vs random address

fig, axs = plt.subplots(1, 2, figsize=(20,10))

# Block representative point
df_blocks.plot(ax=axs[0], color='lightgray')
df_blocks_rep_point.plot(ax=axs[0], color='red', markersize=2)
axs[0].set_title("Representative Point")

# Random address
df_blocks.plot(ax=axs[1], color='lightgray')
df_blocks.geometry_address.plot(ax=axs[1], color='red', markersize=2)
axs[1].set_title("Random Address")

for i in range(len(axs)):
    axs[i].axis('off')

plt.show()

# Number of Blocks per Block Group
n_blocks = block_gids.gid.value_counts().reset_index(name='n_blocks')
df_preds = df_preds.merge(n_blocks, on='gid', how='left')

# Number of Blocks per Block Group: Plot

fig, ax = plt.subplots(figsize=(10,10))

df_preds.plot(ax=ax, column='n_blocks', legend=True, figsize=(10,10))
ax.set_title("Number of Blocks per Block Group")
ax.axis('off')
plt.show()

# Zoom in on block group with most blocks
fig, ax = plt.subplots(figsize=(10,10))

df_blocks.plot(ax=ax)
df_blocks[df_blocks.gid == df_preds.loc[df_preds.n_blocks.idxmax(), 'gid']].plot(ax=ax, color='red', edgecolor='black')

# Get the extent of cluster points for setting the plot limits
geometry = df_preds.loc[df_preds.n_blocks.idxmax(), 'geometry']
geo_series = gpd.GeoSeries([geometry])
x_min, y_min, x_max, y_max = geo_series.total_bounds
buffer = 0.005  # Adjust the buffer for zooming in
# Set plot limits to zoom in around the cluster points
ax.set_xlim(x_min - buffer, x_max + buffer)
ax.set_ylim(y_min - buffer, y_max + buffer)
plt.show()

# Statistical description of number of blocks
df_preds.n_blocks.describe()

def get_address_recommendations(t, coverage, plots=True):
    np.random.seed(42)

    # Get predicted block groups for this threshold
    df_preds_t = df_preds[df_preds.pred_proba>=t]

    df_preds_addresses_t = df_blocks[['sam_address_id', 'geometry_address', 'address', 'gid']].drop(df_blocks.index)
    # Get addresses in each block group to visit
    for i, row in df_preds_t.iterrows():
        # Get number of addresses to select
        n_addresses = math.ceil(row.n_blocks*coverage)
        
        # Get pool of addresses to select from
        addresses_candidates = df_blocks[df_blocks.gid==row.gid]
        
        # Select n_addresses
        addresses_selected = addresses_candidates.sample(n_addresses)[['sam_address_id', 'geometry_address', 'address', 'gid']]
        
        # Concat to df_preds_addresses_t
        df_preds_addresses_t = pd.concat([
            df_preds_addresses_t,
            addresses_selected
        ])
        
    df_preds_addresses_t = gpd.GeoDataFrame(df_preds_addresses_t, geometry='geometry_address')
    print(f"With a threshold of {t} and with {coverage} coverage, we get {df_preds_t.shape[0]} block groups, totalling {df_preds_addresses_t.shape[0]} addresses to visit.")
    
    if plots:
        fig, ax = plt.subplots(figsize=(15,15))
        df_preds.plot(ax=ax, color='lightgray')
        df_preds_t.plot(ax=ax, color='red', edgecolor='black')
        df_preds_addresses_t.plot(ax=ax, color='black', markersize=3)
        ax.axis('off')
        plt.show()
    
    return df_preds_addresses_t

get_address_recommendations(0.6, 0.25)
1*14*60

# Give them anchor point: if you do X extra inspections, you can do Y
# What does a successful allocation of inspectors look like?
#   Treating long rodents/known
#   Fuller/equitable coverage of cities
# Validation:
#   Jump into 311 pipeline or
#   Keep as a separate validation period like sampling was?
# Coverage parameter? Enough visits to catch a rodent problem / Enough visits to confidently say that there is no rodent activity
# Methodology behind on-paper block surveys?

import datetime

epoch = datetime.datetime.now().strftime("%Y%m%d")
print('Current time: ',epoch)
df_address_recommendations = get_address_recommendations(0.6, 0.25, plots=False)
df_address_recs_with_pred = df_address_recommendations.merge(df_preds[['gid','pred_proba','n_blocks']], left_on='gid',right_on='gid')

df_blocks = pd.read_csv(DATA_PATH + r"\census_blocks.csv")
df_blocks.insert(0, "geometry", df_blocks['st_astext'].apply(wkt.loads))
df_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry', crs=CRS_2249)
df_blocks.to_crs(CRS_4326, inplace=True)

df_address_recs_with_pred = df_address_recs_with_pred.merge(df_blocks[['gid','geoid20']],left_on='gid',right_on='gid',how='left')

df_address_recs_with_pred['geoid20'] = df_address_recs_with_pred.geoid20.astype(str)

df_address_recs_with_pred.to_csv(DATA_PATH+rf'/output/address_recommendations_{date_string}.csv',index=False)
df_address_recs_with_pred

import os
if not os.path.exists(f'../data/address_recommendations_{date_string}'):
    os.makedirs(f'../data/address_recommendations_{date_string}')

df_address_recs_with_pred.to_file(f'../data/address_recommendations_{date_string}/address_recommendations_{date_string}.shp')

df_address_recs_with_pred.dtypes