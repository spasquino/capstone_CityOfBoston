# Packages

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Coordinate reference systems
crs_2249 = "EPSG:2249"
crs_4326 = "EPSG:4326"  # lat-lon
crs_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

# Data path
DATA_PATH = r"..\data"

##########################################################################################################################
# Normalized Block Features Data

df_sampling = pd.read_csv("../data/sampling.csv")
df_sampling.geometry = df_sampling.geometry.apply(wkt.loads)
df_sampling = gpd.GeoDataFrame(df_sampling, geometry='geometry', crs=crs_4326)
# df_sampling.drop(['bldg_value_min', 'LU_AH', 'EXT_other', 'EXT_vinyl'], axis=1, inplace=True)


##########################################################################################################################
# 311 Labelled Data

df_311 = pd.read_csv(DATA_PATH + r"\311_rodent_complaints.csv")
df_311.insert(0, "geometry", df_311['st_astext'].apply(wkt.loads))
df_311 = gpd.GeoDataFrame(df_311, geometry = 'geometry', crs = crs_4326)

# subset to recent complaints 
df_311 = df_311[df_311.open_dt > '2024-01-03'] # since March
df_311.reset_index(drop=True, inplace=True)

# add full address col
df_311['full_address'] = df_311['location_street_number'].astype(str) + ' ' + df_311['location_street_name'] + ', ' + df_311['neighborhood'] + ', MA, ' + df_311['location_zipcode'].astype(str) + ', USA'


##########################################################################################################################
# Survey123 Labelled Data

df_survey123 = pd.read_csv(DATA_PATH + r"\survey_123.csv")
df_survey123 = gpd.GeoDataFrame(df_survey123, geometry=gpd.points_from_xy(df_survey123.x, df_survey123.y), crs= crs_4326)

# subset to recent complaints 
df_survey123['Current Date (MM/DD/YYYY)']= pd.to_datetime(df_survey123['Current Date (MM/DD/YYYY)'], format='%m/%d/%Y %I:%M:%S %p')
df_survey123 = df_survey123[df_survey123['Current Date (MM/DD/YYYY)'] > '2024-03-01'] # since March


##########################################################################################################################
# Sam Addresses Data 

df_addresses = pd.read_csv(DATA_PATH + r"\addresses.csv")
df_addresses.insert(0, "geometry", df_addresses['st_astext'].apply(wkt.loads))
df_addresses = gpd.GeoDataFrame(df_addresses, geometry = 'geometry', crs = crs_4326)

# Format zip code
df_addresses.zip_code = df_addresses.zip_code.apply(lambda x: '0'+str(x)[:-2])
# Format full address
df_addresses['full_address'] = df_addresses.apply(lambda row: row.full_address + ", Boston, MA, " + row.zip_code, axis=1)


##########################################################################################################################
# Ward Data

df_wards = pd.read_csv(DATA_PATH + r"\wards.csv")
df_wards.insert(0, "geometry", df_wards['st_astext'].apply(wkt.loads))
df_wards = gpd.GeoDataFrame(df_wards, geometry='geometry', crs=crs_4326)
df_wards = df_wards[['geometry', 'ward']]

# Select addresses closest to centroid of each census block (1 per block) in census block gdf

df_addresses_proj = df_addresses.to_crs(crs_32619) #apply projection crs
df_sampling_proj = df_sampling.to_crs(crs_32619) #apply projection crs

block_centroids = df_sampling.geometry.centroid # get block centroid cohordinates
block_centroids_proj = df_sampling_proj.geometry.centroid # apply projection crs

closest_address_indices = [] # initialize list to store closest addresses to centroid

# iterate over each block centroid
for i, row in df_sampling_proj.iterrows():
    block_geometry = row.geometry
    block_centroid = block_centroids_proj[i]
    
    # extract valid addresses for the block
    valid_addresses = df_addresses_proj[df_addresses_proj.intersects(block_geometry)] # set of addresses in sam that fall within block i

    if valid_addresses.empty: 
        print('No valid addresses in block', i)
        continue

    distances = valid_addresses.geometry.apply(
        lambda g: block_centroids_proj[i].distance(g)
    ).astype(float)
    
    closest_idx = distances.idxmin()
    
    closest_address_indices.append(closest_idx)

df_sampling['closest_address_geometry'] = df_addresses.loc[closest_address_indices, 'geometry'].to_crs(crs_4326).reset_index(drop=True) #original crs
df_sampling['closest_address'] = df_addresses.loc[closest_address_indices, "full_address"].reset_index(drop=True) #original crs

# available points
available_points = df_sampling[['closest_address_geometry', 'closest_address']]
available_points.columns = ['geometry', 'full_address']
available_points = gpd.GeoDataFrame(available_points, geometry='geometry', crs=crs_4326)

# Plot all 'available locations' selected 

fig, ax = plt.subplots()
df_sampling.geometry.plot(ax=ax)
available_points.geometry.plot(ax=ax, color="red", markersize=2)

# Merge 311 and Survey123 data into one labelled_df gdf

df1 = df_311[['geometry', 'full_address']] # 311
df2 = df_survey123[['geometry', 'Approximate Street Address']] # survey123
df2.columns = ['geometry', 'full_address']

labelled_df = pd.concat([df1, df2]).reset_index(drop = True).set_geometry('geometry')

# Check that all point geometries fall within Boston boarders, else drop

# Function to extract invalid indices, i.e. indices in df1 that dont intersect with polygons of df2
def get_invalid_idx(df1, df2):
    '''
    df1: df to iterate over
    df2: df to get indices from 
    e.g. get indices of blocks for each address --> df1: address, df2: blocks
   '''
    join_df = gpd.sjoin(df1, df2, how='left', predicate ='intersects')
    invalid_idx = join_df[join_df.index_right.isna()].index
    
    return invalid_idx


# check geometries
invalid_idx = get_invalid_idx(labelled_df,df_sampling)

if len(invalid_idx) != 0:
    print(f'Complaints df has {len(invalid_idx)} invalid geometries. Drop these rows? (yes/no)')
    user_input = input().strip().lower()

    if user_input == 'yes':
        # Drop rows with invalid indices
        labelled_df = labelled_df.drop(index=invalid_idx)
        print(f'Rows with invalid indices have been dropped. Remaining rows: {len(labelled_df)}')
    elif user_input == 'no':
        print('No rows were dropped.')
    else:
        print('Invalid input. Please enter "yes" or "no".')
else:
    print('No invalid geometries found.')

def density_selection (labelled_points, sampled_points, available_points, n_points = 1, alpha = 1): 
    '''
    IN: 
    - df labelled_points: batch of labelled points, i.e. 311 rodent complaint locations
    - df sampled_points = sampled locations from previous iterations
    - df available_points: batch of available points, i.e. sam addresses across the whole city
    - int n_points: number of points to be selected at each iteration
    - float alpha: parameter for weighted average of minimum distances between sampled and labeled points
    '''

    # Project to right CRS
    labelled_points = labelled_points.to_crs(crs_32619)
    available_points = available_points.to_crs(crs_32619)
    sampled_points = sampled_points.to_crs(crs_32619)
    
    # Calculate distances between sampled points and available points
    sampled_distances = available_points.geometry.apply(lambda g: sampled_points.distance(g))
    min_sampled_distances = sampled_distances.apply(lambda row: min(np.array(row)), axis = 1)

    # Calculate distances between 311 points and available points
    labelled_distances = available_points.geometry.apply(lambda g: labelled_points.distance(g))
    min_labelled_distances = labelled_distances.apply(lambda row: min(np.array(row)), axis = 1)

    min_distances = (alpha)*min_sampled_distances + (1-alpha)*min_labelled_distances

    # Save to dataframe
    distances_df = pd.DataFrame(min_distances, index=available_points.index, columns=['distance'])
    
    # Select n=1 most dissimilar points
    distant_points = distances_df.nlargest(n_points, 'distance')
    chosen_points = available_points.loc[distant_points.index]
    
    # Add chosen points to sample set and remove chosen points from non-labelled set
    sampled_points_new = pd.concat([sampled_points, chosen_points])
    available_points_new = available_points.drop(chosen_points.index)

    return sampled_points_new, available_points_new

# Define distance function inputs

# labelled points
labelled_points = labelled_df

# available points
available_points = available_points

# sampled points
sampled_points = gpd.GeoDataFrame({'geometry': [], 'full_address': []}, geometry='geometry', crs=crs_4326) # initialize empty df to fill recursively through method

# density-based selection loop
iter = 0
max_iter = 400 # total number of points to be sampled

while iter < max_iter:
    sampled_points, available_points = density_selection(labelled_points, sampled_points, available_points, alpha = 0.3)
    iter += 1

# revert to 4326 crs 
density_based_points = sampled_points.to_crs(crs_4326)

# Plot results (labelled points & sampled points)

fig, ax = plt.subplots(figsize = (10,10))
df_sampling.geometry.plot(ax=ax) # census blocks
labelled_df.geometry.plot(ax=ax, color="white", markersize = 2) # already labelled points (311 + Survey123)
density_based_points.geometry.plot(ax=ax, color="red", markersize = 7) # density-selected sampled points
# fig.savefig('./density_sampled_max_min.png')

# assign features to sampled points

# sampled points
available_points = gpd.sjoin(density_based_points, df_sampling.iloc[:, :-2], how='left', predicate ='intersects').drop(columns = 'index_right')

# Select farthest data point

# function to mask nan values when taking euclidean distance between vectors
def nan_euclidean(u, v):
    '''
    - np.array u,v: arrays of vectors to calculate custom distance
    '''

    # mask NaNs in both vectors
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.any(mask): 
        return np.linalg.norm(u[mask] - v[mask]) # calculate distance between entries at indices where neither has NaNs
    else:
        return np.inf # if all NaNs, return a large distance 


# Function for Feature-Distribution based sampling --> Maximizing minimum distance

def distribution_selection_min(sampled_points, available_points, n_points=1):
    '''
    Select point(s) based on maximum dissimilarity between the already-chosen points
    IN:
    - df sampled_points = sampled locations from previous iterations
    - df available_points = points available for sampling
    - int n_points = number of points to select at this iteration (default 1)
    '''
    
    features = ['park_area', 'park_dist',
       'n_waste_bins', 'waste_dist', 'pop_density', 'bldg_density',
       'bldg_value_mean', 'bldg_value_min', 'age_mean', 'age_max',
       'yrs_since_remodel_mean', 'yrs_since_remodel_max',
       'overall_cond_score_mean', 'overall_cond_score_min', 'LU_AH', 'LU_C',
       'LU_CO', 'LU_E', 'LU_I', 'LU_R', 'LU_RC', 'EXT_asbestos',
       'EXT_brick_stone', 'EXT_metal_glass', 'EXT_other', 'EXT_wood',
       'EXT_vinyl', 'n_restaurants', 'restaurant_dist', 'n_establishments',
       'establishment_dist', 'sewer_length', 'sewer_length_per_m2',
       'sewers_junction_count', 'junction_count_per_m2',
       'sewers_access_points_count', 'access_points_per_m2',
       'brick_sewers_proportion', 'average_sewer_age',
       'sewers_condition_score', 'avg_sewer_width', 'avg_n_trash_days',
       'n_unique_trash_days']

    # turn df into numpy arrays for masking cdist
    sampled_array = sampled_points[features].to_numpy() #subset to feature columns
    available_array = available_points[features].to_numpy() #subset to feature columns
    
    # For each non-chosen/non-labelled point, get mean distance to all other points

    # Mean distance to sampled points
    distances = cdist(available_array, sampled_array, metric= nan_euclidean) # use ad-hoc metric to handle missing values in calculating euclidean distances
    min_distances = np.min(distances, axis=1)    
    
    # Save to dataframe
    dissimilarity_df = pd.DataFrame(min_distances, index=available_points.index, columns=['dissimilarity'])

    # Select n=1 most dissimilar points
    dissimilar_points = dissimilarity_df.nlargest(n_points, 'dissimilarity')
    chosen_points = available_points.loc[dissimilar_points.index]
    
    # Add chosen points to sample set and remove chosen points from non-labelled set
    sampled_points_new = pd.concat([sampled_points, chosen_points])
    available_points_new = available_points.drop(dissimilar_points.index)
    
    return sampled_points_new, available_points_new

# Max(Min(dist))

# available points
available_points = available_points

# sampled points: need to start with non-empty data here because there's no 'labelled_df' --> sample one random row of the available points df
np.random.seed(42)
random_idx = np.random.randint(0, available_points.shape[0])
sampled_points = available_points.iloc[[random_idx]] 

# feature distribution-based selection loop
iter = 0
max_iter = 199 # total number of points to be sampled (-1 since we start with one point already)

while iter < max_iter:
    sampled_points, available_points = distribution_selection_min(sampled_points, available_points)
    iter += 1

sampled_points.to_csv('../data/sampled_max_min.csv',index=False)

fig, axs = plt.subplots(len(sampled_points.columns)-3, 2, figsize=(14, 120))

for i, col in enumerate(sampled_points.columns[3:]):
    sampled_points.hist(column=col, ax=axs[i, 0])
    axs[i, 0].set_title(col + " sampled points")

    df_sampling.hist(column=col, ax=axs[i, 1])
    axs[i, 1].set_title(col + " all points")

# fig.savefig('./distributions_max_min.png')
plt.show()

# Merge points to wards
df_wards_points = gpd.sjoin(df_wards, sampled_points)

# Get number of points per ward
df_wards_counts = df_wards_points[['ward', 'geometry']].groupby('ward').count().reset_index()
df_wards_counts.columns = ['ward', 'counts']

# Merge with df_wards
df_wards_counts = df_wards_counts.merge(df_wards, on='ward')

# Get representative point per ward for plotting
df_wards_counts["rep_point"] = df_wards_counts.geometry.apply(lambda x: x.representative_point().coords[0])

# Convert to geodataframe
df_wards_counts = gpd.GeoDataFrame(df_wards_counts, geometry='geometry', crs=crs_4326)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))

# Ward Numbers
df_wards_counts.plot(ax=ax1, column='counts', cmap="Reds")
for i, row in df_wards_counts.iterrows():
    name = row['ward']
    ax1.annotate(text=name, xy=row['rep_point'],
                 horizontalalignment='center', fontsize=7)
ax1.set_title("Ward Numbers, Colored by Number of Points")

# Ward Counts
df_wards_counts.plot(ax=ax2)
sampled_points.plot(ax=ax2, color="white", markersize=2)
for i, row in df_wards_counts.iterrows():
    name = row['counts']
    ax2.annotate(text=name, xy=row['rep_point'],
                 horizontalalignment='center', fontsize=7)
ax2.set_title("Wards Labelled with Number of Points")

plt.show()

#######################################################################################################
# Get lat and lon for k-means clustering
sampled_points_clustering = sampled_points[["geometry", "full_address"]]
sampled_points_clustering["longitude"] = sampled_points_clustering.geometry.x
sampled_points_clustering["latitude"] = sampled_points_clustering.geometry.y

#######################################################################################################
# Fit k-means
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(sampled_points_clustering[['longitude', 'latitude']])

# Get clusters
sampled_points_clustering["cluster1"] = kmeans.labels_

#######################################################################################################
# Plot / analyze
fig, ax = plt.subplots(figsize=(10,10))
df_sampling.plot(ax=ax)
colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
for i, c in enumerate(sorted(sampled_points_clustering.cluster1.unique())):
    this_cluster = sampled_points_clustering[sampled_points_clustering.cluster1==c]
    this_cluster.plot(ax=ax, color=colors[i], markersize=8)
plt.show()

sampled_points_clustering.groupby('cluster1')['full_address'].count().reset_index(name = 'count_addresses')

# Function to get clusters through kmeans, constrained by balanced group size
def get_even_clusters(X, cluster_size, seed=0):
    '''
    From Stack Overflow:
    https://stackoverflow.com/questions/5452576/k-means-algorithm-variation-with-equal-cluster-size
    '''
    n_clusters = int(np.ceil(len(X)/cluster_size))
    kmeans = KMeans(n_clusters, random_state=seed)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
    return clusters

# Add this second clustering method to df
sampled_points_clustering["cluster2"] = get_even_clusters(sampled_points_clustering[['longitude', 'latitude']], 20, 11)

# Plot / analyze
fig, ax = plt.subplots(figsize=(10,10))
df_sampling.plot(ax=ax)
colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
for i, c in enumerate(sorted(sampled_points_clustering.cluster2.unique())):
    this_cluster = sampled_points_clustering[sampled_points_clustering.cluster2==c]
    # this_cluster.plot(ax=ax, color=colors[i], markersize=8)
    
    for j, row in this_cluster.iterrows():
        plt.annotate(text=j, xy=[row.longitude, row.latitude],
                    horizontalalignment='center', fontsize=6, color=colors[i])

plt.title("Points to Sample, Grouped into 10 Balanced-Size Clusters")
plt.show()

sampled_points_clustering.groupby('cluster2')['full_address'].count().reset_index(name = 'count_addresses')

# Slightly editing clusters to make sampling easier - manually as needed

cluster_pink = [484, 285]  # 8
cluster_violet = [115, 110]  # 4
cluster_red = [419]  # 0
cluster_orange = [491, 482] #1

sampled_points_clustering['cluster3'] = sampled_points_clustering['cluster2']

for i in cluster_pink:
    sampled_points_clustering.loc[i, 'cluster3'] = 8

for i in cluster_violet:
    sampled_points_clustering.loc[i, 'cluster3'] = 5

for i in cluster_red:
    sampled_points_clustering.loc[i, 'cluster3'] = 0
    
for i in cluster_orange:
    sampled_points_clustering.loc[i, 'cluster3'] = 1

# Plot / analyze

fig, ax = plt.subplots(figsize=(10,210))
df_sampling.plot(ax=ax)
colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
for i, c in enumerate(sorted(sampled_points_clustering.cluster3.unique())):
    this_cluster = sampled_points_clustering[sampled_points_clustering.cluster3==c]
    # this_cluster.plot(ax=ax, color=colors[i], markersize=8)
    
    for j, row in this_cluster.iterrows():
        plt.annotate(text=j, xy=[row.longitude, row.latitude],
                    horizontalalignment='center', fontsize=6, color=colors[i])

plt.title("Points to Sample, Grouped into 10 Clusters")
plt.show()

sampled_points_clustering.groupby('cluster3')['full_address'].count().reset_index(name = 'count_addresses')

# find wrong addresses
wrong_ads_df = sampled_points_clustering[sampled_points_clustering['full_address'].apply(lambda x: x[-5:-2] not in ['021', '022'])]
wrong_ads_idx = list(wrong_ads_df.index)

# find corresponding block
intersections_ads = [df_sampling.intersects(row.geometry) for i, row in wrong_ads_df.iterrows()]  # For each wrong ads get corresponding block
intersections_ads_idx = np.where(intersections_ads)[1] # for each point get index of corresponding block
wrong_block_df = df_sampling.iloc[intersections_ads_idx]

# for each block, get list of valid addresses
df_valid_addresses = df_addresses[df_addresses.full_address.apply(lambda x: x[-5:-2] in ['021', '022'])] # subset df_addresses to valid addresses
valid_ads = [df_valid_addresses.intersects(row.geometry) for i, row in wrong_block_df.iterrows()]  # For each block with wrong addresses, get list of addresses
indices_lists = [np.where(valid)[0].tolist() for valid in valid_ads] # For each wrong block get indices in df addresses of valid addresses

# Function to find the closest valid address
def find_min_distance_row(df, external_point):
    # Initialize variables for minimum distance calculation
    min_distance = float('inf')
    min_distance_row = None

    # Iterate over filtered rows to find minimum distance to external_point
    for i, row in df.iterrows():
        distance = row.geometry.distance(external_point)
        if distance < min_distance:
            min_distance = distance
            min_distance_row = row[['full_address', 'geometry', 'y_latitude', 'x_longitude']]

    return min_distance_row

####################################################################################################
# Initialize dictionary for valid addresses
ads_dict = {'id': [], 'full_address': [], 'geometry': [], 'y_latitude': [], 'x_longitude': []}

# Iterate through wrong addresses to find the closest valid address
for i in range(wrong_ads_df.shape[0]):

    # Get wrong geometry and index
    wrong_geom = wrong_ads_df.iloc[i].geometry
    wrong_id = wrong_ads_df.iloc[[i]].index.values[0]

    # Get valid addresses within that block
    ads = df_addresses.iloc[indices_lists[i]]

    # Select the closest one
    final_ad = find_min_distance_row(ads, wrong_geom)

    if final_ad is not None:
        # Append results to dictionary lists
        ads_dict['id'].append(wrong_id)
        ads_dict['full_address'].append(final_ad['full_address'])
        ads_dict['geometry'].append(final_ad['geometry'])
        ads_dict['y_latitude'].append(final_ad['y_latitude'])
        ads_dict['x_longitude'].append(final_ad['x_longitude'])

# Convert dictionary to DataFrame
ads_df = pd.DataFrame(ads_dict)

#substitute fulll addresses
for i in range(len(wrong_ads_idx)): 

    sampled_points_clustering.loc[wrong_ads_idx[i], 'full_address'] = ads_df.loc[i, 'full_address']
    sampled_points_clustering.loc[wrong_ads_idx[i], 'geometry'] = ads_df.loc[i, 'geometry']
    sampled_points_clustering.loc[wrong_ads_idx[i], 'latitude'] = ads_df.loc[i, 'y_latitude']
    sampled_points_clustering.loc[wrong_ads_idx[i], 'longitude'] = ads_df.loc[i, 'x_longitude']

# sanity check
current_wrong_ads = sampled_points_clustering[sampled_points_clustering['full_address'].apply(lambda x: x[-5:-2] not in ['021', '022'])]
print('Number of invalid rows: ', current_wrong_ads.shape[0])

# save
sampled_points_clustering.to_csv("../data/sample_addresses.csv")