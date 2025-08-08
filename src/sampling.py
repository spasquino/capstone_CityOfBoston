from pathlib import Path
from .config import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED, CRS_WGS84
from .utils import setup_logging, ensure_dir
import argparse

def sampling():
    # Packages

    import pandas as pd
    import numpy as np
    import re
    import geopandas as gpd
    from shapely import wkt
    from scipy.spatial.distance import cdist
    from collections import Counter
    import matplotlib.pyplot as plt
    import random

    # Coordinate reference systems
    crs_2249 = "EPSG:2249"
    crs_4326 = "EPSG:4326"  # lat-lon
    crs_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

    # Data path
    DATA_PATH = r"C:\Users\40009389\Documents\Rodents\data"


    # Block Features Data
    df_sampling = pd.read_csv("./sampling.csv")
    df_sampling.insert(0, "geometry", df_sampling['st_astext'].apply(wkt.loads))
    df_sampling = gpd.GeoDataFrame(df_sampling, geometry='geometry', crs=crs_2249)
    df_sampling.to_crs(crs_4326, inplace=True)
    df_sampling.drop(['bldg_value_min', 'LU_AH', 'EXT_other', 'EXT_vinyl'], axis=1, inplace=True)


    # 311 Labelled Data
    df_311 = pd.read_csv(DATA_PATH + r"\311_rodent_complaints.csv")
    df_311.insert(0, "geometry", df_311['st_astext'].apply(wkt.loads))
    df_311 = gpd.GeoDataFrame(df_311, geometry = 'geometry', crs = crs_4326)

    # subset to recent complaints (exact timeframe TBD)
    df_311 = df_311[df_311.open_dt > '2024-01-03'] # since March
    df_311.reset_index(drop=True, inplace=True)

    # add full address col
    df_311['full_address'] = df_311['location_street_number'].astype(str) + ' ' + df_311['location_street_name'] + ', ' + df_311['neighborhood'] + ', MA, ' + df_311['location_zipcode'].astype(str) + ', USA'


    # Survey123 Labelled Data
    df_survey123 = pd.read_csv(DATA_PATH + r"\survey123.csv")
    df_survey123 = gpd.GeoDataFrame(df_survey123, geometry=gpd.points_from_xy(df_survey123.x, df_survey123.y), crs= crs_4326)
    df_survey123['Current Date (MM/DD/YYYY)']= pd.to_datetime(df_survey123['Current Date (MM/DD/YYYY)'], format='%m/%d/%Y %I:%M:%S %p')
    df_survey123 = df_survey123[df_survey123['Current Date (MM/DD/YYYY)'] > '2024-03-01'] # since March


    # Sam Addresses Data 
    df_addresses = pd.read_csv(DATA_PATH + r"\addresses.csv")
    df_addresses.insert(0, "geometry", df_addresses['st_astext'].apply(wkt.loads))
    df_addresses = gpd.GeoDataFrame(df_addresses, geometry = 'geometry', crs = crs_4326)
    # Fix zip code
    df_addresses.zip_code = df_addresses.zip_code.apply(lambda x: '0'+str(x)[:-2])
    # Fix full address
    df_addresses['full_address'] = df_addresses.apply(lambda row: row.full_address + ", Boston, MA, " + row.zip_code, axis=1)


    # Ward Data
    df_wards = pd.read_csv(DATA_PATH + r"\wards.csv")
    df_wards.insert(0, "geometry", df_wards['st_astext'].apply(wkt.loads))
    df_wards = gpd.GeoDataFrame(df_wards, geometry='geometry', crs=crs_4326)
    df_wards = df_wards[['geometry', 'ward']]


    # Census Block Data
    df_blocks = pd.read_csv(DATA_PATH + "\census_blocks.csv")
    df_blocks.insert(0, "geometry", df_blocks['st_astext'].apply(wkt.loads))
    df_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry', crs=crs_2249)
    df_blocks.to_crs(crs_4326, inplace=True)

    # Select addresses closest to centroid of each census block (1 per block) in census block gdf

    closest_address_indices = [] # store closest address index for each block
    df_addresses_proj = df_addresses.to_crs(crs_32619) #apply projection crs
    df_sampling_proj = df_sampling.to_crs(crs_32619) #apply projection crs

    block_centroids = df_sampling.geometry.centroid
    block_centroids_proj = df_sampling_proj.geometry.centroid

    # iterate over each block centroid
    for i, block_centroid in enumerate(block_centroids):
        block_geometry = df_sampling.iloc[i].geometry
    
        # extract valid addresses for the block
        valid_addresses = df_addresses[df_addresses.intersects(block_geometry)]
        valid_addresses_proj = valid_addresses.to_crs(crs_32619)

        distances = valid_addresses_proj.geometry.apply(
            lambda g: block_centroids_proj[i].distance(g)
        )
    
        closest_idx = distances.idxmin()
    
        closest_address_indices.append(closest_idx)

    df_sampling['closest_address_geometry'] = df_addresses.loc[closest_address_indices, 'geometry'].to_crs(crs_4326).reset_index(drop=True) #original crs
    df_sampling['closest_address'] = df_addresses.loc[closest_address_indices, "full_address"].reset_index(drop=True) #original crs

    # Plot all selected addresses

    fig, ax = plt.subplots()
    df_sampling.geometry.plot(ax=ax)
    df_sampling.closest_address_geometry.plot(ax=ax, color="red", markersize=2)

    # Merge 311 and Survey123 data into one labelled_df gdf

    df1 = df_311[['geometry', 'full_address']]
    df2 = df_survey123[['geometry', 'Approximate Street Address']]
    df2.columns = ['geometry', 'full_address']
    labelled_df = pd.concat([df1, df2]).reset_index(drop = True).set_geometry('geometry')

    def density_selection (labelled_points, sampled_points, available_points, n_points = 1, alpha = 1): 
        '''
        IN: 
        - df labelled_points: batch of labelled points, i.e. 311 rodent complaint locations
        - df sampled_points = sampled locations from previous iterations
        - df available_points: batch of available points, i.e. sam addresses across the whole city
        - int n_points: number of points to be selected at each iteration
        - float alpha: parameter for weighted average of distance
        '''

        # Project to right CRS
        labelled_points = labelled_points.to_crs(crs_32619)
        available_points = available_points.to_crs(crs_32619)
        sampled_points = sampled_points.to_crs(crs_32619)
    
        # Calculate distances between sampled points and available points
        sampled_distances = available_points.geometry.apply(lambda g: sampled_points.distance(g))
        min_sampled_distances = sampled_distances.apply(lambda row: min(np.array(row)), axis = 1)

        # # Calculate distances between 311 points and available points
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
    available_points = df_sampling[['closest_address_geometry', 'closest_address']]
    available_points.columns = ['geometry', 'full_address']
    available_points = gpd.GeoDataFrame(available_points, geometry='geometry', crs=crs_4326)

    # sampled points
    sampled_points = gpd.GeoDataFrame({'geometry': [], 'full_address': []}, geometry='geometry', crs=crs_4326)

    # density-based selection loop
    iter = 0
    max_iter = 400 # total number of points to be sampled

    while iter < max_iter:
        sampled_points, available_points = density_selection(labelled_points, sampled_points, available_points, alpha = 0.3)
        iter += 1

    # revert to 4326 crs 
    sampled_points = sampled_points.to_crs(crs_4326)

    # merge with block feature gdf
    density_based_points = sampled_points.merge(df_sampling, right_on = 'closest_address', left_on = 'full_address', how = 'left')
    density_based_points.drop(columns = ['geometry_y', 'st_astext', 'closest_address_geometry', 'closest_address'], inplace = True)
    density_based_points.rename(columns = {'geometry_x': 'geometry'}, inplace = True)
    density_based_points = gpd.GeoDataFrame(density_based_points, geometry='geometry', crs=crs_4326)
    cols = list(density_based_points.columns)
    density_based_points = density_based_points.reindex(columns=cols[2:]+cols[:2])


    # Plot results (labelled points & sampled points)

    fig, ax = plt.subplots()
    df_sampling.to_crs(crs_4326).geometry.plot(ax=ax) # census blocks
    density_based_points.to_crs(crs_4326).geometry.plot(ax=ax, color="red", markersize=2) # density-selected sampled points
    labelled_df.geometry.plot(ax=ax, color="white", markersize=2) # already labelled points (311 + Survey123)
    fig.savefig('./density_sampled_max_min.png')

    # Map labelled points to corresponding blocks and assign features

    intersections_labelled = np.array([df_sampling.intersects(row.geometry) for i, row in labelled_df.iterrows()])  # assign to each sampled address its census block
    intersections_labelled_idx = np.where(intersections_labelled)[1] # for each point get index of corresponding block

    labelled_features = df_sampling.iloc[intersections_labelled_idx,:]
    labelled_features.drop(columns=['st_astext', 'geometry'], inplace=True)
    labelled_features.reset_index(drop=True, inplace=True)

    labelled_features['geometry'] = labelled_df.geometry
    labelled_features['full_address'] = labelled_df.full_address



    # Map sampled points to corresponding blocks and assign features

    # sampled_points = pd.read_csv(r"C:sampled_density.csv")
    # sampled_points['geometry'] = sampled_points['geometry'].apply(wkt.loads)
    # sampled_points = gpd.GeoDataFrame(sampled_points, geometry='geometry', crs=crs_4326)

    # # extract block-sampled point intersection
    # intersections_sampled =np.array([df_sampling.intersects(row.geometry) for i,row in sampled_points_mean.iterrows()]) #assign to each sampled address its census block
    # intersections_sampled_idx = np.where(intersections_sampled)[1] # for each point get index of corresponding block

    # # extract block's features
    # sampled_features = df_sampling.iloc[intersections_sampled_idx,:]
    # sampled_features.drop(columns=['st_astext', 'geometry'], inplace=True)
    # sampled_features.reset_index(drop=True, inplace=True)
    # sampled_features['geometry'] = sampled_points['geometry']
    # sampled_features['full_address'] = sampled_points['full_address']

    # Select farthest data point

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


    # Maximizing average distance

    def distribution_selection_mean(sampled_points, available_points, n_points=1):
        '''
        Select point(s) based on maximum dissimilarity between the already-chosen points
        IN:
        - df sampled_points = sampled locations from previous iterations
        - df available_points = points available for sampling
        - int n_points = number of points to select at this iteration (default 1)
        '''
    
        # turn df into numpy arrays for masking cdist
        sampled_array = sampled_points.iloc[:,:-2].to_numpy() #exclude location and address cols
        available_array = available_points.iloc[:,:-2].to_numpy() #exclude location and address cols
    
        # For each non-chosen/non-labelled point, get mean distance to all other points

        # Mean distance to sampled points
        distances = cdist(available_array, sampled_array, metric= nan_euclidean)
        mean_distances = np.mean(distances, axis=1)    
    
        # Save to dataframe
        dissimilarity_df = pd.DataFrame(mean_distances, index=available_points.index, columns=['dissimilarity'])

        # Select n=1 most dissimilar points
        dissimilar_points = dissimilarity_df.nlargest(n_points, 'dissimilarity')
        chosen_points = available_points.loc[dissimilar_points.index]
    
        # Add chosen points to sample set and remove chosen points from non-labelled set
        sampled_points_new = pd.concat([sampled_points, chosen_points])
        available_points_new = available_points.drop(dissimilar_points.index)
    
        return sampled_points_new, available_points_new


    # Maximizing minimum distance

    def distribution_selection_min(sampled_points, available_points, n_points=1):
        '''
        Select point(s) based on maximum dissimilarity between the already-chosen points
        IN:
        - df sampled_points = sampled locations from previous iterations
        - df available_points = points available for sampling
        - int n_points = number of points to select at this iteration (default 1)
        '''
    
        # turn df into numpy arrays for masking cdist
        sampled_array = sampled_points.iloc[:,:-2].to_numpy() #exclude location and address cols
        available_array = available_points.iloc[:,:-2].to_numpy() #exclude location and address cols
    
        # For each non-chosen/non-labelled point, get mean distance to all other points

        # Mean distance to sampled points
        distances = cdist(available_array, sampled_array, metric= nan_euclidean)
        mean_distances = np.min(distances, axis=1)    
    
        # Save to dataframe
        dissimilarity_df = pd.DataFrame(mean_distances, index=available_points.index, columns=['dissimilarity'])

        # Select n=1 most dissimilar points
        dissimilar_points = dissimilarity_df.nlargest(n_points, 'dissimilarity')
        chosen_points = available_points.loc[dissimilar_points.index]
    
        # Add chosen points to sample set and remove chosen points from non-labelled set
        sampled_points_new = pd.concat([sampled_points, chosen_points])
        available_points_new = available_points.drop(dissimilar_points.index)
    
        return sampled_points_new, available_points_new

    # Max(Mean(dist))

    # available points
    available_points = density_based_points
    # available_points.drop(columns = ['closest_address_geometry', 'closest_address'], inplace = True)

    # sampled points with max(mean()) approach
    sampled_points_mean = gpd.GeoDataFrame({
        'park_area': [], 
        'park_dist': [],
        'n_waste_bins': [], 
        'waste_dist': [], 
        'pop_density': [],
        'bldg_density': [],
        'bldg_value_mean': [],
        'age_mean': [],
        'age_max': [], 
        'yrs_since_remodel_mean': [], 
        'yrs_since_remodel_max': [],
        'overall_cond_score_mean': [],
        'overall_cond_score_min': [], 
        'LU_C': [],
        'LU_CO': [],
        'LU_E': [], 
        'LU_I': [], 
        'LU_R': [], 
        'LU_RC': [], 
        'EXT_asbestos': [],
        'EXT_brick_stone': [], 
        'EXT_metal_glass': [], 
        'EXT_wood': [],
        'n_restaurants': [], 
        'restaurant_dist': [],
        'geometry': [], 
        'full_address': []}, geometry='geometry', crs=crs_4326)

    # density-based selection loop
    iter = 0
    max_iter = 200 # total number of points to be sampled

    while iter < max_iter:
        sampled_points_mean, available_points = distribution_selection_mean(sampled_points_mean, available_points)
        iter += 1

    sampled_points_mean.to_csv('./sampled_max_mean.csv',index=False)


    fig, axs = plt.subplots(len(sampled_points_mean.columns)-2, 2, figsize=(14, 70))

    for i, col in enumerate(sampled_points_mean.columns[:-2]):
        sampled_points_mean.hist(column=col, ax=axs[i, 0])
        axs[i, 0].set_title(col + " sampled points")

        df_sampling.hist(column=col, ax=axs[i, 1])
        axs[i, 1].set_title(col + " all points")

    fig.savefig('./distributions_max_mean.png')
    plt.show()

    # Max(Min(dist))

    # available points
    available_points = density_based_points

    # sampled points with max(min()) approach
    # sampled_points_min = available_points.sample(n=1)
    sampled_points_min = available_points.iloc[[292]]  # This was the random sample that we started with. Hard-Coding because there is no seed.

    # density-based selection loop
    iter = 0
    max_iter = 199 # total number of points to be sampled (-1 since we start with one point already)

    while iter < max_iter:
        sampled_points_min, available_points = distribution_selection_min(sampled_points_min, available_points)
        iter += 1

    sampled_points_min.to_csv('./sampled_max_min.csv',index=False)

    fig, axs = plt.subplots(len(sampled_points_min.columns)-2, 2, figsize=(14, 70))

    for i, col in enumerate(sampled_points_min.columns[:-2]):
        sampled_points_min.hist(column=col, ax=axs[i, 0])
        axs[i, 0].set_title(col + " sampled points")

        df_sampling.hist(column=col, ax=axs[i, 1])
        axs[i, 1].set_title(col + " all points")

    # fig.savefig('./distributions_max_min.png')
    plt.show()

    # Plotting Distributions
    fig, axs = plt.subplots(len(sampled_points_min.columns)-2, 3, figsize=(14, 70))

    for i, col in enumerate(sampled_points_min.columns[:-2]):
        sampled_points_mean.hist(column=col, ax=axs[i, 0])
        axs[i, 0].set_title(col + " MEAN sampled points")
    
        sampled_points_min.hist(column=col, ax=axs[i, 1])
        axs[i, 1].set_title(col + " MIN sampled points")

        df_sampling.hist(column=col, ax=axs[i, 2])
        axs[i, 2].set_title(col + " all points")

    fig.savefig('./distributions_max_min.png')
    plt.show()

    # Mapping selected points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # MEAN
    df_sampling.to_crs(crs_4326).geometry.plot(ax=ax1) # census blocks
    sampled_points_mean.to_crs(crs_4326).geometry.plot(ax=ax1, color="red", markersize=2) # density-selected sampled points
    labelled_df.geometry.plot(ax=ax1, color="white", markersize=2) # already labelled points (311 + Survey123)
    ax1.set_title("Mean")

    # MIN
    df_sampling.to_crs(crs_4326).geometry.plot(ax=ax2) # census blocks
    sampled_points_min.to_crs(crs_4326).geometry.plot(ax=ax2, color="red", markersize=2) # density-selected sampled points
    labelled_df.geometry.plot(ax=ax2, color="white", markersize=2) # already labelled points (311 + Survey123)
    ax2.set_title("Min")

    plt.show()

    # How many points do the sampling methodologies share?

    # Get block IDs for mean
    intersections_mean =np.array([df_sampling.intersects(row.geometry) for i,row in sampled_points_mean.iterrows()]) #assign to each sampled address its census block
    intersections_mean_idx = np.where(intersections_mean)[1] # for each point get index of corresponding block

    # Get block IDs for min
    intersections_min =np.array([df_sampling.intersects(row.geometry) for i,row in sampled_points_min.iterrows()]) #assign to each sampled address its census block
    intersections_min_idx = np.where(intersections_min)[1] # for each point get index of corresponding block

    # Get number of shared blocks
    shared_idx = list(set(intersections_mean_idx) & set(intersections_min_idx))
    len(shared_idx)

    # Merge points to wards
    df_wards_points = gpd.sjoin(df_wards, sampled_points_min)

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
    sampled_points_min.plot(ax=ax2, color="white", markersize=2)
    for i, row in df_wards_counts.iterrows():
        name = row['counts']
        ax2.annotate(text=name, xy=row['rep_point'],
                     horizontalalignment='center', fontsize=7)
    ax2.set_title("Wards Labelled with Number of Points")

    plt.show()

    # Need only lat and lon for k-means clustering
    sampled_points_min_clustering = sampled_points_min[["geometry", "full_address"]]
    sampled_points_min_clustering["longitude"] = sampled_points_min_clustering.geometry.x
    sampled_points_min_clustering["latitude"] = sampled_points_min_clustering.geometry.y

    # K-Means Clustering

    from sklearn.cluster import KMeans

    # Fit k-means
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(sampled_points_min_clustering[['longitude', 'latitude']])

    # Get clusters
    sampled_points_min_clustering["cluster1"] = kmeans.labels_

    # Plot / analyze
    fig, ax = plt.subplots(figsize=(10,10))
    df_wards.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white', 'black', 'pink', 'brown']
    for i, c in enumerate(sorted(sampled_points_min_clustering.cluster1.unique())):
        this_cluster = sampled_points_min_clustering[sampled_points_min_clustering.cluster1==c]
        this_cluster.plot(ax=ax, color=colors[i], markersize=2)
    plt.show()

    sampled_points_min_clustering.groupby('cluster1').count()

    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    import numpy as np

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

    sampled_points_min_clustering["cluster2"] = get_even_clusters(sampled_points_min_clustering[['longitude', 'latitude']], 20, 11)

    # Plot / analyze
    fig, ax = plt.subplots(figsize=(20,20))
    df_wards.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
    for i, c in enumerate(sorted(sampled_points_min_clustering.cluster2.unique())):
        this_cluster = sampled_points_min_clustering[sampled_points_min_clustering.cluster2==c]
        # this_cluster.plot(ax=ax, color=colors[i], markersize=7)
    
        for j, row in this_cluster.iterrows():
            plt.annotate(text=j, xy=[row.longitude, row.latitude],
                        horizontalalignment='center', fontsize=6, color=colors[i])

    plt.title("Points to Sample, Grouped into 10 Clusters")
    plt.show()

    sampled_points_min_clustering.groupby('cluster2').count()

    # Slightly editing clusters to make sampling easier
    cluster_red = [182, 22]  # 0
    cluster_black = [75]  # 7
    cluster_white = [281]  # 6

    sampled_points_min_clustering['cluster3'] = sampled_points_min_clustering['cluster2']

    for i in cluster_red:
        sampled_points_min_clustering.loc[i, 'cluster3'] = 0

    for i in cluster_white:
        sampled_points_min_clustering.loc[i, 'cluster3'] = 6

    for i in cluster_black:
        sampled_points_min_clustering.loc[i, 'cluster3'] = 7

    # Plot / analyze
    fig, ax = plt.subplots(figsize=(20,20))
    df_wards.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
    for i, c in enumerate(sorted(sampled_points_min_clustering.cluster3.unique())):
        this_cluster = sampled_points_min_clustering[sampled_points_min_clustering.cluster3==c]
        # this_cluster.plot(ax=ax, color=colors[i], markersize=7)
    
        for j, row in this_cluster.iterrows():
            plt.annotate(text=j, xy=[row.longitude, row.latitude],
                        horizontalalignment='center', fontsize=6, color=colors[i])

    plt.title("Points to Sample, Grouped into 10 Clusters")
    plt.show()

    sampled_points_min_clustering.groupby('cluster3').count()

    # Notes on which addresses to switch and why
    '''
    525 William F McClellan Hwy, Boston, MA, 02128: Highway
    310 William F McClellan Hwy #H, Boston, MA, 02128: Highway
    300 Terminal C, Boston, MA, 02128: Airport
    60 Western Ave, Boston, MA, 02134: Big parking lot
    100 Nashua St, Boston, MA, 02114: Maybe on a highway
    20 Eldon St, Boston, MA, 02131: Park
    155 Rivermoor St, Boston, MA, 02132: Auto dealership
    20 Canaan St, Boston, MA, 02126: Park
    282-308 Bremen St, Boston, MA, 02128: Construction site on Google
    220 Porter St, Boston, MA, 02128: Park
    75 VFW Pkwy, Boston, MA, 02131: Park
    '''

    invalid_addresses = [
        "525 William F McClellan Hwy, Boston, MA, 02128",
        "310 William F McClellan Hwy #H, Boston, MA, 02128",
        "300 Terminal C, Boston, MA, 02128",
        "60 Western Ave, Boston, MA, 02134",
        "100 Nashua St, Boston, MA, 02114",
        "20 Eldon St, Boston, MA, 02131",
        "155 Rivermoor St, Boston, MA, 02132",
        "20 Canaan St, Boston, MA, 02126",
        "282-308 Bremen St, Boston, MA, 02128",
        "220 Porter St, Boston, MA, 02128",
        "75 VFW Pkwy, Boston, MA, 02131"
    ]


    # Merge samples and blocks, subset to only invalid addresses

    sampled_points_blocks = gpd.sjoin(sampled_points_min, df_blocks[['geometry', 'gid']])
    sampled_points_blocks = sampled_points_blocks[sampled_points_blocks.full_address.isin(invalid_addresses)]

    # For each invalid address, find a new valid address

    # Current address
    i = 0
    row = sampled_points_blocks.iloc[i, :]

    # Get block corresponding to this address
    block_row = df_blocks[df_blocks.gid==row.gid]
    block_geometry = block_row.geometry.values[0]

    # Get all addresses in this block
    block_addresses = df_addresses[df_addresses.intersects(block_geometry)].reset_index(drop=True)
    block_addresses['longitude'] = block_addresses.geometry.x
    block_addresses['latitude'] = block_addresses.geometry.y

    # Plot to see which addresses are spatially appropriate
    fig, ax = plt.subplots(figsize=(10,10))
    block_row.plot(ax=ax)
    for j, row in block_addresses.iterrows():
        plt.annotate(text=j, xy=[row.longitude, row.latitude],
                    horizontalalignment='center', fontsize=8, color='white')


    # Explore what address each index corresponds to, in order to find an appropriate new address for this block
    block_addresses.iloc[[0]]

    new_addresses = [
        "111 Waldemar Ave, Boston, MA, 02128",
        "144 Addison St, Boston, MA, 02128",
        "21 Lovell St, Boston, MA, 02128",
        "69 Hopedale St, Boston, MA, 02134",
        "1 Nashua St, Boston, MA, 02114",
        "37 Eldon St, Boston, MA, 02131",
        "221 Rivermoor St, Boston, MA, 02132",
        "223 Itasca St, Boston, MA, 02126",
        "244 Bremen St #3, Boston, MA, 02128",
        "226 Porter St, Boston, MA, 02128",
        "14 Hackensack Ct, Boston, MA, 02467",
    ]

    # Changing out addresses and geometries for sampled_points_min

    for i in range(len(invalid_addresses)):
        # Get old address and new address to replace it with
        invalid_address = invalid_addresses[i]
        new_address = new_addresses[i]
    
        # Get index of sample whose address needs to be changed
        idx = sampled_points_min[sampled_points_min.full_address == invalid_address].index[0]
    
        # Get geometry of new address
        new_geom = df_addresses[df_addresses.full_address==new_address].geometry.values[0]
    
        # Replace address and geometry of sample with new address and new geometry
        sampled_points_min.loc[idx, 'full_address'] = new_address
        sampled_points_min.loc[idx, 'geometry'] = new_geom

    # Changing out addresses and geometries for sampled_points_min_clustering 

    sampled_points_min_clustering.geometry = sampled_points_min.geometry
    sampled_points_min_clustering.full_address = sampled_points_min.full_address

    # Plot / analyze
    fig, ax = plt.subplots(figsize=(20,20))
    df_wards.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
    for i, c in enumerate(sorted(sampled_points_min_clustering.cluster3.unique())):
        this_cluster = sampled_points_min_clustering[sampled_points_min_clustering.cluster3==c]
        # this_cluster.plot(ax=ax, color=colors[i], markersize=7)
    
        for j, row in this_cluster.iterrows():
            plt.annotate(text=j, xy=[row.longitude, row.latitude],
                        horizontalalignment='center', fontsize=6, color=colors[i])

    plt.title("Points to Sample, Grouped into 10 Clusters")
    plt.show()

    sampled_points_min_clustering.groupby('cluster3').count()

    df_final = sampled_points_min_clustering[['full_address', 'cluster3', 'geometry', 'longitude', 'latitude']]
    df_final.columns = ['address', 'cluster', 'geometry', 'longitude', 'latitude']
    df_final = df_final.sort_values('cluster').reset_index(drop=True)
    # df_final.to_csv("./sample_addresses.csv")
    df_final

    # Plot / analyze
    fig, ax = plt.subplots(figsize=(15,15))
    df_wards.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
    for i, c in enumerate(sorted(df_final.cluster.unique())):
        this_cluster = df_final[df_final.cluster==c]
        this_cluster.plot(ax=ax, color=colors[i], markersize=7, label=f"Cluster {i}")
    plt.legend()
    plt.title("Points to Sample by Cluster")

    # fig.savefig('./sample_points.png')

    plt.show()


    # Plot / analyze
    fig, ax = plt.subplots(figsize=(20,20))
    df_wards.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'black', 'deeppink', 'brown']
    for i, c in enumerate(sorted(df_final.cluster.unique())):
        this_cluster = df_final[df_final.cluster==c]

        for j, row in this_cluster.iterrows():
            plt.annotate(text=j, xy=[row.longitude, row.latitude],
                        horizontalalignment='center', fontsize=6, color=colors[i])


    plt.title("Points to Sample, Grouped into 10 Clusters")

    # fig.savefig('./sample_indices.png')

    plt.show()

    df_counts = df_final.groupby('cluster').count().reset_index()
    df_counts = df_counts[['cluster', 'address']]
    df_counts.columns = ['cluster', 'num_points']
    # df_counts.to_csv("./sample_cluster_counts.csv", index=False)
    df_counts

    # read in exported data

    # sampling: note that now sample_addresses.csv contains all the right addresses
    sampling_df = pd.read_csv('./sample_addresses_old.csv', index_col = False)
    sampling_df.drop(columns = ['Unnamed: 0'], inplace = True)
    sampling_df.rename(columns = {'geometry': 'st_astext'}, inplace = True)
    sampling_df.insert(0, "geometry", sampling_df['st_astext'].apply(wkt.loads))
    sampling_df = gpd.GeoDataFrame(sampling_df, geometry='geometry', crs=crs_4326)


    # # blocks
    block_df = pd.read_csv('./sampling.csv')
    block_df.insert(0, "geometry", block_df['st_astext'].apply(wkt.loads))
    block_df = gpd.GeoDataFrame(block_df, geometry='geometry', crs=crs_2249)
    block_df.to_crs(crs_4326, inplace = True)

    # addresses 
    df_addresses = pd.read_csv(r"C:\Users\40009382\Documents\Capstone_repo\data\addresses.csv")
    df_addresses.insert(0, "geometry", df_addresses['st_astext'].apply(wkt.loads))
    df_addresses = gpd.GeoDataFrame(df_addresses, geometry = 'geometry', crs = crs_4326)
    # Fix zip code
    df_addresses.zip_code = df_addresses.zip_code.apply(lambda x: '0'+str(x)[:-2])
    # Fix full address
    df_addresses['full_address'] = df_addresses.apply(lambda row: row.full_address + ", Boston, MA, " + row.zip_code, axis=1)


    # find wrong addresses
    wrong_ads_df = sampling_df[sampling_df['address'].apply(lambda x: x[-5:-2] not in ['021', '022'])]
    wrong_ads_idx = list(wrong_ads_df.index)

    # find corresponding block
    intersections_ads = [block_df.intersects(row.geometry) for i, row in wrong_ads_df.iterrows()]  # For each wrong ads get corresponding block
    intersections_ads_idx = np.where(intersections_ads)[1] # for each point get index of corresponding block
    wrong_block_df = block_df.iloc[intersections_ads_idx]



    # for each block, get list of valid addresses
    valid_ads = [df_addresses.intersects(row.geometry) for i, row in wrong_block_df.iterrows()]  # For each block, get list of addresses
    indices_lists = [np.where(valid)[0].tolist() for valid in valid_ads]

    # function to select among these the closest one to the invalid one
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

    # create a df for each valid address in the block of the formerly invalid one

    block_1 = df_addresses.iloc[indices_lists[0]]
    block_1 = block_1[block_1['full_address'].apply(lambda x: x[-5:-2] in ['021', '022'])]

    block_2 = df_addresses.iloc[indices_lists[1]]
    block_2 = block_2[block_2['full_address'].apply(lambda x: x[-5:-2] in ['021', '022'])]

    block_3 = df_addresses.iloc[indices_lists[2]]
    block_3 = block_3[block_3['full_address'].apply(lambda x: x[-5:-2] in ['021', '022'])]

    row_1 = find_min_distance_row(block_1, wrong_ads_df.iloc[0,0])
    row_2 = find_min_distance_row(block_2, wrong_ads_df.iloc[1,0])
    row_3 = find_min_distance_row(block_3, wrong_ads_df.iloc[2,0])

    print(sampling_df.iloc[wrong_ads_idx[0]].address)
    print(row_1.full_address)

    #substitute fulll addresses
    sampling_df.loc[wrong_ads_idx[0], 'address'] = row_1.full_address
    sampling_df.loc[wrong_ads_idx[1],'address'] = row_2.full_address
    sampling_df.loc[wrong_ads_idx[2],'address'] = row_3.full_address

    #substitute geometry
    sampling_df.loc[wrong_ads_idx[0],'geometry'] = row_1.geometry
    sampling_df.loc[wrong_ads_idx[1],'geometry'] = row_2.geometry
    sampling_df.loc[wrong_ads_idx[2],'geometry'] = row_3.geometry

    #substitute latitude
    sampling_df.loc[wrong_ads_idx[0],'latitude'] = row_1.y_latitude
    sampling_df.loc[wrong_ads_idx[1],'latitude'] = row_2.y_latitude
    sampling_df.loc[wrong_ads_idx[2],'latitude'] = row_3.y_latitude

    #substitute longitude
    sampling_df.loc[wrong_ads_idx[0],'longitude'] = row_1.x_longitude
    sampling_df.loc[wrong_ads_idx[1],'longitude'] = row_2.x_longitude
    sampling_df.loc[wrong_ads_idx[2],'longitude'] = row_3.x_longitude


    # sanity check
    current_wrong_ads = sampling_df[sampling_df['address'].apply(lambda x: x[-5:-2] not in ['021', '022'])]
    print('Number of invalid rows: ', current_wrong_ads.shape[0])

    #make wrong_ads_df geometric to plot
    wrong_ads_df = gpd.GeoDataFrame(wrong_ads_df, geometry='geometry', crs=crs_4326)

    # plot check
    fig, ax = plt.subplots(figsize = (10,15))
    df_wards.to_crs(crs_4326).geometry.plot(ax=ax, color = 'grey') # census blocks
    sampling_df.iloc[wrong_ads_idx].to_crs(crs_4326).geometry.plot(ax=ax, color="black", markersize=4) # new sampled points 
    wrong_ads_df.to_crs(crs_4326).geometry.plot(ax=ax, color="red", markersize=4) # old sampled points
    # sampling_df.iloc[~wrong_ads_idx].to_crs(crs_4326).geometry.plot(ax=ax, color="black", markersize=2) # sampled points (excluding wrong)
    # fig.savefig('./final_sample.png')


    # extract zip codes for ordering
    sampling_df['zip_code'] = sampling_df.address.str[-5:]
    sampling_df.sort_values(by = ['cluster', 'zip_code', 'latitude', 'longitude'], inplace = True)
    sampling_df.reset_index(drop = True)

    # plot to validate
    cluster = sampling_df[sampling_df.cluster == 0].reset_index(drop = True)

    fig, ax = plt.subplots(figsize = (10,5))
    df_wards.to_crs(crs_4326).geometry.plot(ax=ax, color = 'white', edgecolor = 'black') # census blocks

    for idx, row in cluster_0.iterrows():
        ax.annotate(str(row.order), xy=(row.geometry.x, row.geometry.y),
                    xytext=(3, 3), textcoords='offset points', color='blue', fontsize=10)

    # Get the extent of cluster points for setting the plot limits
    x_min, y_min, x_max, y_max = cluster_0.total_bounds
    buffer = 0.01  # Adjust the buffer for zooming in

    # Set plot limits to zoom in around the cluster points
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)


    mapping_dict_0 = {0: 7,
    1: 1,
    2: 8,
    3: 21,
    4: 4,
    5: 3,
    6: 2,
    7: 5,
    8: 14,
    9: 6,
    10: 13,
    11: 12,
    12: 16,
    13: 15,
    14: 11,
    15: 17,
    16: 9,
    17: 10,
    18: 18,
    19: 19,
    20: 20}

    mapping_dict_2 = {0: 20,
    1: 19,
    2: 14,
    3: 11,
    4: 13,
    5: 12,
    6: 15,
    7: 18,
    8: 16,
    9: 17,
    10: 3,
    11: 5,
    12: 4,
    13: 2,
    14: 6,
    15: 1,
    16: 7,
    17: 8,
    18: 10,
    19: 9}

    mapping_dict_4 = {0: 18,
    1: 17,
    2: 16,
    3: 15,
    4: 4,
    5: 6,
    6: 5,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 1,
    12: 2,
    13: 3,
    14: 20,
    15: 19,
    16: 7,
    17: 12,
    18: 13,
    19: 14}

    mapping_dict_6 = {0: 10,
    1: 9,
    2: 8,
    3: 7,
    4: 5,
    5: 1,
    6: 4,
    7: 2,
    8: 17,
    9: 20,
    10: 21,
    11: 18,
    12: 15,
    13: 19,
    14: 14,
    15: 16,
    16: 11,
    17: 12,
    18: 13,
    19: 3,
    20: 6}

    mapping_dict_8 = {0: 13,
    1: 14,
    2: 11,
    3: 10,
    4: 9,
    5: 19,
    6: 20,
    7: 18,
    8: 12,
    9: 7,
    10: 6,
    11: 17,
    12: 15,
    13: 16,
    14: 5,
    15: 4,
    16: 3,
    17: 2,
    18: 1,
    19: 8}


    cluster_0 = sampling_df[sampling_df.cluster == 0].reset_index(drop = True)
    cluster_2 = sampling_df[sampling_df.cluster == 2].reset_index(drop = True)
    cluster_4 = sampling_df[sampling_df.cluster == 4].reset_index(drop = True)
    cluster_6 = sampling_df[sampling_df.cluster == 6].reset_index(drop = True)
    cluster_8 = sampling_df[sampling_df.cluster == 8].reset_index(drop = True)



    order_0 = [value for key,value in mapping_dict_0.items()]
    cluster_0['order'] = order_0

    order_2 = [value for key,value in mapping_dict_2.items()]
    cluster_2['order'] = order_2

    order_4 = [value for key,value in mapping_dict_4.items()]
    cluster_4['order'] = order_4

    order_6 = [value for key,value in mapping_dict_6.items()]
    cluster_6['order'] = order_6

    order_8 = [value for key,value in mapping_dict_8.items()]
    cluster_8['order'] = order_8

    # concatenate and save
    even_samples = pd.concat([cluster_0, cluster_2, cluster_4, cluster_6, cluster_8], axis=0).reset_index(drop = True)
    even_samples.to_csv("./even_samples.csv")

    # format and save
    # sampling_df.drop(columns = ['st_astext'], inplace = True)
    sampling_df.to_csv("./sample_addresses.csv")

    # Sample Data
    sampling_df = pd.read_csv('./sample_addresses_reordered.csv')
    sampling_df['geometry'] = sampling_df['geometry'].apply(wkt.loads)
    sampling_df = gpd.GeoDataFrame(sampling_df, geometry='geometry', crs=crs_4326)

    # Ward Data
    df_wards = pd.read_csv(r"C:\Users\40009382\Documents\Capstone_repo\data\wards.csv")
    df_wards.insert(0, "geometry", df_wards['st_astext'].apply(wkt.loads))
    df_wards = gpd.GeoDataFrame(df_wards, geometry='geometry', crs=crs_4326)
    df_wards = df_wards[['geometry', 'ward']]

    # Census Block Data
    df_blocks = pd.read_csv(r"C:\Users\40009382\Documents\Capstone_repo\data\census_blocks.csv")
    df_blocks.insert(0, "geometry", df_blocks['st_astext'].apply(wkt.loads))
    df_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry', crs=crs_2249)
    df_blocks.to_crs(crs_4326, inplace=True)

    # Plot / analyze
    fig, ax = plt.subplots(figsize=(6,6))
    df_blocks.plot(ax=ax)
    colors = ['red', 'orange', 'yellow', 'lime', 'blue', 'violet', 'white', 'darkgray', 'indigo', 'brown']
    for i, c in enumerate(sorted(sampling_df.cluster.unique())):
        this_cluster = sampling_df[sampling_df.cluster==c]
        this_cluster.plot(ax=ax, color=colors[i], markersize=3, label=f"Cluster {i}")
    plt.legend()
    plt.title("Points to Sample by Cluster")

    # fig.savefig('./sample_points.png')

    plt.show()





# Backward compatibility: call into original function if present
def run_sampling(**kwargs):
    """
    Public entry point for this step. Accepts keyword args to customize behavior.
    """
    logger = setup_logging()
    logger.info("Starting run_sampling()")
    if "sampling" in globals():
        return sampling(**kwargs) if callable(globals().get("sampling")) else None
    # Fallback: nothing to run
    logger.warning("Original function 'sampling()' not found; nothing executed.")
    return None

def _cli():
    parser = argparse.ArgumentParser(description="run_sampling step")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to data dir")
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR), help="Path to figures dir")
    args = parser.parse_args()
    ensure_dir(Path(args.figures_dir))
    return run_sampling()

if __name__ == "__main__":
    _cli()
