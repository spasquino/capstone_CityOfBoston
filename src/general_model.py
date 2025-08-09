# Packages

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import random
from datetime import timedelta
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier,  plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from shapely.geometry import Point
from sklearn.utils import resample

# Coordinate reference systems
crs_2249 = "EPSG:2249"
crs_4326 = "EPSG:4326"  # lat-lon
crs_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

# Define function to extract 'invalid' point geometry indexes, i.e. locations that don't fall within Boston blocks 

def get_invalid_idx(df1, df2):
    '''
    df1: df to iterate over
    df2: df to get indices from 
    e.g. get indices of blocks for each address --> df1: address, df2: blocks
   '''
    join_df = gpd.sjoin(df1, df2, how='left', predicate ='intersects')
    invalid_idx = join_df[join_df.index_right.isna()].index
    
    return invalid_idx

# Data path
DATA_PATH = r"C:\Users\40009382\Documents\Capstone_repo\data"

######################################################################################################################################################################################################
# Block-Wise Features

features_df = pd.read_csv(r'C:\Users\40009382\Documents\GitHub\analytics-mit-capstone-2024\data\features.csv')
# turn into geodataframe
features_df['geometry'] = wkt.loads(features_df.geometry)
features_df = gpd.GeoDataFrame(features_df, geometry = 'geometry', crs = crs_4326)
features_df['bid'] = features_df.reset_index().index #create block id through indices
features_df.drop(columns = 'gid', inplace = True)

######################################################################################################################################################################################################
# Violations data
violations_df = gpd.read_file(DATA_PATH + r"\rodent_violations\rodent_violations.shp")
violations_df.to_crs(crs_4326, inplace = True)

# subset to Summer 2024 violations 
violations_df = violations_df[violations_df.casedttm >= '2024-06-01'] # since June 1st, 2024
violations_df.reset_index(drop = True, inplace = True)

# check geometries
invalid_idx = get_invalid_idx(violations_df,features_df)

if len(invalid_idx) != 0:
    print(f'Violations df has {len(invalid_idx)} invalid geometries. Drop these rows? (yes/no)')
    user_input = input().strip().lower()

    if user_input == 'yes':
        # Drop rows with invalid indices
        violations_df = violations_df.drop(index=invalid_idx)
        print(f'Rows with invalid indices have been dropped. Remaining rows: {len(violations_df)}')
    elif user_input == 'no':
        print('No rows were dropped.')
    else:
        print('Invalid input. Please enter "yes" or "no".')
else:
    print('No invalid geometries found.')

######################################################################################################################################################################################################
# 311 rodent complaints data

complaints_df = pd.read_csv(DATA_PATH + r"\311_rodent_complaints.csv")
# turn into geodataframe
complaints_df.insert(0, "geometry", complaints_df['st_astext'].apply(wkt.loads))
complaints_df = gpd.GeoDataFrame(complaints_df, geometry = 'geometry', crs = crs_4326)
complaints_df.drop(columns=['geom_4326', 'st_astext'], inplace = True) #drop old geom columns

# subset to Summer 2024 complaints
complaints_df = complaints_df[complaints_df.open_dt >= '2024-06-01'] # since June 1st, 2024
complaints_df.reset_index(drop=True, inplace=True)

# check geometries
invalid_idx = get_invalid_idx(complaints_df,features_df)

if len(invalid_idx) != 0:
    print(f'Complaints df has {len(invalid_idx)} invalid geometries. Drop these rows? (yes/no)')
    user_input = input().strip().lower()

    if user_input == 'yes':
        # Drop rows with invalid indices
        complaints_df = complaints_df.drop(index=invalid_idx)
        print(f'Rows with invalid indices have been dropped. Remaining rows: {len(complaints_df)}')
    elif user_input == 'no':
        print('No rows were dropped.')
    else:
        print('Invalid input. Please enter "yes" or "no".')
else:
    print('No invalid geometries found.')

######################################################################################################################################################################################################
# Survey123 Labelled Data

survey123_df = pd.read_csv(DATA_PATH + r"\survey_123.csv")
# turn into geodataframe
survey123_df = gpd.GeoDataFrame(survey123_df, geometry=gpd.points_from_xy(survey123_df.x, survey123_df.y), crs= crs_4326)

# subset to Summer 2024 proactive inspections
survey123_df['Current Date (MM/DD/YYYY)'] = pd.to_datetime(survey123_df['Current Date (MM/DD/YYYY)'])
survey123_df['Current Date (MM/DD/YYYY)'] = survey123_df['Current Date (MM/DD/YYYY)'].dt.strftime('%Y-%m-%d')
survey123_df = survey123_df[survey123_df['Current Date (MM/DD/YYYY)'] >= '2024-06-01'] # since June 1st, 2024

# exclude Brendan's sampling rows
condition = (survey123_df['Inspector\'s License Number'] == 136869.) & (survey123_df['CreationDate'] >= '2024-06-27')
survey123_df = survey123_df.loc[~condition].reset_index(drop=True)
idx = survey123_df[survey123_df['Comments'] == 'No rodent activity found. Property is well maintained. MIT Survey.'].index
survey123_df = survey123_df.drop(idx).reset_index(drop = True)

# subset to positive survey123 inspections
conditions = (survey123_df['Type of Baiting'].isna()) & (survey123_df['General Baiting'].isna()) & (survey123_df['Bait Added (number)'].isna()) & (survey123_df['Total Bait Left (number)'].isna())
survey123_df = survey123_df.loc[~conditions]

# check geometries
invalid_idx = get_invalid_idx(survey123_df,features_df)

if len(invalid_idx) != 0:
    print(f'Survey123 df has {len(invalid_idx)} invalid geometries. Drop these rows? (yes/no)')
    user_input = input().strip().lower()

    if user_input == 'yes':
        # Drop rows with invalid indices
        survey123_df = survey123_df.drop(index=invalid_idx)
        print(f'Rows with invalid indices have been dropped. Remaining rows: {len(survey123_df)}')
    elif user_input == 'no':
        print('No rows were dropped.')
    else:
        print('Invalid input. Please enter "yes" or "no".')
else:
    print('No invalid geometries found.')


######################################################################################################################################################################################################
# Sampling data

sampling_df = pd.read_csv(DATA_PATH + r"\sampling_data.csv")
# turn into geodataframe
sampling_df['geometry'] = sampling_df.apply(lambda row: Point(row['x'], row['y']), axis=1)
sampling_df = gpd.GeoDataFrame(sampling_df, geometry='geometry', crs = crs_4326)

# check geometries
invalid_idx = get_invalid_idx(sampling_df,features_df)

if len(invalid_idx) != 0:
    print(f'Sampling df has {len(invalid_idx)} invalid geometries. Drop these rows? (yes/no)')
    user_input = input().strip().lower()

    if user_input == 'yes':
        # Drop rows with invalid indices
        sampling_df = sampling_df.drop(index=invalid_idx)
        print(f'Rows with invalid indices have been dropped. Remaining rows: {len(sampling_df)}')
    elif user_input == 'no':
        print('No rows were dropped.')
    else:
        print('Invalid input. Please enter "yes" or "no".')
else:
    print('No invalid geometries found.')

# Get positive_sampling by subsetting to 'Rats' and 'Pest Control in Place' (any evidence of rat activity) 
positive_sampling_df = sampling_df.loc[(sampling_df.Sampling == 'Rats') | sampling_df.Comments.str.contains('pest control', case = False, na = False)]
positive_mask = sampling_df.index.isin(positive_sampling_df.index)
positive_sampling_df.reset_index(drop = True, inplace = True)

# Get negative_sampling using the Boolean mask
negative_sampling_df = sampling_df[~positive_mask]
negative_sampling_df.reset_index(drop = True, inplace = True)

# Plot of all rat issues around Boston

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,10))

# Axis 1
features_df.geometry.plot(ax=ax1)
complaints_df.geometry.plot(ax = ax1, color = 'black', markersize = 3, label = 'complaints')
survey123_df.geometry.plot(ax = ax1, color = 'red', markersize = 3, label = 'survey123')
violations_df.geometry.plot(ax = ax1, color = 'yellow', markersize = 3, label = 'violations')
ax1.legend()

# Get the extent of cluster points for setting the plot limits
block_id = 479 # Choose block to zoom on
x_min, y_min, x_max, y_max = features_df.iloc[[block_id]].total_bounds 
buffer = 0.01  # Adjust the buffer for zooming in

# Set plot limits to zoom in around the cluster points
ax1.set_xlim(x_min - buffer, x_max + buffer)
ax1.set_ylim(y_min - buffer, y_max + buffer)

# Set Title
ax1.set_title("Zoomed Rat Activity Locations per Block")


# Axis 2
features_df.geometry.plot(ax=ax2)
complaints_df.geometry.plot(ax = ax2, color = 'black', markersize = 1,label = 'complaints')
survey123_df.geometry.plot(ax = ax2, color = 'red', markersize = 1, label = 'survey123')
violations_df.geometry.plot(ax = ax2, color = 'yellow', markersize = 1,label = 'violations')
ax2.legend()

# Set Title
ax2.set_title("Rat Activity Locations per Block")

# 1. Apply buffer to both violations_df

def apply_buffer(gdf, buffer_radius):
    #projec to meters to apply buffer in meters
    result = gdf.to_crs(crs_32619).copy()
    result['geometry'] = result['geometry'].buffer(buffer_radius)
    return result

buffer_radius = 25 # 25 meters as buffer radius, tune as needed and appropriate
violations_buffer = apply_buffer(violations_df, buffer_radius)

# 2. Project complaints_df
complaints_proj = complaints_df.to_crs(crs_32619)

# 3. Get intersecting complaints and drop from complaints_df
all_violations_intersections = [violations_buffer.intersects(row.geometry) for i, row in complaints_proj.iterrows()] # get boolean of all intersectiosn between complaints_df and violations_df

violations_intersections = [1 if any(sublist) else 0 for sublist in all_violations_intersections] # get 1 if complaints location intersects with any violation buffer, 0 else

violations_idx = [index for index, value in enumerate(violations_intersections) if value == 1] # extract indices of intersecting complaint locations
violations_idx = list(map(int, violations_idx))

complaints_df.drop(violations_idx, inplace = True)

# Visualize survey123 

fig, ax = plt.subplots(figsize = (8,8))

features_df.geometry.plot(ax = ax)
survey123_df.geometry.plot(ax=ax, color = 'red', markersize = 2)


# Set Title
ax.set_title("Distribution of Proactive Inspections")

# Define function to assign a block to each location
def get_intersection_indices(df1, df2):
    '''
    df1: df to iterate over
    df2: df to get indices from 
    e.g. get indices of blocks for each address --> df1: address, df2: blocks
   '''
    all_intersections = [df2.intersection(row.geometry) for i, row in df1.iterrows()]
    single_intersection = [i[~i.is_empty] for i in all_intersections]
    idx = np.array([i.index[0] for i in single_intersection])

    return idx

# Define function to undersample from each block with excessive number of proactive inspections
def remove_outliers(df):

    # Assign block to each inspection location
    idx = get_intersection_indices(df, features_df)
    df['idx'] = idx
    df_block = df.groupby('idx').size().reset_index(name = 'count') #get inspection count by block
    df_block.sort_values(by = ['count'], ascending = False, inplace = True) # sort in descending order

    # Boxplot of initial distribution of positive count per block
    plt.figure(figsize=(5, 5))
    plt.boxplot(df_block['count'])
    plt.title('Boxplot of Survey Point Counts per Block' )
    plt.ylabel('Count of Survey Points')
    plt.show()

    # Extract quartiles
    Q1 = df_block['count'].quantile(0.25)
    Q3 = df_block['count'].quantile(0.75)
    IQR = Q3 - Q1 # get interquartile range

    # Define outlier thresholds
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    outliers = df_block[df_block['count'] > upper_threshold]
    outlier_indices = outliers['idx'].tolist() # indices of blocks with too many points

    for idx in outlier_indices: 
        indices = list(df[df['idx'] == idx].index)
        count_to_drop = int(len(indices) - upper_threshold) # define number of locations within each block to drop
        if count_to_drop > 0:  # if count of locations higher than upper threshold
            points_to_drop = random.sample(indices, count_to_drop) #randomly undersample
        else: 
            points_to_drop = [] # else no undersampling
        df.drop(points_to_drop, inplace = True) # drop selected locations for undersampling
        df.reset_index(drop = True, inplace = True)


    df_block = df.groupby('idx').size().reset_index(name = 'count') # regroup by index and compute new count of locations per block
    df_block.sort_values(by = ['count'], ascending = False, inplace = True)

    # Boxplot of new distribution of positive count per block
    plt.figure(figsize=(5, 5))
    plt.boxplot(df_block['count'])
    plt.title('New Boxplot of Survey Point Counts (without Outliers)')
    plt.ylabel('Count of Survey Points')
    plt.show()

    return df

survey123_df_copy = survey123_df.copy() # create a copy of original inspections df to plot for comparison
survey123_df = remove_outliers(survey123_df) # winsorize survey123_df

# Visualize difference in distributions

# Visualize survey123 in time frames

fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (15,10))

features_df.geometry.plot(ax = ax1)
survey123_df_copy.geometry.plot(ax=ax1, color = 'red', markersize = 2)
ax1.set_title("Original Survey123")


features_df.geometry.plot(ax = ax2)
survey123_df.geometry.plot(ax=ax2, color = 'red', markersize = 2)
ax2.set_title("Survey123 without Outliers")

'''
complaint_df = without overlapping data, all positives
violations_df = all positives
positive_survey_df = winsorized, all positives
negative_survey_df = winsorized, all negatives
positive_sampling_df = all positives
negative_sampling _df = all negatives
'''
pd.options.mode.chained_assignment = None # suppress warning on slicing

###############################################################################################
# Add labels and uniform formatting

complaints_df['label'] = 1
violations_df['label'] = 1
survey123_df['label'] = 1
positive_sampling_df['label'] = 1
negative_sampling_df['label'] = 0

# Format: keep only label and geometry for each df
final_locations = pd.concat([complaints_df[['label', 'geometry']], 
                             violations_df[['label', 'geometry']], 
                             survey123_df[['label', 'geometry']], 
                             positive_sampling_df[['label', 'geometry']], 
                             negative_sampling_df[['label', 'geometry']]], axis = 0)
final_locations.head()

###############################################################################################
# Oversample class 0 to have balanced classes

majority_class = final_locations[final_locations['label'] == 1]
minority_class = final_locations[final_locations['label'] == 0]
size = int(majority_class.shape[0])

minority_class_oversampled = resample(
    minority_class,
    replace=True,     # Sample with replacement
    n_samples=size,  # Match the size of the majority class
    random_state=42   
)

final_locations = pd.concat([majority_class, minority_class_oversampled])

# Shuffle
final_locations = final_locations.sample(frac=1, random_state=42).reset_index(drop=True)


###############################################################################################
# Assign features to each point

final_df = gpd.sjoin(final_locations, features_df, how='left', predicate ='intersects').drop(columns = 'index_right')
final_df.head()

# Visualize positive and negative label distribution

# Group by block and count classes (1 or 0)
count_positives = final_df[final_df.label == 1].groupby('bid').size().reset_index(name = 'count_pos')
count_negatives = final_df[final_df.label == 0].groupby('bid').size().reset_index(name = 'count_neg')
count_labels = count_positives.merge(count_negatives, on = 'bid', how = 'outer').fillna(0)
count_labels = count_labels.merge(features_df[['geometry', 'bid']], on = 'bid', how = 'left').drop(columns = 'bid')
count_labels = gpd.GeoDataFrame(count_labels, geometry = 'geometry', crs = crs_4326)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,15))

# Axis 1
count_labels.plot(column='count_pos', cmap='Reds', linewidth=0.5, ax=ax1, edgecolor='0.8', legend=True, legend_kwds={'shrink': 0.3}, label = 'count positives')
ax1.set_title("Distribution of Positive Labels")
# Axis 2
count_labels.plot(column='count_neg', cmap='Reds', linewidth=0.5, ax=ax2, edgecolor='0.8', legend=True, legend_kwds={'shrink': 0.3}, label = 'count negatives')
ax2.set_title("Distribution of Negative Labels")

# Split into features and label
X = final_df.iloc[:, 2:-1]
y = final_df.label.astype(int)

X.drop(columns = ['n_waste_bins', 'waste_dist', 'access_points_per_m2'], inplace = True) # these features are dropped as missing data in them interferes with tree splits

# Initialize the model
cart_model = DecisionTreeClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [3, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'min_weight_fraction_leaf': [0.0, 0.001, 0.005, 0.01, 0.05],
    'random_state': [42]
}

# Initialize K-Fold splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV with K-Fold
grid_search = GridSearchCV(
    estimator=cart_model,
    param_grid=param_grid,
    cv=kf,
    scoring='accuracy'  # scoring metric
    # n_jobs=-1  # Use all available cores
)

# Perform GridSearchCV
grid_search.fit(X, y)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best parameters found: ", best_params)

threshold = 0.5 # change this to tune precision-recall tradeoff

# Predict on train set
final_y_pred_proba_train = best_model.predict_proba(X)[:, 1]
final_y_pred_train = (final_y_pred_proba_train >= threshold).astype(int)

final_accuracy_train = accuracy_score(y, final_y_pred_train)
print(f'Final Train Set Accuracy: {final_accuracy_train}')

print(classification_report(y, final_y_pred_train))

# Visualize classification tree

# print order in which classes are displayed within tree
print(best_model.classes_)

# plot tree
plt.figure(figsize=(25, 10))
plot_tree(best_model, filled=True, feature_names=X.columns, rounded=True, fontsize = 8)
plt.title("Decision Tree Classification")
plt.show()

# Plot final confusion matrix
final_cm = confusion_matrix(y, final_y_pred_train)
final_disp = ConfusionMatrixDisplay(confusion_matrix=final_cm)
final_disp.plot(cmap=plt.cm.Blues)
plt.title('Final Confusion Matrix')
plt.show()

# Plot final ROC curve and AUC
final_fpr, final_tpr, _ = roc_curve(y, final_y_pred_proba_train)
final_auc_score = roc_auc_score(y, final_y_pred_proba_train)
plt.figure()
plt.plot(final_fpr, final_tpr, color='blue', label=f'ROC curve (area = {final_auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Extract samples per split throughout the tree

# 1. Get tree attributes
tree = best_model.tree_
n_nodes = tree.node_count
children_left = tree.children_left
children_right = tree.children_right
features = tree.feature
thresholds = tree.threshold

# 2. Initialize node-to-samples mapping
node_samples = {node: ([], []) for node in range(n_nodes)}
X_array = X.values if isinstance(X, pd.DataFrame) else X # convert X to Numpy Array

# Function to map samples to nodes and store left/right sample indices
def map_samples_to_nodes(node, samples):
    if not samples:
        return
    if children_left[node] == children_right[node]:  # Leaf node
        return
    else:
        feature_index = features[node]
        if feature_index != -2:  # If not Leaf node
            threshold = thresholds[node]
            left_samples = [i for i in samples if X_array[i, feature_index] <= threshold]
            right_samples = [i for i in samples if X_array[i, feature_index] > threshold]
            # Store left and right sample indices in the dictionary
            node_samples[node] = (left_samples, right_samples)
            # Recursively map samples to left and right nodes
            if left_samples:
                map_samples_to_nodes(children_left[node], left_samples)
            if right_samples:
                map_samples_to_nodes(children_right[node], right_samples)

# Start mapping from the root node
map_samples_to_nodes(0, list(range(X_array.shape[0])))

'''
node_samples now maps each node to a tuple of left and right split sample indices
'''

# get unique Block ID in node 1 and 14 (after split at 0)

# Get indices of samples in nodes 1 and 14
x_left = node_samples[0][0] # 0 = left
x_right = node_samples[0][1] # 1 = right

# Get corresponding unique Block IDs in final_df
print('Unique blocks in node 1: ', len(final_df.iloc[x_left].bid.unique()))
print('Unique blocks in node 14: ', len(final_df.iloc[x_right].bid.unique()))