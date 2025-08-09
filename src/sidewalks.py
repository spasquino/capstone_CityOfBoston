# Packages
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
import pandas as pd
import numpy as np
from collections import Counter
from shapely import wkt
import math
from tqdm import tqdm

# Data path
DATA_PATH = r"C:\Users\40009389\Documents\Rodents\data"
# DATA_PATH = r"C:\Users\40009382\Documents\Capstone_repo\data"

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

# Copy/subset data to avoid reading in every time

# Make copy of raw df; make new ID
df = df_raw.copy()
df.reset_index(names='SID', inplace=True)

# Make copy of raw dfs
df_sidewalk = df_sidewalk_raw.copy()
df_complaint = df_complaint_raw.copy()

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

# Find nearest sidewalks per missing complaint (~24 secs)
idx = []
for i, row in tqdm(df_complaint_missing.to_crs(CRS_3857).iterrows()):
    # idx.append([row.geometry.distance(df.to_crs(CRS_3857).geometry).idxmin()])  # Nearest sidewalk
    idx.append(list(row.geometry.distance(df.to_crs(CRS_3857).geometry).sort_values()[:3].index))  # 3 nearest sidewalks

# # Plotting 3 nearest sidewalks
# colors = ['orange', 'lime', 'cyan']

# fig, axs = plt.subplots(22, 2, figsize=(10,120))

# for i in tqdm(range(len(OBJECTID_missing))):
#     row = df_complaint_missing.iloc[[i]]
    
#     # Get lon and lat to zoom in on sidewalk
#     lon = row.geometry.representative_point().x.values[0]
#     lat = row.geometry.representative_point().y.values[0]
    
#     # Plot
#     # Zoom in
#     axs[i, 0].set_xlim(lon-0.0025, lon+0.0025)
#     axs[i, 1].set_xlim(lon-0.0025, lon+0.0025)
#     axs[i, 0].set_ylim(lat-0.0025, lat+0.0025)
#     axs[i, 1].set_ylim(lat-0.0025, lat+0.0025)
    
#     # All sidewalks from joined df
#     df.plot(ax=axs[i, 0], color="gainsboro")
#     df.plot(ax=axs[i, 1], color="gainsboro")
    
#     # Complaint
#     row.plot(ax=axs[i, 0], edgecolor="red", facecolor="red")
#     row.plot(ax=axs[i, 1], edgecolor="red", facecolor="red")
    
#     # 3 nearest sidewalks
#     legends = []
#     for j in range(len(idx[i])):
#         df.iloc[[idx[i][j]]].plot(ax=axs[i, 1], edgecolor=colors[j], facecolor=colors[j])
#         legends.append(Patch(color=colors[j], label=idx[i][j]))
    
#     # Title & Legend
#     axs[i, 0].set_title(f"({i}) {OBJECTID_missing[i]}: Complaint")
#     axs[i, 1].set_title(f"({i}) {OBJECTID_missing[i]}: Nearest Sidewalks")
#     axs[i, 1].legend(handles=legends)

# plt.show()

# Mapping complaints to sidewalks

# List of sidewalk idx to match complaint to
complaint_idx = list(df_complaint_missing.index)
sidewalk_idx = [None, None, None, None, 3122, 16630, 14658, None, 15689, 19513, 20554, 17659, 16633, 18840, 11143, 20537, 15514, 2421, 16490, 5373, 19513, 16630]

# List of columns to replace in df
df_columns = ['OBJECTID_1', 'SWK_ID', 'MATERIAL_1', 'SWK_WIDTH_1', 'DISTRICT', 'SWK_AREA', 'PARENT_1', 'SEG_ID', 'SIDE_1',
               'CG_ID', 'Rpr_Yr', 'New_SCI', 'Recon_Date', 'HPNETWORK', 'TEST_AREA',
               'COST_RECON', 'GOODCANDID', 'TOTAREA', 'SCI_2023_2024', 'DAM_AREA23',
               'DETER_RATE', 'UNI_COST23', 'COST_2023', 'SW_HIST23_1', 'AproxStati',
               'InspectDat', 'InspectBy', 'ConditionC', 'InspectedS', 'Shape__Are',
               'Shape__Len', 'GlobalID_1', 'CreationDa', 'Creator', 'EditDate',
               'Editor', 'MeasureSCI', 'Obstructio', 'Notes_1', 'Priority', 'HIGHWAY']
df_complaint_columns = ['OBJECTID', 'SWK_ID', 'MATERIAL', 'SWK_WIDTH', 'DISTRICT', 'SWK_AREA',
                        'PARENT', 'SEG_ID', 'SIDE', 'CG_ID', 'Rpr_Yr', 'New_SCI', 'Recon_Date',
                        'HPNETWORK', 'TEST_AREA', 'COST_RECON', 'GOODCANDID', 'TOTAREA',
                        'SCI_2023', 'DAM_AREA23', 'DETER_RATE', 'UNI_COST23', 'COST_2023',
                        'SW_HIST23', 'AproxStati', 'InspectDat', 'InspectBy', 'ConditionC',
                        'InspectedS', 'Shape__Are', 'Shape__Len', 'GlobalID', 'CreationDa',
                        'Creator', 'EditDate', 'Editor', 'MeasureSCI', 'Obstructio', 'Notes',
                        'Priority', 'HIGHWAY']


# Loop over each complaint and fill in corresponding columns in df
for i in range(len(complaint_idx)):
   i_sidewalk = sidewalk_idx[i]
   i_complaint = complaint_idx[i]
   
   # Skip if this complaint has no corresponding sidewalk
   if i_sidewalk is None:
      continue

   # If this sidewalk doesn't already have a complaint, edit the corresponding columns to have complaint information
   if pd.isna(df.loc[i_sidewalk, "OBJECTID_1"]):
      # print("sidewalk doesn't have complaint yet")
      df.loc[i_sidewalk, df_columns] = df_complaint_missing.loc[i_complaint, df_complaint_columns].values
   
   # If this sidewalk does already have a complaint, just skip it
   # Maybe in future try to duplicate the row and overwrite the new column's complaint information
   else:
      print(f"sidewalk {i_sidewalk} has complaint")
      continue
      # row_to_duplicate = df.iloc[i_sidewalk].copy()
      # df = pd.concat([df, row_to_duplicate.to_frame().transpose()], ignore_index=True)
      # df.loc[df.index[-1], df_columns] = df_complaint_missing.loc[i_complaint, df_complaint_columns].values

# Sanity check: # unique complaint geometries
print(f"number of complaint rows: {df_complaint.shape[0]}")
print(f"number of unique complaint geometries: {len(df_complaint.geometry.unique())}")
print(f"number of complaints in merged dataset: {sum(~pd.isna(df.OBJECTID_1))}")

# Merge data to tracts
# Assign each sidewalk to a tract
df = gpd.sjoin(df, df_tracts[['gid', 'geometry']], how="left", predicate="intersects")

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

df_sidewalk_gids = df.groupby('SID').apply(get_biggest_gid, include_groups=False).reset_index()
df = df.merge(df_sidewalk_gids, on=['SID', 'gid'], how='inner')

# Clean/engineer data

# Relevant to analysis

# SCI_2023
# Drop row where SCI is out of bounds
df = df.drop(df[df.SCI_2023 > 100].index)

# Priority
# Remove only ONS complaints
df.loc[:, 'Priority'] = df.Priority.replace('ONS', None)

# SCI_bin
# Sort sidewalks into severity bins (credit to Jeff Kaplan)
# Score over 80: Do nothing
# 50-80 Localized repair
# Under 50: full reconstruction
df.insert(0, "SCI_bin", ["Do Nothing" if s > 80 else "Full Reconstruction" if s < 50 else "Localized Repair" for s in df.SCI_2023])

# has_complaint
# 0 or 1 if sidewalk has a complaint
df.insert(0, "has_complaint", df.OBJECTID_1.apply(lambda x: 0 if pd.isna(x) else 1))

# # Not relevant to analysis

# # MATERIAL
# # Replace Cc with CC
# df.loc[:, 'MATERIAL'] = df.MATERIAL.str.replace('Cc', 'CC')

# # SWK_WIDTH
# # Fix non-digit characters, convert to numeric
# df.loc[:, 'SWK_WIDTH'] = pd.to_numeric(df.SWK_WIDTH.str.replace('O', '0').replace(r"[^\d.]", '', regex=True))

# # SIDE
# # Replace non-standard formats with None
# df.loc[:, 'SIDE'] = df.SIDE.replace(r"( |BR)", None, regex=True)

# # DISTRICT
# # Replace 'THE NORTH END' with 'NORTH END'
# df.loc[:, 'DISTRICT'] = df.DISTRICT.str.replace('THE NORTH END', 'NORTH END')

# # Difference between SCI columns
# # 'SCI_2023', 'New_SCI', 'SCI_2023_2024', 'MeasureSCI'

# SCIs = ['SCI_2023', 'New_SCI', 'SCI_2023_2024', 'MeasureSCI']

# # Histograms
# fig, axs = plt.subplots(1, 4, figsize=(20, 4))
# for i, col in enumerate(SCIs):
#     axs[i].hist(df[col])
#     axs[i].set_title(col)
#     axs[i].set_xlabel(col)
# axs[0].set_ylabel('Count')
# plt.show()


# # Scatterplots
# fig, axs = plt.subplots(4, 4, figsize=(10, 10))
# for i, col_i in enumerate(SCIs):
#     for j, col_j in enumerate(SCIs):
#         axs[i,j].scatter(df[col_j], df[col_i], s=2)
#         axs[i,0].set_ylabel(col_i)
#         axs[0,j].set_title(col_j)
# plt.show()

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
# Change font sizes
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)

# True sidewalk status
colors = ["red" if s=="Full Reconstruction" else "yellow" if s=="Localized Repair" else "green" for s in df.SCI_bin]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
df.plot(ax=ax1, edgecolor=colors)
ax1.set_title("Sidewalk Scores")

# Complaints
df.plot(ax=ax2, color="black")
df_complaint.plot(ax=ax2, edgecolor="red")
ax2.set_title("Complaints")

# Legend
legend_red = Patch(color="red", label="Full Reconstruction")
legend_yellow = Patch(color="yellow", label="Localized Repair")
legend_green = Patch(color="green", label="Do Nothing")
legend_complaint = Patch(color="red", label="Complaints")
legend_sidewalk = Patch(color="lightgray", label="Sidewalks")

ax1.legend(handles=[legend_red, legend_yellow, legend_green], loc="upper left", prop={'size': 26})
ax2.legend(handles=[legend_complaint, legend_sidewalk], loc="upper left",prop={'size': 26})

plt.show()

plt.rcParams.update(plt.rcParamsDefault)

# Number of sidewalks/complaints per block

fig, axs = plt.subplots(1, 3, figsize=(25,5))
axs[0].hist(df.groupby('gid').size(), bins=50)
axs[0].set_title("Distribution of Number of Sidewalks per Block")
axs[0].set_xlabel("Number of Sidewalks in Block")
tmp = pd.get_dummies(df, columns=['SCI_bin'])[['gid', 'SCI_bin_Do Nothing']].groupby('gid').apply(np.sum, include_groups=False, axis=0).reset_index().rename(columns={'SCI_bin_Do Nothing':'num_bad'})
axs[1].hist(tmp.num_bad, bins=50)
axs[1].set_title("Distribution of Number of BAD Sidewalks per Block")
axs[1].set_xlabel("Number of Bad Sidewalks in Block")
tmp = df.groupby('gid').has_complaint.apply(np.sum)
axs[2].hist(tmp, bins=50)
axs[2].set_title("Distribution of Number of Complaints per Block")
axs[2].set_xlabel("Number of Complaints in Block")

print("Description of #Sidewalks per Block")
print(f"{df.shape[0]} total sidewalks")
print(df.groupby('gid').size().describe())
print("Description of #Complaints per Block")
print(f"{df[df.has_complaint==1].shape[0]} total complaints")
print(tmp.describe())

# Get sidewalks needing repair
df_sidewalk_bad = df[df.SCI_bin == "Full Reconstruction"]

# Get counts of reports and problems per tract
def combine_functions(group):
    result = {}
    result['complaints'] = np.sum(group.has_complaint)
    result['problems'] = group.shape[0]
    return pd.Series(result)
df_metrics = df_sidewalk_bad.groupby('gid').apply(combine_functions, include_groups=False).reset_index()

# Get geometry
df_metrics = df_metrics.merge(df_tracts[['gid', 'geometry']], on='gid')
df_metrics = df_metrics[['gid', 'geometry', 'complaints', 'problems']]
df_metrics = gpd.GeoDataFrame(df_metrics, geometry='geometry')

# Create metrics for underreporting
def combine_functions(group):
    result = {}
    
    # 1: Proportion of bad sidewalks that have a complaint
    result['m1'] = np.mean(group.has_complaint)
    
    # 2: 0+1/total+1 for tracts with 0 complaints
    if sum(group.has_complaint)==0:
        result['m2'] = 1/(group.shape[0]+1)
    else:
        result['m2'] = np.mean(group.has_complaint)
    
    # 3: 0+1/total+1 for all tracts
    result['m3'] = (sum(group.has_complaint)+1)/(group.shape[0]+1)
    
    return pd.Series(result)

df_metrics = df_metrics.merge(df_sidewalk_bad.groupby('gid').apply(combine_functions, include_groups=False).reset_index(), on='gid')

# Impute m for tracts that don't have m
df_metrics_imputed = df_metrics.drop(df_metrics.index)

for i, row in df_tracts[~df_tracts.gid.isin(df_metrics.gid)].iterrows():
    # Get neighbors of this tract
    this_tract_neighbors = df_tracts[df_tracts.intersects(row.geometry)]

    # Get metrics of neighbors
    this_tract_neighbors = df_metrics[['gid', 'm1', 'm2', 'm3']].merge(this_tract_neighbors[['gid']], on='gid')

    # Take average metric
    this_tract_m1 = this_tract_neighbors.m1.mean()
    this_tract_m2 = this_tract_neighbors.m2.mean()
    this_tract_m3 = this_tract_neighbors.m3.mean()
    df_metrics_imputed = pd.concat([df_metrics_imputed, 
            gpd.GeoDataFrame(pd.DataFrame({'gid': [row.gid],
                                'geometry': [row.geometry],
                                'm1': [this_tract_m1],
                                'm2': [this_tract_m2],
                                'm3': [this_tract_m3],
                                }), crs=CRS_4326, geometry='geometry')
            ],
            ignore_index=True)

df_metrics = pd.concat([df_metrics, df_metrics_imputed]).sort_values('gid').reset_index(drop=True)

# Drop tracts with literally all NaN (just harbor islands)
df_metrics = df_metrics[~pd.isna(df_metrics.m1)]

# # Plotting metrics
# # Histograms
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))
# ax1.hist(df_tract_metrics.m1, bins=50)
# ax1.set_title("Proportion of Bad Sidewalks Reported")
# ax2.hist(df_tract_metrics.m2, bins=50)
# ax2.set_title("Add 1 to Tracts w/o Complaints")
# ax3.hist(df_tract_metrics.m3, bins=50)
# ax3.set_title("Add 1 to All Tracts")
# plt.show()

# # Metric Maps
# fig, axs = plt.subplots(1, 3, figsize=(24,5))
# ms = ['m1', 'm2', 'm3']

# for i in range(len(ms)):
#     # Tracts that don't have a metric are red
#     df_tracts.plot(ax=axs[i], color='darksalmon')
    
#     # Tracts that do have a metric are green/yellow
#     df_tract_metrics.plot(ax=axs[i], column=ms[i], cmap='summer', legend=True)
    
#     # Tracts that have m=0 are gray
#     df_zero = df_tract_metrics[df_tract_metrics[ms[i]]==0]
#     if df_zero.shape[0] > 0:
#         df_zero.plot(ax=axs[i], color='darkslategray')
    
#     axs[i].set_title(ms[i])
# plt.show()

# Metric vs Problems vs Complaints
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,7))
# Problems
# df_tracts.plot(ax=ax1, color='gainsboro')
df_metrics.plot(ax=ax1, column='problems', cmap='summer', legend=True)
ax1.set_title("Problems (Lighter->More Problems)")
# Complaints
df_metrics.plot(ax=ax2, column='complaints', cmap='summer', legend=True)
ax2.set_title("Complaints (Darker->Fewer Complaints)")
# Metric
df_metrics.plot(ax=ax3, column='m1', cmap='summer', legend=True)
ax3.set_title("Metric 1 (Darker->More Pseudosampling)")
plt.show()

# What if we use these metrics as divisors? What is the difference between these metrics?

# Dropping NAs for this analysis

df_metrics_no_na = df_metrics[~pd.isna(df_metrics.complaints)]

complaints = df_metrics_no_na.complaints
problems_true = df_metrics_no_na.problems
problems_m2 = df_metrics_no_na.complaints/df_metrics_no_na.m2
problems_m3 = df_metrics_no_na.complaints/df_metrics_no_na.m3

# Plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,4))
ax1.hist(problems_true, bins=50)
ax1.set_title("True Number of Problems")
ax2.hist(complaints, bins=50)
ax2.set_title("Number of Complaints")
ax3.hist(problems_m2, bins=50)
ax3.set_title("Complaints/m2")
ax4.hist(problems_m3, bins=50)
ax4.set_title("Complaints/m3")
plt.show()

# # Diffs
# diffs = [abs(problems_m2[i] - problems_m3[i]) for i in range(len(problems_m2))]
# plt.figure(figsize=(5.5, 4))
# plt.hist(diffs, bins=50)
# plt.title("|estimate2 - estimate3|")
# plt.show()

from scipy.stats import pearsonr
print('Correlations')
print(f"true and complaints: {pearsonr(problems_true, complaints)[0]}")
print(f"true and m2: {pearsonr(problems_true, problems_m2)[0]}")
print(f"true and m3: {pearsonr(problems_true, problems_m3)[0]}")

# Compute p ~ underreporting
df_metrics.insert(df_metrics.shape[1], 'm1_inverse', df_metrics.m1.apply(lambda m: 1-m))
df_metrics.insert(df_metrics.shape[1], 'p1', df_metrics.m1_inverse.apply(lambda m: m / sum(df_metrics.m1_inverse)))
df_metrics.insert(df_metrics.shape[1], 'p1_overreporting', df_metrics.m1.apply(lambda m: m / sum(df_metrics.m1)))

# Plot m1 and p1: Should be similar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
df_tracts.plot(ax=ax1, color="gainsboro")
df_metrics.plot(ax=ax1, column="m1", cmap='summer', legend=True)
ax1.set_title("m1")
df_tracts.plot(ax=ax2, color="gainsboro")
df_metrics.plot(ax=ax2, column="p1", cmap=matplotlib.colormaps['summer'].reversed(), legend=True)
ax2.set_title("p1")
plt.show()

# How many times should we pseudosample?

# Option 1: How many problems are going unreported?
print(f"{sum(df_sidewalk_bad.has_complaint==0)} of {df_sidewalk_bad.shape[0]} bad sidewalks are not reported ({round(100*sum(df_sidewalk_bad.has_complaint==0)/df_sidewalk_bad.shape[0], 2)}%)")

# Option 2: How many reports do we have already?
print(f"{sum(df_sidewalk_bad.has_complaint)} bad sidewalks are reported ({round(100*sum(df_sidewalk_bad.has_complaint)/df_sidewalk_bad.shape[0], 2)}%)")

# Example pseudosampling
np.random.seed(19)
np.random.choice(df_metrics.gid, size=10, p=df_metrics.p1)

df_metrics.to_csv("../data/reporting_metrics.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

df_metrics.plot(ax=axs[0], column='p1', cmap='Reds', legend=True)
axs[0].set_title("P1 for Underreporting")
df_metrics.plot(ax=axs[1], column='p1_overreporting', cmap='Reds', legend=True)
axs[1].set_title("P1 for Overreporting")
plt.show()