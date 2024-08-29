import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px

# Parse the XML data
tree = ET.parse('dataset.xml')
root = tree.getroot()

# Dictionary to store SongID ratings
song_ratings = {}

# Iterate through each UserID
for item in root.findall(".//item"):
    user_id_element = item.find('UserID')
    if user_id_element is None:
        continue
    user_id = user_id_element.text

    user_rating_list = []

    for song in item.findall('.//Songs/item'):
        song_id = song.find('SongID').text

        perceived_section = song.find('.//Perceived')
        v_score = float(perceived_section.find('.//V').text)
        a_score = float(perceived_section.find('.//A').text)

        user_rating_list.append(song_id)
        user_rating_list.append(v_score)
        user_rating_list.append(a_score)

        if song_id not in song_ratings:
            song_ratings[song_id] = []
        song_ratings[song_id].append(f"{v_score}:{a_score}")

# Create a DataFrame
data_list = []
for song_id, ratings in song_ratings.items():
    for rating in ratings:
        v_score, a_score = map(float, rating.split(':'))
        data_list.append({'SongID': song_id, 'Valence': v_score, 'Arousal': a_score})

df = pd.DataFrame(data_list)

# Calculate variance for both mean and median
variance_valence_mean = df.groupby('SongID')['Valence'].var().reset_index()
variance_arousal_mean = df.groupby('SongID')['Arousal'].var().reset_index()

variance_valence_median = df.groupby('SongID')['Valence'].var().reset_index()
variance_arousal_median = df.groupby('SongID')['Arousal'].var().reset_index()

df_mean = df.groupby('SongID').agg({'Valence': 'mean', 'Arousal': 'mean'}).reset_index()
df_median = df.groupby('SongID').agg({'Valence': 'median', 'Arousal': 'median'}).reset_index()

df_mean = pd.merge(df_mean, variance_valence_mean, on='SongID', suffixes=('_Mean', '_var_valence'))
df_mean = pd.merge(df_mean, variance_arousal_mean, on='SongID', suffixes=('_Mean', '_var_arousal'))

df_median = pd.merge(df_median, variance_valence_median, on='SongID', suffixes=('_Median', '_var_valence'))
df_median = pd.merge(df_median, variance_arousal_median, on='SongID', suffixes=('_Median', '_var_arousal'))

df_mean['AvgVariance'] = (df_mean['Arousal_var_arousal'] + df_mean['Valence_var_valence']) / 2
df_median['AvgVariance'] = (df_median['Arousal_var_arousal'] + df_median['Valence_var_valence']) / 2

# Define the quadrant assignment function
def assign_quadrant(row):
    if row['Valence_Mean'] > 0 and row['Arousal_Mean'] > 0:
        return 'Green'
    elif row['Valence_Mean'] < 0 and row['Arousal_Mean'] > 0:
        return 'Blue'
    elif row['Valence_Mean'] < 0 and row['Arousal_Mean'] < 0:
        return 'Red'
    elif row['Valence_Mean'] > 0 and row['Arousal_Mean'] < 0:
        return 'Yellow'

df_mean['Color'] = df_mean.apply(assign_quadrant, axis=1)

def assign_quadrant_median(row):
    if row['Valence_Median'] > 0 and row['Arousal_Median'] > 0:
        return 'Green'
    elif row['Valence_Median'] < 0 and row['Arousal_Median'] > 0:
        return 'Blue'
    elif row['Valence_Median'] < 0 and row['Arousal_Median'] < 0:
        return 'Red'
    elif row['Valence_Median'] > 0 and row['Arousal_Median'] < 0:
        return 'Yellow'
    
df_median['Color'] = df_median.apply(assign_quadrant_median, axis=1)

# Create the scatter plots
fig_mean = px.scatter(df_mean, x='Valence_Mean', y='Arousal_Mean', size='AvgVariance', title='Song Ratings (Mean) (AvgVar)', color='Color', hover_name='SongID')
fig_mean.update_layout(xaxis=dict(range=[-1, 1], title='Valence'), yaxis=dict(range=[-1, 1], title='Arousal'))

fig_median = px.scatter(df_median, x='Valence_Median', y='Arousal_Median', size='AvgVariance', title='Song Ratings (Med) (AvgVar)', color='Color', hover_name='SongID')
fig_median.update_layout(xaxis=dict(range=[-1, 1], title='Valence'), yaxis=dict(range=[-1, 1], title='Arousal'))

print(df.head())
# Display the scatter plots
fig_mean.show()
fig_median.show()
