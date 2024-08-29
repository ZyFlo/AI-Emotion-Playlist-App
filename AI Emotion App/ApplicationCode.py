#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import numpy as np

tree = ET.parse('dataset.xml')
root = tree.getroot()

user_song_ratings = {}

for item in root.findall(".//item"):
    user_id_element = item.find('UserID')
    if user_id_element is None:
        continue
    user_id = user_id_element.text

    for song in item.findall('.//Songs/item'):
        song_id = song.find('SongID').text

        perceived_section = song.find('.//Perceived')
        v_score = float(perceived_section.find('.//V').text)
        a_score = float(perceived_section.find('.//A').text)

        if user_id not in user_song_ratings:
            user_song_ratings[user_id] = []
        user_song_ratings[user_id].append({'SongID': song_id, 'Valence': v_score, 'Arousal': a_score})

data_list = []
for user_id, song_ratings in user_song_ratings.items():
    for rating in song_ratings:
        data_list.append({'UserID': user_id, 'SongID': rating['SongID'], 'Valence': rating['Valence'], 'Arousal': rating['Arousal']})

df = pd.DataFrame(data_list)

variance_valence_mean = df.groupby(['UserID', 'SongID'])['Valence'].var().reset_index()
variance_arousal_mean = df.groupby(['UserID', 'SongID'])['Arousal'].var().reset_index()

df_mean = df.groupby(['UserID', 'SongID']).agg({'Valence': 'mean', 'Arousal': 'mean'}).reset_index()

df_mean = pd.merge(df_mean, variance_valence_mean, on=['UserID', 'SongID'], suffixes=('_Mean', '_var_valence'))
df_mean = pd.merge(df_mean, variance_arousal_mean, on=['UserID', 'SongID'], suffixes=('_Mean', '_var_arousal'))

df_mean['AvgVariance'] = (df_mean['Arousal_var_arousal'] + df_mean['Valence_var_valence']) / 2


emotion_lists = {
    'Happy': [],
    'Relaxed': [],
    'Angry': [],
    'Sad': [],
    'Neutral': []
}

test_songs = [504, 367, 115, 184]

def get_user_input(song_id):
    print(f"SongID: {song_id}")
    valence_input = float(input("Enter valence score: "))
    arousal_input = float(input("Enter arousal score: "))
    return valence_input, arousal_input

new_user_data = []

for test_song in test_songs:
    valence, arousal = get_user_input(test_song)
    new_user_data.append({
        'UserID': 'new_user',
        'SongID': str(test_song),
        'Valence': valence,
        'Arousal': arousal
    })

df_mean = pd.concat([df_mean, pd.DataFrame(new_user_data)], ignore_index=True)

imputer = SimpleImputer(strategy='mean')
df_mean[['Valence_Mean', 'Arousal_Mean']] = imputer.fit_transform(df_mean[['Valence_Mean', 'Arousal_Mean']])

combined_neighbors_data = pd.DataFrame(columns=['UserID', 'SongID', 'Valence_Mean', 'Arousal_Mean'])

for test_song in test_songs:
    valence, arousal = df_mean[(df_mean['UserID'] == 'new_user') & (df_mean['SongID'] == str(test_song))][['Valence', 'Arousal']].values[0]

    same_song_users_data = df_mean[(df_mean['SongID'] == str(test_song)) & (df_mean['UserID'] != 'new_user')][['UserID', 'Valence_Mean', 'Arousal_Mean']]

    if len(same_song_users_data) < 1:
        print(f"Not enough users who have rated the song {test_song}. Unable to provide recommendations.")
    else:
        knn_model = NearestNeighbors(n_neighbors=3)
        knn_model.fit(same_song_users_data[['Valence_Mean', 'Arousal_Mean']])

        distances, indices = knn_model.kneighbors([[valence, arousal]])

        similar_users_data = same_song_users_data.iloc[indices[0]][['UserID', 'Valence_Mean', 'Arousal_Mean']]

        similar_users_data['SongID'] = str(test_song)

        similar_users_data['Ranking'] = 0

        combined_neighbors_data = pd.concat([combined_neighbors_data, similar_users_data], ignore_index=True, sort=False)

print("\nCombined k-Nearest Neighbors for the 4 test songs:")
print(combined_neighbors_data)

loop = 1
while(loop):
    emotion_input = input("What kind of song would you like to listen to? Neutral, Happy, Sad, Relaxed, or Angry: ")
    if emotion_input == "Happy":
        target_V_input = .5
        target_A_input = .5
        loop = 0
    elif emotion_input == "Sad":
        target_V_input = -.5
        target_A_input = -.5
        loop = 0
    elif emotion_input == "Neutral":
        target_V_input = 0.0
        target_A_input = 0.0
        loop = 0
    elif emotion_input == "Relaxed":
        target_V_input = .5
        target_A_input = -.5
        loop = 0
    elif emotion_input == "Angry":
        target_V_input = -.5
        target_A_input = .5
        loop = 0
    else:
        print("Invalid input please choose a valid one.")
        
for test_song in test_songs:
    df_mean = df_mean[df_mean['SongID'] != test_song].reset_index(drop=True)

unique_user_ids = combined_neighbors_data['UserID'].unique()

all_rated_songs_data = pd.DataFrame(columns=['SongID', 'Valence_Mean', 'Arousal_Mean', 'UserID', 'Ranking'])

for user_id in unique_user_ids:
    user_rated_songs_data = df_mean[df_mean['UserID'] == user_id][['SongID', 'Valence_Mean', 'Arousal_Mean', 'UserID']]
    user_ranking = combined_neighbors_data[combined_neighbors_data['UserID'] == user_id][['SongID', 'Ranking']]
    user_rated_songs_data = pd.merge(user_rated_songs_data, user_ranking, on='SongID', how='left')
    all_rated_songs_data = pd.concat([all_rated_songs_data, user_rated_songs_data], ignore_index=True)
print(all_rated_songs_data)
#print(combined_neighbors_songs_data)
X_songs = all_rated_songs_data[['Valence_Mean', 'Arousal_Mean']].values
print(X_songs)

if len(X_songs) < 1:
    print("Not enough songs rated by the similar users. Unable to provide recommendations.")
else:
    knn_model_songs = NearestNeighbors(n_neighbors=5)  
    knn_model_songs.fit(X_songs)

    distances_songs, indices_songs = knn_model_songs.kneighbors([[target_V_input, target_A_input]])

    recommended_songs_with_users = all_rated_songs_data.iloc[indices_songs[0]][['SongID', 'UserID', 'Valence_Mean', 'Arousal_Mean', 'Ranking']]

    print("\nRecommended Songs based on Similar User Ratings:")
    print(recommended_songs_with_users)

    recommended_song_ids = recommended_songs_with_users['SongID'].unique()
    print("\nAll Recommended Song IDs:")
    print(recommended_song_ids)

print("Did you listen to them all? How did it go?")
print("Please let me know what you thought of the songs.")

ratings = [str(input(f"Did {song} suit your liking?")) for song in recommended_song_ids]

for i, (song_id, rating) in enumerate(zip(recommended_song_ids, ratings), 1):
    matching_row = recommended_songs_with_users[recommended_songs_with_users['SongID'] == song_id]
    
    user_id = matching_row['UserID'].values[0]
    row_index = recommended_songs_with_users[(recommended_songs_with_users['SongID'] == song_id) & (recommended_songs_with_users['UserID'] == user_id)].index
    row2_index = combined_neighbors_data[(combined_neighbors_data['SongID'] == song_id) & (combined_neighbors_data['UserID'] == user_id)].index

    if not row_index.empty:
        if rating.lower() == "yes":
            
            recommended_songs_with_users.loc[row_index, 'Ranking'] += 1
            combined_neighbors_data.loc[row2_index, 'Ranking'] += 1

            if song_id in emotion_lists[emotion_input]:
                continue
            else:
                emotion_lists[emotion_input].append(song_id)
            
        else:
            recommended_songs_with_users.loc[row_index, 'Ranking'] -= 1
            combined_neighbors_data.loc[row2_index, 'Ranking'] -= 1

        if recommended_songs_with_users.loc[row_index, 'Ranking'].values[0] <= -3:
            combined_neighbors_data = combined_neighbors_data[combined_neighbors_data['UserID'] != user_id]

print("\nUpdated Recommendations with User Ratings:")
print(recommended_songs_with_users)
print("\n")
print(combined_neighbors_data)

def generatePlaylist(emo):
    if len(emotion_lists[emo]) != 0:
        print("\n")
        print(f"Here are is you {emo} playlist, personally customized to your taste!")
        print(emotion_lists[emo])
        print("\n")
        return
    else:
        print(f"Oops sorry looks like we haven't gathered enough data to generate a {emo} playlist for you.")
    
def recommendSong(emo, group):
    loop = 1
    while loop:
        if emo == "Happy":
            target_V_input, target_A_input = 0.5, 0.5
            loop = 0
        elif emo == "Sad":
            target_V_input, target_A_input = -0.5, -0.5
            loop = 0
        elif emo == "Neutral":
            target_V_input, target_A_input = 0.0, 0.0
            loop = 0
        elif emo == "Relaxed":
            target_V_input, target_A_input = 0.5, -0.5
            loop = 0
        elif emo == "Angry":
            target_V_input, target_A_input = -0.5, 0.5
            loop = 0
        else:
            print("Invalid input. Please choose a valid one.")
            recommendSong(emo, group)
    
    unique_user_ids = group['UserID'].unique()
    all_rated_songs_data = pd.DataFrame(columns=['SongID', 'Valence_Mean', 'Arousal_Mean', 'UserID', 'Ranking'])
    for user_id in unique_user_ids:
        user_rated_songs_data = df_mean[df_mean['UserID'] == user_id][['SongID', 'Valence_Mean', 'Arousal_Mean', 'UserID']]
        user_ranking = group[group['UserID'] == user_id][['SongID', 'Ranking']]
        user_rated_songs_data = pd.merge(user_rated_songs_data, user_ranking, on='SongID', how='left')
        all_rated_songs_data = pd.concat([all_rated_songs_data, user_rated_songs_data], ignore_index=True)
    print(all_rated_songs_data)
    
    X_songs = all_rated_songs_data[['Valence_Mean', 'Arousal_Mean']].values

    if len(X_songs) < 1:
        print("Not enough songs rated by similar users. Unable to provide recommendations.")
    else:
        knn_model_songs = NearestNeighbors(n_neighbors=5)
        knn_model_songs.fit(X_songs)

        distances_songs, indices_songs = knn_model_songs.kneighbors([[target_V_input, target_A_input]])

        recommended_songs_with_users = all_rated_songs_data.iloc[indices_songs[0]][['SongID', 'UserID', 'Valence_Mean', 'Arousal_Mean', 'Ranking']]
        filtered_recommendations = recommended_songs_with_users[recommended_songs_with_users['Ranking'] >= -3]
        filtered_song_ids = filtered_recommendations['SongID'].unique()

        print("\nRecommended Songs based on Similar User Ratings:")
        print(recommended_songs_with_users)

        recommended_song_ids = recommended_songs_with_users['SongID'].unique()
        print("\nAll Recommended Song IDs:")
        print(recommended_song_ids)
        print("\n\n")
        print("Did you listen to them all? How did it go?")
        print("Please let me know what you thought of the songs.")

        ratings = [str(input(f"Did {song} suit your liking?")) for song in recommended_song_ids]

        for i, (song_id, rating) in enumerate(zip(filtered_song_ids, ratings), 1):
            matching_row = filtered_recommendations[filtered_recommendations['SongID'] == song_id]
    
            user_id = matching_row['UserID'].values[0]
            row_index = filtered_recommendations[(filtered_recommendations['SongID'] == song_id) & (filtered_recommendations['UserID'] == user_id)].index
            row2_index = group[(group['SongID'] == song_id) & (group['UserID'] == user_id)].index

            if not row_index.empty:
                if rating.lower() == "yes":
            
                    recommended_songs_with_users.loc[row_index, 'Ranking'] += 1
                    group.loc[row2_index, 'Ranking'] += 1

                    if song_id in emotion_lists[emo]:
                        continue
                    else:
                        emotion_lists[emo].append(song_id)
            
                else:
                    recommended_songs_with_users.loc[row_index, 'Ranking'] -= 1
                    group.loc[row2_index, 'Ranking'] -= 1

            if recommended_songs_with_users.loc[row_index, 'Ranking'].values[0] <= -3:
                group = group[group['UserID'] != user_id]
                recommended_songs_with_users = recommended_songs_with_users[recommended_songs_with_users['UserID'] != user_id]
    
    combined_neighbors_data = group
    
    print("\nUpdated Recommendations with User Ratings:")
    print(recommended_songs_with_users)
    print("\n")
    print(combined_neighbors_data)
    return
    
active = True
while(active):
    print("What would you like to do?")
    choice = int(input("1: Recommend another song?\n2: Generate a playlist?\n3. Quit.\n"))
    if(choice == 1):
        emotion = str(input("Which kind of song would you like?\n Happy, Sad, Relaxed, Angry, or Neutral"))
        recommendSong(emotion, combined_neighbors_data)
    if(choice == 2):
        emotion = str(input("Which kind of playlist would you like?\n Happy, Sad, Relaxed, Angry, or Neutral"))
        generatePlaylist(emotion)
    if(choice == 3):
        print("Thank you for using our service we look forward to seeing again")
        active = False
    
        
        


# In[ ]:




