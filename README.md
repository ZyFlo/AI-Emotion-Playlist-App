# AI-Emotion-Playlist-App

Project Overview
This project is an AI-driven music playlist application that creates personalized song recommendations based on users' emotional preferences. The app uses a dataset in which users rated songs on the arousal-valence scale. These ratings help determine the emotional content of the music, allowing the app to suggest songs that align with the user's current mood or desired emotional state.

Data Utilization
The data for this project was extracted from the Moodo dataset (http://mood.musiclab.si/index.php/en/dataset) where users rated 10 random songs on the arousal-valence scale, which measures emotional responses. This scale includes two dimensions:

Valence: Reflects the positivity or negativity of an emotion.
Arousal: Indicates the energy level or intensity of the emotion.
Key Features
Data Parsing and Processing:

XML data is parsed to extract user IDs, song IDs, and corresponding valence and arousal ratings.
The data is aggregated to calculate the mean and variance of valence and arousal scores for each song.
Emotion-Based Song Recommendations:

Users can input their desired emotional state (e.g., Happy, Sad, Relaxed, Angry, Neutral).
The app uses k-Nearest Neighbors (k-NN) to find users with similar music tastes and recommends songs accordingly.
Personalized Playlists:

Users can generate playlists tailored to specific emotions, created from the collective ratings and preferences of users with similar tastes.
Interactive User Feedback:

Users provide feedback on recommended songs, which updates the ranking of songs and refines future recommendations.
User-Driven Adaptation:

The app dynamically adjusts song recommendations based on user feedback, enhancing the accuracy of emotion-based playlist creation.
