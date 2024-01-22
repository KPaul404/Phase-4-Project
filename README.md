# Phase-4-Project

## **Film Recommendation System**

### Introduction:
Welcome to the Film Recommender System, a solution designed to address the contemporary challenge faced by movie enthusiasts in navigating the vast landscape of the film industry. With an abundance of genres, directors, actors, and production styles available, the process of choosing a movie has become both thrilling and daunting for consumers.

### Business Problem: 
The modern film enthusiast encounters a paradox of choice – an abundance of cinematic options, yet a struggle to find films aligning with their preferences. The challenge extends beyond the initial selection, encompassing the quest for related movies within the same niche or genre. Users often find themselves lost in the vast sea of content, seeking a solution that not only recommends the first movie but also facilitates a fluid journey through related titles.

### Our Solution: 
The Film Recommender System aims to seamlessly connect viewers with movies that resonate with their tastes, streamlining the process of discovering subsequent films within the same niche. By delving into the nuances of user preferences, the system provides personalized recommendations, making the movie-watching experience more enjoyable and efficient.
### Objectives:
Enhance user satisfaction and engagement by delivering highly personalized and relevant movie recommendations
Improve customer retention by continuously tailoring suggestions based on changing user preferences
Increase active usage and interactions with the platform through accurate recommendations

### Differentiation Strategy:
Focus on implementing a hybrid recommendation system that combines collaborative filtering and content-based filtering to leverage both user-item interactions as well as movie content features
Fine-tune the recommendation models to account for unique characteristics of movie preferences and viewing behaviors
Prioritize transparency and user control by allowing customization of recommendation filters

### Impact:
Positive impact on key metrics including number of active users, time spent on platform, recommendation accuracy
Enhanced user experience leading to improved customer satisfaction and retention
Increased adoption and continuous usage of platform due to high-quality recommendations

### Scope:
Utilize MovieLens latest dataset containing movie information, user ratings and tags
Implement collaborative filtering, content-based models and a hybrid approach
Focus on personalized suggestions tailored to each user's tastes and preferences

### Success Criteria:
Increased user engagement measured by interactions and time spent
Higher perceived relevance and satisfaction scores from user surveys
Growth in registered and active user base attributable to recommendations
Quantifiable improvements in recommendation accuracy metrics

#### Data Understanding:

*Primary Data Source:*  

Our dataset, labeled *'ml-latest-small'*, is a comprehensive collection encompassing 100,836 ratings and 3,683 tag applications spread across 9,742 distinct movies. This rich dataset is the culmination of contributions from 610 individual users, spanning a period from March 29, 1996, to September 24, 2018. 

This project leverages data aggregated into four distinct, interconnected files:

### 1. Movies Dataset ('movies.csv'):

This file serves as a repository of fundamental movie details, offering insights into each movie's title and genre classification.   

*Key Columns:*

- *movieId:* This serves as the primary key, uniquely identifying each movie in the dataset.   

- *title:* Provides the movie’s title, inclusive of the release year bracketed within parentheses, offering a quick reference to the movie's era.

- *genres:* A list of genres associated with the movie, presented in a pipe-separated format, such as *'Action|Adventure|Comedy'*, aiding in categorical analysis.   

### 2. Links Dataset ('links.csv'):

Designed to facilitate connections with external, established movie databases like *IMDb* and *TMDb*, this file is crucial for expanding our understanding beyond the dataset.  

*Key Columns:*   

- *movieId:* The unique movie identifier, ensuring consistency across all files within the dataset.  
  
- *imdbId:* The specific identifier used by *IMDb (Internet Movie Database)* to reference the movie.  

- *tmdbId:* The identifier utilized by *TMDb (The Movie Database)* for the movie.  

### 3. Ratings Dataset ('ratings.csv'):

Central to understanding user preferences, this file encapsulates individual user ratings for movies, presented on a nuanced 5-star scale.  

*Key Columns:*  

- *userId:* The identifier representing each unique user within the dataset.  

- *movieId:* The unique identifier associated with each movie.  

- *rating:* Reflects the user's rating of the movie, scaled from 0.5 to 5.0 in half-star increments, providing a granular view of user preferences.  

- *timestamp:* Marks the exact moment the rating was logged, recorded as seconds elapsed since the midnight of January 1, 1970, UTC.  

### 4. Tags Dataset ('tags.csv'):

This file is a collection of user-generated tags that offer deeper insights into movies through personal interpretations and descriptors.  

*Key Columns:*   

- *userId:* The unique identifier for each participating user.  

- *movieId:* The consistent unique identifier for each movie across the dataset.  

- *tag:* User-generated tags that provide descriptive insights about a movie, typically concise phrases or single words.   

- *timestamp:* Indicates when the tag was applied, recorded in seconds since the midnight of January 1, 1970, UTC.  

Each of these files plays a pivotal role in creating a *holistic view* of the movie landscape, facilitating a more nuanced and personalized movie recommendation system.

#### MODELLING
Modelling involves the following steps:
1. Feature engineering
2. Splitting data
3. Buildng and fitting the model
4. Make predictions for each user in the test set and predict the ratings for unrated movies
5. Work with an evaluation metric


##### POPULARITY BASED RECOMMENDER
Popularity based recommender systems are preferred because:
Simplicity: Easy implementation and straightforward.
No User History Required: Suitable for new or sparse datasets without relying on user-specific data.
Cold Start Problem: Addresses the challenge of limited user interaction data.
Diversity: Introduces users to popular, widely-liked items for broad appeal.
Robustness: Resilient to outliers, as recommendations are based on overall item popularity.


The Popularity-Based Recommender's RMSE of approximately 1.0425 indicates a substantial deviation, considering the rating scale of 0.5 to 5. This high RMSE highlights the need for a more effective model with improved accuracy.
Experimenting with different parameters and fine-tuning the model can enhance the recommendation system's effectiveness.
Loading surprise Library
We will use the surprise library validate our models The Surprise library is used for more advanced collaborative filtering techniques. The dataset is loaded into Surprise's format, specifying a rating scale of 1 to 5.
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

#### ADVANCED MODELLING

This segment introduces User-Based Collaborative Filtering, Singular Value Decomposition (SVD), K-Nearest Neighbors (KNN) with means, and K-Nearest Neighbors (KNN) basic (KNNBasic) models, along with their respective RMSE values.
Different KNN-based models KNNBasic, KNNWithMeans and SVD are trained and evaluated using cross-validation. The Pearson similarity metric is employed for KNN models. The performance is measured using RMSE (Root Mean Square Error).
The KNNBasic model from the surprise library is used with the option user_based set to True. It is User-Based Model: Utilizes user similarity to make recommendations.
The KNNBasic model is an Item-Based Model: Focuses on item (movie) similarities. So its user_based set to False.
Singular Value Decomposition (SVD): A matrix factorization technique that is critical for our recommendation systems.SVD employs a matrix factorization technique in a recommendation systems like ours to uncover latent features underlying the interactions between users and items (movies). It helps in capturing complex patterns in the data which might not be immediately apparent.



The best Root Mean Squared Error (RMSE) achieved is 0.8713. The optimal parameters for the Singular Value Decomposition (SVD) model are found to be 15 epochs, a learning rate of 0.01, and
a regularization term of 0.06. These parameters represent the configuration that minimizes the prediction error on the given dataset.
#### DEPLOYMENT
To bring this personalized movie recommendation system to life, we chose Streamlit for its neat web application deployment capabilities. Streamlit allowed us to take our machine learning model from Jupyter notebooks to an interactive, user-friendly web interface.
We configured Streamlit to load our preprocessed movie dataset, our trained SVD recommendation model, and The Movie Database (TMDB) metadata API. The app accepts user input, generates personalized recommendations, and displays movie posters dynamically.

![Screenshot 1](Screenshot%202024-01-19%20at%202.39.17%20PM.png)
---
![Screenshot 2](Screenshot%202024-01-19%20at%202.39.59%20PM.png)
---

#### CONCLUSIONS
The recommendation system has achieved a remarkable 86% accuracy in aligning user preferences with suggested movies, effectively addressing content navigation challenges. 

This enhancement in accuracy contributes significantly to user satisfaction and the overall success of the platform. By streamlining movie searches and maximizing content enjoyment through personalized recommendations, the system boosts efficiency and cultivates user loyalty. 

This, in turn, results in prolonged user engagement and sustained platform prosperity. Users consistently finding appealing content are more likely to remain engaged, creating a positive impact on long-term retention and the ongoing success of the platform.

#### RECOMMENDATIONS
1. Explore the development of a hybrid recommender system, combining the strengths of the SVD model and a content-based approach. This integration aims to maximize the benefits of both methods for an enhanced user recommendation system.

2. Introduce content-based recommendation features, leveraging the analysis of movie attributes like genre, actors, directors, and individual user preferences for a more varied and personalized recommendation experience.

3. Opt for showcasing films with a minimum rating of 3.5 and above, as these tend to appeal to a broad user base.

#### NEXT STEPS
1. Implement content-based recommendations analyzing attributes like genre, actors, and directors.
2. Identify and showcase films rated 3.5 and above for a diverse user appeal.
3. Develop a hybrid recommender system, combining SVD model and content-based approaches.
4. Establish a feedback loop for continuous algorithm refinement based on user preferences and engagement metrics.

---

## System Requirements
- Python 3.8 or later
- Libraries: Pandas, NumPy, Scikit-Learn, and more. (see requirements.txt for a full list)
- A modern web browser

## Installation
Clone the repository:
bash
git clone [https://github.com/KPaul404/Phase-4-Project.git]

Install dependencies:
bash
pip install -r requirements.txt


## Usage
To run the recommendation engine:
bash
python recommender.py


## Features
- Popularity-based recommender
- Content-based filtering using movie metadata
- Model-Based Collaborative Filtering (SVD)
- Interactive user interface

## Contributing
We welcome contributions! Please read CONTRIBUTING.md for guidelines on how to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

---
