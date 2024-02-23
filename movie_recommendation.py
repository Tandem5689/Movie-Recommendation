# -*- coding: utf-8 -*-


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('movie_metadata_new.csv')
dataset_1 = pd.read_csv('movie_metadata_new.csv')
dataset_1['duration'] = dataset_1['duration'].fillna(np.mean(dataset_1['duration']))
dataset_1['aspect_ratio'] = dataset_1['aspect_ratio'].fillna(dataset_1['aspect_ratio'].mode()[0])
dataset_1 = dataset_1[dataset_1.isnull().sum(axis=1) < 7]
condition = dataset_1['title_year'] < 2000
opposite_condition = condition.apply(lambda x: not x)
dataset_1.loc[condition, 'color'] = dataset_1.loc[condition, 'color'].fillna(' Black and White')
dataset_1.loc[opposite_condition, 'color'] = dataset_1.loc[opposite_condition, 'color'].fillna('Color')
no_of_duplicates_rows = dataset_1.duplicated().sum()
dataset_1 = dataset_1.drop_duplicates(subset=None, keep='first', inplace=False)
dataset_1["actors"] = dataset_1["actor_1_name"].str.cat(dataset_1[["actor_2_name", "actor_3_name"]].astype(str), sep="|")
dataset_1 = dataset_1.drop(columns=['actor_1_name', 'actor_2_name', 'actor_3_name'])

ran_for_reg = dataset_1[['director_name', 'country', 'language', 'color', 'genres','title_year','aspect_ratio','movie_facebook_likes','cast_total_facebook_likes','num_user_for_reviews', 'num_voted_users','num_critic_for_reviews','gross', 'budget','director_facebook_likes','facenumber_in_poster', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'duration', 'imdb_score']].copy() 

# data preprocessing

ran_for_reg['director_facebook_likes'] = ran_for_reg['director_facebook_likes'].fillna(np.mean(ran_for_reg['director_facebook_likes']))
ran_for_reg['actor_1_facebook_likes'] = ran_for_reg['actor_1_facebook_likes'].fillna(np.mean(ran_for_reg['actor_1_facebook_likes']))
ran_for_reg['actor_2_facebook_likes'] = ran_for_reg['actor_2_facebook_likes'].fillna(np.mean(ran_for_reg['actor_2_facebook_likes']))
ran_for_reg['actor_3_facebook_likes'] = ran_for_reg['actor_3_facebook_likes'].fillna(np.mean(ran_for_reg['actor_3_facebook_likes']))
ran_for_reg['budget'] = ran_for_reg['budget'].fillna(np.mean(ran_for_reg['budget']))
ran_for_reg['facenumber_in_poster'] = ran_for_reg['facenumber_in_poster'].fillna(ran_for_reg['facenumber_in_poster'].median())
ran_for_reg['gross'] = ran_for_reg['gross'].fillna(np.mean(ran_for_reg['gross']))
ran_for_reg['num_critic_for_reviews'] = ran_for_reg['num_critic_for_reviews'].fillna(np.mean(ran_for_reg['num_critic_for_reviews']))
ran_for_reg['num_user_for_reviews'] = ran_for_reg['num_user_for_reviews'].fillna(np.mean(ran_for_reg['num_user_for_reviews']))
ran_for_reg['title_year'] = ran_for_reg['title_year'].fillna(ran_for_reg['title_year'].mode()[0])
ran_for_reg_genres = ran_for_reg.genres.str.get_dummies(sep = '|')
ran_for_reg = pd.concat([ran_for_reg, ran_for_reg_genres], axis=1)
ran_for_reg = ran_for_reg.drop('genres', axis = 1)
ran_for_reg = pd.get_dummies(data = ran_for_reg, columns=['color', 'language', 'country', 'director_name'])


b = ran_for_reg.isnull().sum(axis = 0)


# random forest regression
y = np.array(ran_for_reg['imdb_score'])
ran_for_reg = ran_for_reg.drop('imdb_score', axis = 1)


from sklearn.model_selection import train_test_split
train_ran_for_reg, test_ran_for_reg, train_y, test_y = train_test_split(ran_for_reg, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(train_ran_for_reg, train_y);


predictions = regressor.predict(test_ran_for_reg)
errors = abs(predictions - test_y)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate percentage error 
percentage_error = 100 * (errors / test_y)
# Calculate accuracy
accuracy = 100 - np.mean(percentage_error)
print('Accuracy:', round(accuracy, 2), '%.')



from sklearn.metrics import mean_squared_error

from math import sqrt
actual = test_y
predicted = predictions

rmse = sqrt(mean_squared_error(actual, predicted))

print(rmse)
print('the rmse is:')
print(rmse)

a= ran_for_reg.head()