# Generated from: Purchase+Analytics+Descriptive+Analysis+with+Comments+7.4.py.ipynb
# Converted at: 2026-10-18T18:49:18.428Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## ${\textbf{Libraries}}$


import numpy as np
import pandas as pd

# We import the sk learn modules we'll need to segment our new data. We'll need scaler, pca and k-means.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# We import pickle in order to be able to load our pickled objects.
import pickle

# ## ${\textbf{Data Import}}$


# Load data. We read the purchase data csv file.
df_purchase = pd.read_csv('purchase data.csv')

# ## ${\textbf{Data Exploration}}$


# We use the head method on our data and check for missing values.
df_purchase.head()

df_purchase.isnull().sum()

# ## ${\textbf{Data Segmentation}}$


# ### ${\textbf{Import Segmentation Model}}$


# We load our pickled objects in order to segment the purchase data set.
scaler = pickle.load(open('scaler.pickle', 'rb'))

pca = pickle.load(open('pca.pickle', 'rb'))

kmeans_pca = pickle.load(open('kmeans_pca.pickle', 'rb'))

# ### ${\textbf{Standardization}}$


# We standardize the purchase data in the same way we did the segmentation data, using the standard scaler.
features = df_purchase[['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
df_purchase_segm_std = scaler.transform(features)

# ### ${\textbf{PCA}}$


# We apply pca on the purchase data and obtain 3 principal components for each row in the table. 
df_purchase_segm_pca = pca.transform(df_purchase_segm_std)

# ### ${\textbf{K-means PCA}}$


# Based on the principal components, we use the predict method from pca to segment the purchase data into the four segments.
purchase_segm_kmeans_pca = kmeans_pca.predict(df_purchase_segm_pca)

# Copy resulting data frame as a purchase predictors data frame. 
# We'll be changing the predictors data frame, by adding and changing columns.
# Therefore, we want to keep an original copy.
df_purchase_predictors = df_purchase.copy()

# Add segment information as a new column in our predictors data frame.
df_purchase_predictors['Segment'] = purchase_segm_kmeans_pca