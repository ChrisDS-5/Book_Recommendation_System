import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process
import numpy as np

# Load and filter data
df = pd.read_csv("Updated ratings dataset.csv")


# Filter users who rated at least 20 books
user_counts = df['user_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 20].index)]

# Filter books rated at least 20 times
book_counts = df['book_id'].value_counts()
df = df[df['book_id'].isin(book_counts[book_counts >= 20].index)]

# Now create the pivot table (safer memory-wise)
user_item_matrix = df.pivot_table(
    index='user_id',
    columns='book_id',
    values='rating'
).fillna(0)

# Train KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix)

# Suggest book name using fuzzy matching
def suggest_book_name(input_title, book_titles):
    match, _, _ = process.extractOne(input_title, book_titles)
    return match

# Recommend books using KNN
def recommend_books_knn(user_id, user_item_matrix, df, n_neighbors=5, n_recommendations=10):
    if user_id not in user_item_matrix.index:
        return []

    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(user_vector, n_neighbors=n_neighbors + 1)
    similar_users = user_item_matrix.index[indices.flatten()[1:]]

    similar_users_ratings = user_item_matrix.loc[similar_users]
    mean_ratings = similar_users_ratings.mean().sort_values(ascending=False)

    user_rated_books = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    recommendations = mean_ratings[~mean_ratings.index.isin(user_rated_books)]

    book_id_to_title = df.drop_duplicates('book_id').set_index('book_id')['original_title'].to_dict()
    recommended_titles = [(book_id_to_title.get(book_id, 'Unknown Title'), round(score, 2))
                          for book_id, score in recommendations.head(n_recommendations).items()]

    return recommended_titles

# Streamlit UI
st.title("Book Recommendation System")
st.markdown("Rate at least 5 books to get your personalized recommendations!")

book_titles = df['original_title'].dropna().astype(str).unique().tolist()
book_id_map = df.drop_duplicates('original_title').set_index('original_title')['book_id'].to_dict()

user_data = []
user_ratings = {}

num_inputs = st.number_input("How many books would you like to rate? (Minimum 5)", min_value=5, max_value=50, value=5, step=1)
st.markdown("### Rate the books below:")

for i in range(num_inputs):
    col1, col2 = st.columns([3, 1])
    with col1:
        book = st.selectbox(f"Book {i+1}", options=sorted(book_titles), key=f"book_{i}")
    with col2:
        rating = st.slider(f"Rating {i+1}", min_value=1, max_value=5, value=3, key=f"rating_{i}")
    if book:
        book_id = book_id_map.get(book)
        if book_id:
            user_data.append({'book_id': book_id, 'rating': rating, 'original_title': book})

if st.button("Get Recommendations"):
    if len(user_data) < 5:
        st.warning("Please rate at least 5 books to continue.")
    else:
        new_user_id = user_item_matrix.index.max() + 1
        new_user_vector = pd.Series(data=0, index=user_item_matrix.columns, name=new_user_id)

        for row in user_data:
            new_user_vector[row['book_id']] = row['rating']

        user_item_matrix.loc[new_user_id] = new_user_vector

        new_ratings_df = pd.DataFrame([{'user_id': new_user_id, **row} for row in user_data])
        updated_df = pd.concat([df, new_ratings_df], ignore_index=True)

        recommendations = recommend_books_knn(new_user_id, user_item_matrix, updated_df)

        st.markdown("Top Book Recommendations for You:")
        for i, (title, score) in enumerate(recommendations, 1):
            st.write(f"{i}. {title} (Predicted Score: {score})")
