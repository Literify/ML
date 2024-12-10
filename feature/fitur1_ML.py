def predict(title, data, cos_sim, similarity_weight=0.7, top_n=10):
    index_movie = data[data['book_title'] == title].index
    similarity = cos_sim[index_movie].T

    sim_df = pd.DataFrame(similarity, columns=['similarity'])
    final_df = pd.concat([data, sim_df], axis=1)

    final_df['final_score'] = final_df['weighted_average']*(1-similarity_weight) + final_df['similarity']*similarity_weight

    final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
    final_df_sorted_show = final_df_sorted[['book_title', 'description', 'authors', 'genre', 'publisher', 'Price', 'image', 'previewLink', 'infoLink']]
    
    selected_title = final_df[final_df['book_title'] == title][['book_title', 'description', 'authors', 'genre', 'publisher', 'Price', 'image', 'previewLink', 'infoLink']].to_dict(orient='records')[0]

    recommendations = final_df_sorted_show.to_dict(orient='records')

    output = {
        "selected_title": selected_title,
        "recommendations": recommendations
    }

    return output
