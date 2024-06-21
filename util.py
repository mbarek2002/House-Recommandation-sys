import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tfidf = TfidfVectorizer(stop_words='english')
data = pd.read_csv('Hotel_Reviews.csv')

def impute(column):
    column = column[0]
    if(type(column) != list):
        return "".join(literal_eval(column))
    else :
        return column
def preProcess():
    data.Hotel_Address = data.Hotel_Address.str.replace('Netherlands', 'NL')
    data.Hotel_Address = data.Hotel_Address.str.replace('United Kingdom', 'UK')
    data.Hotel_Address = data.Hotel_Address.str.replace('France', 'FR')
    data.Hotel_Address = data.Hotel_Address.str.replace('Spain', 'ES')
    data.Hotel_Address = data.Hotel_Address.str.replace('Italy', 'IT')
    data.Hotel_Address = data.Hotel_Address.str.replace('Austria', 'AT')
    data['countries'] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
    print(data.countries.unique())
    data.drop(['Additional_Number_of_Scoring', 'Review_Date', 'Reviewer_Nationality', 'Negative_Review',
               'Review_Total_Negative_Word_Counts',
               'Total_Number_of_Reviews', 'Positive_Review', 'Review_Total_Positive_Word_Counts',
               'Total_Number_of_Reviews_Reviewer_Has_Given',
               'Reviewer_Score', 'days_since_review', 'lat', 'lng'
               ], axis=1, inplace=True)
    print(data.columns)
    data['Tags'] = data[['Tags']].apply(impute, axis=1)
    print(data.Tags)
    data['Tags'] = data['Tags'].str.lower()
    data['countries'] = data['countries'].str.lower()


def recommender1(location, description):
    country = data[data['countries'] == location.lower()] if location != '' else data
    country = country.set_index(np.arange(country.shape[0]))

    tfidf_matrix = tfidf.fit_transform(country['Tags'])
    desc = tfidf.transform([description])

    cosine_sim = linear_kernel(desc, tfidf_matrix)
    print(len(cosine_sim[0]))
    sim = []

    sim_scores = enumerate(cosine_sim[0])

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]

    for index, score in sim_scores:
        print(f"Index: {index}, Score: {score}")

    sim_scores = [i[0] for i in sim_scores]

    recommendations = data[['Hotel_Name', 'Average_Score', 'Hotel_Address', 'Tags']].iloc[sim_scores]

    return recommendations.to_dict('records')

if __name__ == "__main__" :
    recommender1("FR", '2-bedroom condo near Marignan ')

