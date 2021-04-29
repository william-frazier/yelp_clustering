# William Frazier
# CS 565 Project 2
# Spring 2021
# Boston University


import json
import pickle
import numpy as np
from statistics import mean
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer

def some_examples(city='Boston', num_reviews=100):
    """
    Here's some simple drivercode to make viewing my code a bit easier.
    """
    
    IDs = get_IDs(city, num_reviews)
    reviews = find_reviews(IDs)
    combined_reviews = combine_reviews(reviews)
    X = create_vector(combined_reviews)
    info = business_info(IDs)
    info_usable_form = []
    stars = business_stars(IDs)
    for key in info:
        info_usable_form.append(info[key])
    input("For each iteration, graphs are on PCA axes. First graph is colored by clustering algorithm while second graph is colored by lat. and long. Press enter to run k-means++.")
    evaluate_clusters(X, 10, info_usable_form)
    input("Press enter to run knn.")
    X,y = create_vector(combined_reviews, stars)
    knn(X,y)
    print("Note that the accuracy of my knn varies wildly based on which points end in the test set. It is much better at classifying highly rated restaurants as there are so few low-rated ones in the dataset.")
    input("Press enter to explore sentiment analysis.")
    sentiment_scores = sentiment(reviews)
    sentiment_scores_usable_form, y, keys = sentiment_stars(sentiment_scores, stars)
    kmpp_sentiment_analysis(4, sentiment_scores_usable_form.reshape(-1,1), y)
    input("Here we see clustering where the x-axis is our sentiment analysis and the color is the true clustering label. Press enter to see knn.")
    y_prime = []
    print("hit")
    for i in y:
        y_prime.append(int(i)) # Convert away from floats
    print("Second hit")
    knn_sentiment_analysis(sentiment_scores_usable_form.reshape(-1,1)[:-20],y_prime[:-20],sentiment_scores_usable_form.reshape(-1,1)[-20:],y_prime[-20:])




###########################
# Preprocessing Functions #
###########################

def get_IDs(city, num_reviews=200, write=False):
    """
    Get the business IDs for restaurants in given city which have more than
    num_reviews reviews. If write is set to true then their JSON data will be 
    dumped to a file.
    """
    
    print(f"Searching for businesses in {city}.", end=' ')
    businessIDs = set()
    if write:
        cities = {city: []}
    
    with open("yelp_academic_dataset_business.json", encoding='utf-8') as f:
        for line in f: # Line by line because the file is so massive
            data = json.loads(line) # Convert to python dict
            if data['city'] == city: # Look only for the city we want
                if data['review_count'] > num_reviews:
                    if data['categories']:
                        if 'Chinese' in data['categories']: # Only Chinese restaurants
                            businessIDs.add(data['business_id'])
                            if write:
                                cities[city].append(data)
            
    if write:
        with open(f'{city}.json', 'w') as f:
            json.dump(cities,f, indent=2)
    print(f"Identified {len(businessIDs)} businesses.")
    return businessIDs


def find_reviews(IDs, write=False):
    """
    Given a list of buisness IDs, this function will search through the review
    dataset and return a list of all of the reviews about those businesses. If
    write is a string, reviews will be saved to a file.
    """

    print(f"Finding reviews for the {len(IDs)} selected businesses.", end=' ')
    reviews = []
    with open("yelp_academic_dataset_review.json", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in IDs:
                reviews.append(data)
    if write:
        pickle.dump(reviews, open(f"{write}-JSON-Reviews.pkl", "wb"))
    print(f"Found {len(reviews)} reviews.")
    return reviews


def combine_reviews(reviews, write=False):
    """
    Given a list of reviews, this function will return a dictionary where the 
    key is a business ID and the value is a long string of every review for 
    that business. If write is a string, the dictionary will be saved to a file.
    """
    
    print("Combining reviews.")
    docs = {}
    for review in reviews:
        try:
            docs[review['business_id']] += review['text']
        except:
             docs[review['business_id']] = review['text']
    if write:
        pickle.dump(docs, open(f"{write}-reviews.pkl", "wb"))    
    
    return docs


def create_vector(reviews, stars=False, count=False, stop='english', write=False):
    """
    Given a dictionary where each key is a business ID and each value is a long 
    string containing the text for reviews for that business, returns the 
    TF-IDF matrix. If stars is set to the output of business_stars(), function 
    will return y along with X. The reason I do it this way is to ensure the 
    same ordering. If count is set to True, a count matrix will be returned
    instead of a TF-IDF matrix. stop can be changed to my_stop_words to use my
    custom stop words list. If write is set to a string, output will be saved
    to a file.
    """
    
    print("Creating matrix from documents.")
    tokens = [] # Set would be faster but we need to keep the ordering
    y = []
    for key in reviews:
        tokens.append(reviews[key])
        if stars:
            y.append(stars[key])
    if stars:
        le = LabelEncoder()
        y = le.fit_transform(y) # Converts floats to ints so we can use knn
    if count:
        vectorizer = CountVectorizer(stop_words=stop)
    else:
        vectorizer = TfidfVectorizer(min_df=4, 
                                 stop_words=stop)
    X = vectorizer.fit_transform(tokens)
    if write:
        pickle.dump(X, open(f"{write}-vector.pkl", "wb"))
    if stars: # Probably not the best way to do this but it's ok for class
        return X,y
    return X


def business_info(IDs):
    """
    Given a list of business IDs, returns a dictionary where keys are business
    IDs and values are the latitude and longitude of that business. Probably 
    should be combined with following function but this is easier. There was a
    version that also worked with categories but I've removed it because I 
    didn't find it particularly useful.
    """
    
    print("Finding lat. and long. for selected businesses.")
    info = {} # Dictionary allows us to maintain ordering
    with open("yelp_academic_dataset_business.json", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in IDs:
                info[data['business_id']]=[data['latitude'],data['longitude']]
    return info


def business_stars(IDs):
    """
    Given a list of business IDs, returns a dictionary where the keys are 
    business IDs and the values are the rating for that business.
    """
    
    print("Finding ratings for selected businesses.")
    stars = {} # Dictionary allows us to maintain ordering
    with open("yelp_academic_dataset_business.json", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in IDs:
                stars[data['business_id']]=data['stars']
    return stars





######################
# Sentiment Analysis #
######################

def sentiment(reviews):
    """
    Given a list of reviews, returns VADER's sentiment analysis of each review
    stored as a dictionary where the key is the business ID and the value is 
    a list of all sentiment scores.
    """
    
    print("Performing sentiment analysis.")
    sia = SentimentIntensityAnalyzer()
    docs = {}
    for review in reviews:
        try:
            docs[review['business_id']].append(sia.polarity_scores(review['text'])["compound"])
            
        except:
             docs[review['business_id']] = [sia.polarity_scores(review['text'])["compound"]]
    return docs


def sentiment_stars(sentiment, stars):
    """
    Useful function for testing. Given a sentiment dictionary from sentiment()
    and stars from business_stars(), it returns X (a numpy array where each
    value is the mean of the sentiment scores for a business), y (a list where
    each value is the number of stars for a business), and keys (a list of all
    business IDs in this collection). This is all done in one function to ensure
    the ordering remains the same.
    """
    
    X = []
    y = []
    keys = []
    for key in sentiment:
        X.append(mean(sentiment[key]))
        y.append(stars[key])
        keys.append(key)
    return np.array(X),y,keys





########################
# K-means and KNN code #
########################

def kmpp(k, X, compare=False):
    """
    Run k-means++. Takes as input a number of clusters k, and a TF-IDF matrix X.
    If compare is set to a different matrix, the algorithm will also show the 
    clustering based on those features. I use it for lattitude and longitude and
    it currently only works for 2-d matrices but that could be fixed.
    """
    
    print(f"Running k-means++ with k={k}.")
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    data = km.fit_predict(X) # First, we run k-means++
    error = km.inertia_
    pca = PCA(n_components=2) # Then we transform our data
    reduced_dimension = pca.fit_transform(X.todense())
    # Display our predictions on the reduced dimensions
    plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], marker='x', c=data)
    plt.show()
    if compare:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
        data_info = km.fit_predict(compare) # If compare is a matrix, run k-means on it
        plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], marker='x', c=data_info)
        plt.show()
        # Code below plots the two runs of k-means++ we have done by lat. and long.
        # Not very important, I just wanted to see. Feel free to uncomment.
#        compare = np.array(compare)
#        plt.scatter(compare[:,0], compare[:, 1], marker='x', c=data)
#        plt.show()
#        plt.scatter(compare[:,0], compare[:, 1], marker='x', c=data_info)
#        plt.show()
#        print(clustering_similarity(data,data_info, k))
    return error

def knn(X, y, k=8):
    """
    Runs k-nearest neighbors algorithm. Takes as input a TF-IDF matrix X, and a 
    list of labels for those points y. Optionally can be given a value k to change 
    the number of neighbors in the classifier.
    """

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    colors = neigh.predict(X_test) # Run predictions on all the data
    pca = PCA(n_components=2) # Then reduce the dimensions
    reduced_dimension = pca.fit_transform(X_test.todense())
    # Plot the predicitons on the reduced dimensions
    plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], marker='x', c=colors)
    plt.show()
    print(classification_report(y_test, colors))
    
def knn_sentiment_analysis(X, y, X_test, y_test, k=8):
    """
    Runs k-nearest neighbors algorithm. Takes as input a sentiment analysis np 
    array X, and a list of labels for those points y. Optionally can be given a
    value k to change the number of neighbors in the classifier. Different than
    knn() just because it's easier for demonstration purposes.
    """

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    colors = neigh.predict(X_test) # Run predictions on all the data
    # Plot the predicitons on the reduced dimensions
    plt.scatter(X_test, colors, marker='x', c=y_test)
    plt.show()
    print(classification_report(y_test, colors))
    
def kmpp_sentiment_analysis(k, X, y, compare=False):
    """
    Run k-means++. Takes as input a number of clusters k, and a TF-IDF matrix X.
    If compare is set to a different matrix, the algorithm will also show the 
    clustering based on those features. I use it for lattitude and longitude and
    it currently only works for 2-d matrices but that could be fixed.
    """
    
    print(f"Running k-means++ with k={k}.")
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    data = km.fit_predict(X) # First, we run k-means++
    error = km.inertia_
    # Display our predictions on the reduced dimensions
    plt.scatter(X, data, marker='x', c=y)
    plt.show()
    return error
    
    
def evaluate_clusters(reviews, max_clusters, info=None):
    """
    Not mine, adapted from online for testing purposes.
    """
    
    error = np.zeros(max_clusters+1)
    error[0] = 0;
    for k in range(1,max_clusters+1):
        error[k] = kmpp(k, reviews, info)
    plt.figure(1)
    plt.plot(range(1,len(error)),error[1:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')    
    plt.show()




####################
# Helper Functions #
####################

def categories_stars(keys,data,stars):
    """
    Very simple function which takes keys (business IDs to analyze), data (the
    output of a clustering algorithm), and stars (the output of business_stars()).
    Function returns a dictionary where keys are cluster labels and values are
    lists of the stars for each business in that category. Useful for visualizing
    how an algorithm did.
    """
    
    total={0:[],1:[],2:[],3:[],4:[]}
    for i in range(len(data)):
        total[data[i]].append(stars[keys[i]])
    return total

       
def clustering_similarity(c1, c2):
    """
    My code from project 1. Given two lists of clusterings, it computes a simple
    similarity score.
    """
    
    assert len(c1) == len(c2), "These are not clusterings of the same dataset"
    
    similarity = 0
    for x in range(len(c1)):
        for y in range(len(c1)):
            similarity += ((c1[x]==c1[y])^(c2[x]==c2[y]))
    return similarity


def sil(X,y):
    """
    Taken from sklearn's website in order to analyze the silhouette method.
    """
    
    X=X.todense()
    for n_clusters in range(2,11):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters,init='k-means++')
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        pca = PCA(n_components=2) # Then reduce the dimensions
        rd = pca.fit_transform(X)
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(rd[:, 0], rd[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()


# This code hides an annoying warning, just ignore this  
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



#############
# Variables #
#############

remove_words = ['15','20','asian','chinatown','boston'] # I've tried lots of iterations of this
my_stop_words = ENGLISH_STOP_WORDS.union(remove_words)


# Business IDs of Chinese restaurants in Boston with >= 100 reviews
ids = {'-uPcDd5ulyWh4iZm--LY-Q',
 '0oYvvuKf15nZ05iLd1TC8Q',
 '2bncbx08BFs_IO6H-yWBxw',
 '44uFnSanJ1umQRsYFm5zCQ',
 '4mClpg19Ntiw1s75kF19Bg',
 '63q10aw-6XREzuRadM3_HA',
 '6yPo1VyadJozt8KBTOhPdQ',
 '72PQGMhrEcIuWH-S44TprA',
 '7dzyLlNQXwuPsmvquuaVIA',
 '7pV_lZa8FZz2_54BvwFywQ',
 '8OlvNJkV6B4_h0GIijuuCw',
 '9Jiax4wZ94ZmMcMzmpdvJA',
 '9PLGvlh7gi8AMjL_4VySBA',
 'AA3s5dGi0fQCwBZnhFb2eQ',
 'AkaYjquGGgmjvmnii8eELQ',
 'BBblBHBnjpepOP1Q1kfi1A',
 'C9SJfJJmlqDvsgSNR569Xw',
 'ChzM9HTXqqe0VdXSbMGoqQ',
 'CzN-Nozj-x6rYU1cqSKHnQ',
 'DIRXMPneSiHoOani8QnI3A',
 'DImi8qeCP-OUaLA8L2QR6A',
 'DoS7jpSaKT7Ru2NAYDCIKg',
 'Dp6uvrBNLkimgP1-iIAo4A',
 'FDeCah18Y-SAuVvd3IZvCw',
 'G8mYHODB3zYw4RsSVebACw',
 'GVFqAZYceTZFrrzkohe59w',
 'IRQfnjyFcg3j1Z3i7kPIfg',
 'KFVgWwwGgepVeR6tDs3yfQ',
 'L7tgbiUHe6gLbvbAq9CEFw',
 'Lo7ggtNcJH_he6f9osnBfg',
 'MgAG54IJeQvT3BQoSNwWBw',
 'MnA_dF1xkX-bnLR6hq4EmQ',
 'O4D1fulSH8S-PwaX4-PdJw',
 'OztN9An_uaa38MZZ2K8a_g',
 'Px5OwdzKaMiwzvSFc_3Qkg',
 'QDlctMcZhlwJl9S0dEqtKA',
 'QbG-Ju21BhbPaNztMx8p1w',
 'RtY75rfzEP19n9Ou4DSEHQ',
 'SAEDSSDFKFO8ke1SSnTuzw',
 'SJy51U18u3zJ_TccSKpPdQ',
 'SP97o6xotOT4fATH9HDsmQ',
 'SRxyxADqvY3gb8f76nDRTQ',
 'SvkuIlhOeQZCznPbABa7KQ',
 'T6cTsRXJ8Yze9DM8L--svA',
 'T85lsS8Qz8yUNOfPj4sMmQ',
 'WX5HDdUTwYWhu6XDtawDhw',
 'XrY1nmgajhBinQmJRZ6wfQ',
 'Yza2V-zlc1iCCeiHxhdKDA',
 'ZJw7_sCOfaUmib1Q0vBAZA',
 '_A6fj7b4qwnmcEEOYOu5vw',
 'aYlIvAqd6cnHX6AGkhjq5Q',
 'acRkrnOMeIiJSdbIBqDJvQ',
 'dB42voEqPL4lcAZIe6TEQQ',
 'dmLrPLWGLGkI3qBixFDEvw',
 'expooI-n3P-brTBFKs2Qxg',
 'g3OJi94JTYhIoEwqNfhI8g',
 'gYMzNlDMnwJj5vsEX_NosA',
 'gwm_eHIetrkMaOAZVem5qQ',
 'hHVq3NS0ZW9G-ZcLurvUJQ',
 'iOoBA4u3N9ic4m4zC99brQ',
 'iQyKo3VDNzRbGQPAqFkZPA',
 'iSE_LieK4AUM8A5c0-dlzg',
 'j7VXGiKU_mvP7NrlfqVhJw',
 'joA7OEK0JqjRT3Qx7c9r-w',
 'kb0VfKZKHHhvDze3JZqi0Q',
 'mV7NZgzVGvuwAEfzz7xWJA',
 'moghdY0n6S7FWUL2eL2N5w',
 'mppWsveP2lfKLv2O7HuiCg',
 'nV4Fso9lyZJ2W_DBYL7agw',
 'n_L84L3OrtR43x8ewYHTHw',
 'nzyu4lcjIOH_zPMlaPbmrw',
 'oI2gEfrxxSdn55mK1pt2iw',
 'pnVPqIpOfpToxA6OI2P81g',
 'sgJzINGN3Njv_Q_h_Xggyw',
 'tDPO_D0tjoWUOzqgRUcDkA',
 'uJQGdkNrzEL6pTcW3MmNRA',
 'w6ikRnLVAyNjv_2WJZoY8w',
 'wOsTAvCIVjB-fPQBJepMQg'}

if __name__ == '__main__':
    print("Running example program...")
    some_examples()