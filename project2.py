

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords
import json
#import re
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier



def get_IDs(city, num_reviews=200, write=False):
    """
    Get the business IDs for restaurants in given city which have more than
    num_reviews reviews. If write is set to true then their JSON data will be 
    dumped to a xfile.
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
                    businessIDs.add(data['business_id'])
                    if write:
                        cities[city].append(data)
            
    if write:
        with open(f'{city}.json', 'w') as f:
            json.dump(cities,f, indent=2)
    print(f"Identified {len(businessIDs)} businesses.")
    return businessIDs
            

def find_reviews(IDs):
    """
    Given a list of buisness IDs, this function will search through the review
    dataset and return a list of all of the reviews about those businesses.
    """

    print("Finding reviews for selected businesses.", end=' ')
    reviews = []
    with open("yelp_academic_dataset_review.json", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in IDs:
                reviews.append(data)
    print(f"Found {len(reviews)} reviews.")
    return reviews


#def tokenize_reviews(review):
#    stop_words = set(stopwords.words('english')) 
#    punctuation = re.compile(r'[-.?!,:;()|0-9]')
#    tokenized_words = word_tokenize(review)
#    tokens = []
#    pst = PorterStemmer()
#    for word in tokenized_words:
#        word = punctuation.sub("", word)
#        if word not in stop_words and word:
#            tokens.append(pst.stem(word))
#    return tokens
    

def create_vector(reviews, knn=False):
    """
    Given a dictionary where each key is a business ID and each value is a long 
    string containing the text for reviews for that business, returns the 
    TF-IDF matrix.
    """
    
    print("Creating matrix from documents.")
    tokens = set()
    y = []
    for key in reviews:
        tokens.add(reviews[key])
        if knn:
            y.append(knn[key]) # Convert float to category
    if knn:
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(tokens)
    if knn:
        return X,y
    return X
    
def combine_reviews(reviews):
    """
    Given a list of reviews, this function will return a dictionary where the 
    key is a business ID and the value is a long string of every review for 
    that business.
    """
    
    docs = {}
    for review in reviews:
        try:
            docs[review['business_id']] += review['text']
        except:
            docs[review['business_id']] = review['text']
    return docs
    
def kmpp(k, X, compare=False):
    print("Running k-means++.")
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    data = km.fit_predict(X)
    pca = PCA(n_components=2)
    reduced_dimension = pca.fit_transform(X.todense())
    plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], marker='x', c=data)
    plt.show()
    if compare:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
        data_info = km.fit_predict(compare)
        plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], marker='x', c=data_info)
        plt.show()
        compare = np.array(compare)
        plt.scatter(compare[:,0], compare[:, 1], marker='x', c=data)
        plt.show()
        plt.scatter(compare[:,0], compare[:, 1], marker='x', c=data_info)
        plt.show()
        print(clustering_similarity(data,data_info, k))
        
def knn(X,y, X_test, y_test):
    neigh = KNeighborsClassifier()
    neigh.fit(X, y)
    colors = neigh.predict(X_test)
    pca = PCA(n_components=2)
    reduced_dimension = pca.fit_transform(X_test.todense())
    plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], marker='x', c=colors)
    plt.show()
    print(classification_report(y_test, colors))
    
        
def clustering_similarity(c1, c2, k):
    assert len(c1) == len(c2), "These are not clusterings of the same dataset"
    
    similarity = 0
    for x in range(len(c1)):
        for y in range(len(c1)):
            similarity += ((c1[x]==c1[y])^(c2[x]==c2[y]))
    return similarity
        
#def knn(k,X,compare=False):
#    print("Running k nearest neighbors.")
#    neigh = KNeighborsClassifier(n_neighbors=3)

def test(city='Boston', num_reviews=1000):
    
    IDs = get_IDs(city, num_reviews=num_reviews)
    reviews = find_reviews(IDs)
    stars = business_stars(IDs)
##    reviews = large_reviews
    tokens = combine_reviews(reviews)
    X,y = create_vector(tokens, stars)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.33)
    knn(X, y, X_test, y_test)
#    info = business_info(IDs)
#    kmpp(5, X, info)
    
    
def business_info(IDs):
    print("Finding lat. and long. for selected businesses.")
    info = []
    with open("yelp_academic_dataset_business.json", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in IDs:
                info.append([data['latitude'],data['longitude']])
    return info

def business_stars(IDs):
    print("Finding ratings for selected businesses.")
    stars = {}
    with open("yelp_academic_dataset_business.json", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in IDs:
                stars[data['business_id']]=data['stars']
    return stars
    
def evaluate_clusters(reviews, max_clusters):
    """
    Not mine, taken from online for testing purposes.
    """
    error = np.zeros(max_clusters+1)
    error[0] = 0;
    for k in range(1,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k)
        kmeans.fit_predict(reviews)
        error[k] = kmeans.inertia_
    plt.figure(1)
    plt.plot(range(1,len(error)),error[1:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')    

    
reviews_over_1000 = [{'review_id': 'xq1HkKoLzCdOlJkVzUeg5Q',
  'user_id': '0DmuCPKJ5l4otcf5ar_sew',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Still an exceptional restaurant with several new menu items including a gimmicky (and tasty) "sorta-Asian" fried chicken and waffles.  I\'m a spicy food fan with a fire resistant pallete, but the hot sauce with the otherwise-excellent chicken wings was WAY over applied and actually ruined my whole dinner.  Service has gotten slow, forgetful and a little too cool for school (second time we\'ve noticed this...).',
  'date': '2010-07-07 16:01:03'},
 {'review_id': 'nbT8-UnEryWBscBm8ccdPQ',
  'user_id': '0oZKhKKqqUqfKGr3DPz6Qg',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Love this place! Their food and drink menus are both very creative, which is perfect for me. \n\nThe Pumpkin Martini is delicious (ask for it with a cinnamon sugar rim). The Keylime Pie Martini is also amazing. They have a lengthy beer list as well.\n\nAs far as food is concerned, the mac and cheese is scrumtrulescent (half portion is definitely sufficient). Also, the Rialto is lovely, and the bread pudding is heaven (generous portion, good to share).\n\nThe service is generally phenomenal. The waitresses are really sweet and easy going, which makes the experience that much better. I've only experienced poor service here once. The girl had a pissy attitude and was simply unpleasant, but don't let that detour you from going to Parish; I'd give it 10 stars if that was an option.",
  'date': '2010-11-22 07:13:10'},
 {'review_id': '_HzC_hxPE5d-zpHZcCkxLg',
  'user_id': 'B2tIcQCyoEq-UpB11-N1JA',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'The Barking Crab was a family tradition for us for years before I moved to Boston. Once a summer, when visiting our family in the city, we would go to the Barking Crab for some decent food and wonderful atmosphere. Sadly, once I became vegetarian, the menu became extremely limited for me, which made it much less appealing. But still, many happy memories from the Barking Crab! Great family place!',
  'date': '2013-09-04 03:08:23'},
 {'review_id': 'uJTu15Y5MKCAlyYL-BTEqw',
  'user_id': 'RYVmdVIZ95mOinIXrfaJyA',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 3.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Had a amazing dinner. Price was on point nothing over the top. Food quality was great. Only thing was they were so busy that service was hard but still wasn't to a point that is bad. The lobster was yummy and so was the sides.",
  'date': '2018-04-04 00:13:26'},
 {'review_id': 'aGGs8CmDWUX-wzCrF4QYPw',
  'user_id': 'Ibnfijx091VmOXp3YCOdfQ',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 3.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'I shall give 3.5 stars. The dining environment is great and you can tell how food is cooked by sitting on the bar seats. The food is nice but a little salty. Other is good.',
  'date': '2014-05-25 23:01:43'},
 {'review_id': '4QyK34uSqW_bD-1mqEsk5Q',
  'user_id': 'UqUWk9FfF3yVXHw6p1y9dg',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Wow loved the food here! The spring rolls were divine. The raspberry lychee drink was sweet and flavorful. The Vietnamese eggplant was delightful. Very lively friendly environment. The waitress was knowledgeable and helpful. Definitely visit again.',
  'date': '2015-10-23 14:18:15'},
 {'review_id': 'Ltk21NKmKW9godxCWw7-1Q',
  'user_id': 'ao4_LR8V8vsEaP6cjL1s_A',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'I came here after a bruins game, loved it. Poutine is one of my all time favorite foods and this is the best I have had in Boston. They have a decent beer selection as well. Only complaint would have been the poutine could have used a little more gravy, was dry once I got to the bottom.',
  'date': '2015-03-18 19:42:13'},
 {'review_id': 'Osvvn5MbHPA_hualyzeguQ',
  'user_id': 'yLoc8z3K7qV1t63mELyG_A',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Would I drive hours for this pizza? Probably not. But if you are in the area- get it. It is not earth shattering, but just a very good pizza. Good sauce, great cheese and a nice crust. That's it. A place that does one thing very very well.",
  'date': '2016-01-17 13:00:58'},
 {'review_id': 'tga3pGl7bTV14dhJhMsAXQ',
  'user_id': 'vEcB6oOsINykmBzeh6FR2g',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 5.0,
  'useful': 1,
  'funny': 1,
  'cool': 0,
  'text': 'Nice unpretentious.  My son loved his steak tips   My pizza without cheese and veggies was perfect. He also raved about his cheese pizza.',
  'date': '2014-05-26 00:38:43'},
 {'review_id': 'Hem-tJJcuU7ykoXjGBPDMA',
  'user_id': 'yLS7sNv4FxvrFC-GVOun0w',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Santarpio's is not just about the pie and bbq, it's about the whole experience. The neighborhood,the line, the rec room decor, the locals, and the throw back waiters all rolled into one of best classic pizza joints in Boston. The pizza itself is thin crust (corn meal on bottom) plain sauce, and the usual cast of toppings. The pies are on the small size, but very inexpensive. Not New Haven good but pretty good for Boston. Go on summer night throw back a few cold beers with a garlic and onion pizza with some friends and call it a night.",
  'date': '2012-03-30 02:16:01'},
 {'review_id': 'srJvs5C49O6Z94XAKnEQpA',
  'user_id': 'z573LamGhVlA6msrfDJ3Qw',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 5.0,
  'useful': 4,
  'funny': 0,
  'cool': 4,
  'text': "The best fried clams in the city (state? East coast?) are served at B&G, no question.  Add the freshly made tartar sauce and it's heavenly!  A good secret is that they take reservations on the day-of, after 11am.  Sometimes you'll get stuck at a table that's too small for your party, but if you improvise, share plates and get cozy, you'll have a great time!  Or sit at the bar and watch the chefs prepare each plate and shuck oyster after oyster - you'll even get to see Barbara in action every now and then. Try the rieslings that are on the wines-by-the-glass menu as a great complement to a salty, briny oyster plate.  And save room for dessert!!",
  'date': '2006-04-25 20:08:49'},
 {'review_id': 'anKLQjDv-azuynT2TdL-xQ',
  'user_id': 'b5nItcKLY0URubUWERZjcA',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Called these guys up at 11pm the other night to get some much needed group for my well-drank and poorly fed group. We ordered onion rings as an app (super oily and terrible for you but hey that means they actually taste good). \n\nOne guy somehow ordered more plates of food than would fit on the table and they were gone within 15 minutes. I would assume he liked his.\n\nI ordered the meatloaf sandwich, which I've had many times and always love. The only issue with the sandwich is that it has the Captain Crunch effect on the roof of my mouth? Weird, but worth the trouble.\n\nAnyways, it's quite nice that this place has a kitchen open til 1am. Seriously? I had no idea you could get good food like this that late. If you come for lunch or dinner, you may have to wait, but it's worth it.\n\nThis is also the place that I took my girlfriend on our first date. Since the food was really good and our mouths were always full, it cut back a bit on the awkwardness.",
  'date': '2015-07-03 22:53:35'},
 {'review_id': 'jzGm7bKTlWluuPrYP0j8WA',
  'user_id': 'veWf1zwyUSf8NmePciOsTg',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Best lamb tips and BBQ sausage. Pizza is best around. It's a dive. But worth going to.",
  'date': '2015-10-23 13:30:34'},
 {'review_id': 'kObBQjCwbkmjJq5-wv7Rpg',
  'user_id': 'W2lmOAJRzNTxHzFxgRxHIw',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 3.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': "There are not that many places where you can dig in with your hands and some tools of shellfish distruction around the city.  Although this place is pricy, it brings back a piece of home where my family would sit around and pick at crab legs while we talked for hours.  \n\nWe've gotten numerous things over the past few times we've been here, but these stand out.  \n\nOld Bay Crawfish: 1  1/2 lbs of whole crawfish seasoned w/ old bay. served w/ lemon $9.00 Not as flavorful as Boiling Crab, but still good.  I actually think because it wasn't drowned in spices, you can actually taste the flavor of the crawfish.  \n\nMixed Crab Bowl: 2lb snow crab, 1 lb dungeness crab clusters, and 1 lb king crab legs w/ drawn butter.  (all weights are prior to cooking) $84.00.  This was definitely a splurge. I was a bit hesitant because 4 lbs of crab isn't necessarily a lot and restaurants usually overcharge.  I was pleasantly surprised when the crab that came out was practically all meat.  The crabs were cooked just right, the meat was juicy and warm, and the overall digging in with our bare hands and riping up the crab was just right down our alley.  It took us a bit of time to devoure this feast so by the end legs were starting to get a bit cold.  It was still overpriced, but if I had another celebration to attend at this restaurant I would not hesitate to order this dish again with someone.  \n\nI've been there where they have offered us a great catch, a 20lb lobster or whatever giant monstrosity they happen to catch that day.  Never partook in them, but it's interesting to know that you can if you wanted to.",
  'date': '2012-11-06 04:24:51'},
 {'review_id': '7phEXOkvB4iDsgj8wcGqgw',
  'user_id': 'WujXjjkUfb4N7cH5oXaoiw',
  'business_id': 'VImbIWfxODVsiRHebSQePw',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "I visited the Sam Adams brewery during a trip to Boston in September 2008.  It was much more intimate than I expected, as it is one of the largest breweries in the US.  It turns out that this is the original location, but more of an R&D facility currently.\n\nWe had to wait around a bit for the next tour, but it wasn't too long.  I was quite surprised to learn that Sam Adams was founded in 1984 (I would have guessed 1884, if those were my two options).  The tastings were good and we even got to sample a couple of new brews they were testing.  \n\nI haven't been to Harpoon, so I can't compare, but plan to check it out the next time I'm in Boston.",
  'date': '2010-03-12 21:01:08'},
 {'review_id': 'aJRsCniOLDBnPQw5L6LN7w',
  'user_id': 'qfMelB37HWsSPB-On1OvNw',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 3.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': 'Came in on a Saturday for Lunch to an unenthusiastic hostess who sat us down on an actual bench. The entire tent is filled with long wooden tables and thus it was not a private setting. Atmosphere reminded me of being in a large, festive camping tent, but tinted brightly orange. The dish we wanted was to split the Lobster Mac and Cheese: \n-Portion: Large \n-Cheese: bit "watery", could be thicker \n-Lobster: Average portion \n-Was it worth $27 without adding table service? Probably not.\n\nBy the way, if you\'re looking for a classic Lobster roll, do NOT get it here. I saw several faces of disappointment when the waiter brought them out. These lobster rolls are definitely underwhelming and expensive! Head over across the street to James Hook for a $20 one that will at least overfill the bun.',
  'date': '2017-04-30 03:26:44'},
 {'review_id': 'u8faJSOwHRM8LPB9HxgCAA',
  'user_id': 'YJnDWw5mAmSN75SvpJIbDg',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "eat the COPPA. best thing ever. \n\nI usually get the mashed potato with gravy as my side and then my bf gets the Boss salad and we split the two plates between the two of us. The coppa is amazing but it's quite heavy so half the sandwich is perfect amount for any foodie person. Trust me.\nService is solid. They do have a 5 drink max policy. \nIf you go there often get the beer card so you could eventually get a mug.",
  'date': '2010-08-20 22:14:16'},
 {'review_id': 'tSOOjddB1A-UA2DSnxjrwA',
  'user_id': 'WZYjAsWPzXet90BgKN_1rg',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 3.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Don't get me wrong, I am all for elevating the stature of asian food to chic and trendy, but the food here was uneven. There were some hits, like the pork belly buns and papaya slaw. The tacos were also pretty good. But there were also some misses like the tea smoked ribs and chicken wings. My main objections to the ribs and wings were that they were very salty without being terribly savory. I think that this place could be even better if the owners tweaked parts of the menu.",
  'date': '2011-11-30 02:24:32'},
 {'review_id': '1tv4zZ2hcVSyPcU4oN7vGw',
  'user_id': 'z8R5GrJVG2AVEESPopaHlQ',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "This is one of my favorite places to come eat. So far everytime i've been here the waitresses/waiters has been on their A game. The apps are good esp wings/hummus platter. My favorite sandwich is the BLT but there was just a little too much tomatoes inside. Recently i also tried the egg sandwich lyonnaise and i was in love! It's a good place to come and grab something to eat. I love their late hours too!",
  'date': '2014-09-25 13:46:00'},
 {'review_id': 'qMVE_bPYD4nSoOQ8FgYTgA',
  'user_id': '71hcP2Rtu2RnDivBVBBSaA',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "My favorite pizza in Boston.  Was there last night after having an urge for this pizza for a week.  It must be the sauce or the dough, but this pizza tastes so fresh and delicious.  You gotta be into the ambiance which is not fancy, to say the least.  It's a funky place, but the pizza and grilled meats are basic and exceptional.  Starting to want some more right now!",
  'date': '2011-02-13 16:19:57'}]
















large_reviews = [{'review_id': 'gsmDleBgB1RPUC_7mWEWtA',
  'user_id': 'OFbu_wN1ExYqP6lhbX8IsQ',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 2.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "The restaurant itself is kind of cool. It's small and all the tables are around the centered kitchen. They play really cool and hip music. I was def disappointed in the food. I had the Pan seared halibut which was over cooked and $30 for a 4oz or 5oz portion. They definitly had a great selection on oysters but are over priced. The best part of our meal was the $14 dollar banana split desert but again over priced. We started with some oyster to share then both had a main entree with 2 glasses of wine. Oh and can't forget the $14 dollars desert and the bill was around $150. I don't mind spending $150-200 on dinner but when you don't enjoy it it's disappointing",
  'date': '2012-06-28 13:46:38'},
 {'review_id': '3VrdWcWUJjvB_9ZktcEc1A',
  'user_id': 'NCy-D6NZsOuUUub9i1ppEQ',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Fantastic meal!\n\nMade a reservation on Opentable, but ended up at the bar as the place was still pretty full at 9:45 PM on a Saturday. Open kitchen was fun to watch and food exceeded expectations. Oyster prices were a bit high, but the rest of the meal was in line with expectation for the type of place.\n\nWe had fancy sweet, cotuit, and well fleet oysters, halibut tartare with chorizo, blue cod, whole fried (and salty) Bronzino and two cold blue point summer ales.\n\nFood - 5\nService - 5\nPrices - 4\nReservations - 3 (only late times, table wasn't ready, but not their fault, small restaurant)",
  'date': '2015-05-26 19:18:13'},
 {'review_id': '0tztvfgcR_NMy5ysnhuR5w',
  'user_id': 'yPbKL6-gYz4wdIsr5K8FEA',
  'business_id': 'mP1EdIafQKMuOm9O4PzAfA',
  'stars': 1.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "I have been here for lunch and that experience was significantly better. I was here on the later end of a Saturday night and the mood was trashy (for lack of a better word).  A bartender walked around table to table pouring wine down the throats of women as he stared down their blouses. Our service was really terrible - we received the wrong order three separate times and our waiter was consistently unavailable to us but chatty with other tables. I do find it notable to report that I was with three folks of color and it felt like differential treatment.  Super disappointed and won't be back.  Bizarre experience and I strongly suggest other tapas restaurants over Barcelona.",
  'date': '2016-01-18 03:25:05'},
 {'review_id': 'EQGsjNco7tgVeGxy-5voYQ',
  'user_id': 'lzMigK1zIDGH0SD7hyUE_Q',
  'business_id': 'qbpJFE-XlspCCk3PWhZ0AA',
  'stars': 5.0,
  'useful': 7,
  'funny': 2,
  'cool': 3,
  'text': 'Wow, some pretty harsh reviews of the HOB. Not really sure as to why? \n\nJust saw Aaron Lewis play here and had absolutely NO problems with the show or facility.\n\nAfter reading the reviews, I was expecting the worst and almost felt let down by the lack of problems (I mean, some of you are really laying it on thick about the place....).\n\nI had floor seats and got there when doors opened. We passed through security with no issues and sat at our seats without issue. \n\nYes, the beers are $7.00 for Bud Light, but how much do you pay at a Sox game across the street? How much do you pay for a beer at one of those ultra-chic fly-by-night clubs downtown? How much is the beer at the Garden or the Mohegan Sun (yeah, it is more than $7..). Why the concern that the HOB charges $6 or $7 for a beer is a bit overkill.\n\nThe show sounded great. The bathrooms were better than most places. I did not see overcrowding and when the show was over, we made it out of there in about 2 minutes. \n\nOutside of having Aaron play in my living room, I am not really sure how much easier going to a show here could have been?\n\nPerhaps the problem is not with the HOB, but many of the Generation-ME folk that have little to do but complain about how everyone is always making their parent-supported lives so difficult.\n\nBeen to lots of shows and Clubs in my time and the HOB was by far one of the better ones that I have visited.',
  'date': '2010-10-10 01:35:17'},
 {'review_id': '5gJ_rvOsyK-QzBuWz_fgOw',
  'user_id': 'Y5uc9gvW-bY7bs9BqK9okQ',
  'business_id': 'nqKL5PbJbwwoCK_Xon31kA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Food was fine the one time I went. The setting was great if you like oldy worldy wood panelling. Made me feel like I was in an old movie (in a good way).',
  'date': '2015-06-22 00:19:25'},
 {'review_id': 'K9IJiGbrIivejwhPYziMtg',
  'user_id': 'bnyL1Eiy6posyEf6ehhQmg',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "skip the frites - they weren't anything special and neither were the sauces - i've had real ones in europe and these just didn't compare...maybe we went on an off day but they didn't have flavor, were soggy, and the sauces did not impress me.  i wished they had a thai peanut sauce, too.\n\nthe waffle, however, oh man.....now THAT was the bomb.  salted caramel - i ate two by myself. unfrickinbelievable. so my 4 stars, that's only for the waffle.  the waffle also needs a pure chocolate option or something, not just nutella. i know this is blasphemy, but i just don't like hazelnut.",
  'date': '2011-07-03 02:54:10'},
 {'review_id': 'LF-k0eHw_BeitsDkWpHEOg',
  'user_id': 'xB80M3Wn0Nqrmvt-PyqtZA',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 2.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': "i randomly walked by this cafe today and was hungry, so i stopped by. not a good idea. this highly overpriced cafe itself is a very quaint little place, close to the garden. i liked the outdoor patio, but unfortunately it was way overcrowded. i had the regal regis, a steak and mushroom sandwich on french bread. the meat was nothing to rave about, it tasted overcooked and salty. the bread was decent when warm, but upon cooling became hard and tasteless. my meal's only saving grace was the cole slaw, which was pretty delish! if i ever come back to this place, it will be for the cole slaw only-- and maybe the bread pudding (saw another patron's dessert and reaction upon chowing that puppy down), which looked pretty fantastic.\nthe service was good initially, but the waitress took over 10 minutes to give me $10 in change, which she eventually gave to me in ten $1 bills. thanks.",
  'date': '2007-05-28 01:51:03'},
 {'review_id': 'PQuHpTnyJy-XwltJWe2jyA',
  'user_id': 'vzRNVVXLcaKeGOMSW6tkeA',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Best Chinese food in Boston! The udon noodles were so good we bought their cook book so we could make it at home',
  'date': '2018-09-25 23:38:27'},
 {'review_id': 'RD2aQBjegDsAUUiUO1_9Xg',
  'user_id': '36EO5HNYz-FzPxU8nluhSw',
  'business_id': 'nqKL5PbJbwwoCK_Xon31kA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "In this economy it makes too much sense to get as much bang for your buck as you can, and Maggiano's takes the cake here.  I love Italian food anyway, and I also love heaping portions of food, no matter what style or nationality.  If you do it right here you will not only be fully satisfied but you will have leftovers for days, which me and my friends had the last time we went here.  The key is to do the family dinners, eat as much as you can of the first portions, order the free refills, eat a little bit of them, then have them bagged so you can take them home.  It's the greatest two-for-one deal in town, and again, in this economy, you can't take something as simple as food for granted.  If I had a last meal, it might be here.",
  'date': '2009-03-09 18:52:45'},
 {'review_id': 'ugG_TQAkVPM6zsZvrU4rfQ',
  'user_id': 'Wyad1gr5iCCwTwhUTij9fg',
  'business_id': 'mP1EdIafQKMuOm9O4PzAfA',
  'stars': 2.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "I was in the neighborhood so I went there for lunch. The decor was modern but the tapas were mediocre. The waitress's service was great. \n\nWe ordered eight tapas.Jam on & Manchego Croquettes and truffle bikini were good. Farm egg, apple pancakes, shrimps and monk fish were average. Grilled hanger steak and prawns were horrible. Hanger steak needs some technique to cook it delicious and I am sorry to say the chef does not have it. It was one of the worst hanger steaks I've ever had. The prawns were not fresh. After one piece, I didn't want to eat it anymore.\n\nLater on, I checked out the restaurant's website and found out it is a chain restaurant. Like most chain restaurants, the food has no soul, no character and is mediocre. It is true that through the food, you can feel the chef's energy without meeting the chef in person. I definitely will not come back.",
  'date': '2017-08-29 22:03:24'},
 {'review_id': '1qqAvx--x8XDPbx_4qtIBg',
  'user_id': 'JBIvJ-JMlwkn-7EM_rqQ4w',
  'business_id': 'mP1EdIafQKMuOm9O4PzAfA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'This place is one of my favorites!\ngreat food and fair prices.\nService is excellent despite the well deserved crowds',
  'date': '2016-01-18 17:52:13'},
 {'review_id': 'kJkaImPkbR63kCVls2HRTQ',
  'user_id': 'LKhHhlEn38TFH1I8ACX3hg',
  'business_id': 'VImbIWfxODVsiRHebSQePw',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 1,
  'text': "The parking situation here is TERRIBLE but the tour and brewery experience is FANTASTIC. I have been at least 4 times (usually when I have friends visiting from out of town). The tour guides are funny and informative and the tasting is super. Go to Doyle's before (and hop on the trolley) or after and grab a bite and a pint (if you order Sam Adams, you get the glass for free!)",
  'date': '2012-03-03 14:09:08'},
 {'review_id': '6hHWqVxwi4tp4BUbldq2iw',
  'user_id': 'yNEuxv_oq2w2bVkMy7Gz6A',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "We have to give this 4 stars for overall quality and service. Oysters excellent but $'s. Fried fish ok. Louis salad outstanding. Fried oyster roll delish. Nice bellys. Clam chowda looked great. \n\nGood drinks. Bloody Mary bloody good. \n\nCrab cake decent. Ceaser Salads with grilled seafood was huge.",
  'date': '2014-03-16 17:09:15'},
 {'review_id': '6o55A_FwJtiHE48M7fSdAQ',
  'user_id': 'vF55AGOxrwaxtyHPA_-Pnw',
  'business_id': 'VImbIWfxODVsiRHebSQePw',
  'stars': 5.0,
  'useful': 0,
  'funny': 1,
  'cool': 0,
  'text': 'Me & my 3 out of town friends went here & had a great time - the tour is really interesting (especially since 3 of us work in the food industry) & our tour guide was very funny/informative.  The sampling portion was fun even though I am not a beer drinker.  I would definitly go back & bring more out of town/visiting friends therer.  All MA locals should definitly go as well :)',
  'date': '2011-03-22 00:48:38'},
 {'review_id': 'Ut8X0fjgl9Vl5aXryAS9hA',
  'user_id': 'z8CYW_rmusX_jMJANPbo8w',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Nice views on the water, fun atmosphere and tasty seafood! Lobster rolls were delish!',
  'date': '2017-10-25 12:38:05'},
 {'review_id': '6BqyjAlOfxtxtq9pu7U-5Q',
  'user_id': '2kgXYJrcgXuJZPhpfUeUew',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 3.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': "Honestly I forgot I've been here. It's fine enough pizza, we got it for carry out and it took long enough that we almost just left, but we wanted to try the famous santarpios. That was two years ago and I live nearby but haven't bothered to go back, if that tells you anything.",
  'date': '2017-07-29 13:04:08'},
 {'review_id': 'SHepd0iupzMknICVhs3kjQ',
  'user_id': 'HrGyeHrKIgX5tlibSdUH6g',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Second visit to Saus have to say absolutely delicious!! The staff is very friendly. I ordered the herb chicken sandwich and a side of frites... to die for on my first trip. Second  trip I ordered the spicy tuna on the brioche roll and picked up the grilled cheese and mushroom for a friend who said it was absolutely delicious!! I will definitely be back!!',
  'date': '2014-08-11 21:14:50'},
 {'review_id': 'AJ2O9hdvb8RXnvHxDNx2Ew',
  'user_id': '4iJnq4hZAdVt4JNqYa1mlA',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 4.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': "My wife and I spent a great afternoon at B&G's and tried a wide number of items on the menu.  My favorite oysters were the Bristol Bay but I also tried the Island Creek and Conway Royale.",
  'date': '2010-01-12 00:13:03'},
 {'review_id': 'ebU9T-7VN_8NLjhAOOUR0w',
  'user_id': 't7C7Esa_uKD_fq0x2-nIdw',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 3.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Went last week with my boyfriend after work. He ordered a Sam Adams and we split the fisherman's platter. Personally I felt like they were skimping it on the seafood. The piece of fish was thin and smaller than a dollar bill. The coleslaw was bland and dry like they forgot the dressing. Tartar and cocktail sauce had absolutely no flavor. The shrimp were to die for though. Our dish was okay, but for 35$ for one beer and a meh entree we split, I wouldn't go back. There's probably somewhere else in Boston that is better.",
  'date': '2014-06-29 11:15:06'},
 {'review_id': 'GoKlGM4S6uCPHArkwl7aHg',
  'user_id': 'xz41B2HzxbrgY8MEuV1Sdw',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 2.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Live around the corner and been here about 10 times. Each time they seem to feel more entitled. Not sure why or what but they have great fries.  Yesterday I order two of their special fries and a sand which but received two normal fries and sandwiches.  I wish them luck and again they have great fries.  Weakness is the super hipster employees that stare at you for ordering anything extra.',
  'date': '2015-11-07 14:42:47'},
 {'review_id': 'fLYtaTCBnxrzRFfkQIOoBg',
  'user_id': 'Otn6tYIKmFAdjnZyTW35Fg',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 2.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "We were seated after the requisite 40 minute wait, and we put in an order for drinks--only to wait nearly half an hour for our wine and beer to arrive.  The chicken sandwich is very delicious, but I suggest The Parish Cafe staffs up to accommodate the amount of people they serve...we shouldn't have to wait that long for a drink and ask three times about the status.  If you go in expecting a wait and sub-par service, you won't be disappointed!",
  'date': '2013-04-12 13:34:57'},
 {'review_id': 'DtOLC_ufWO51F-Xvn0U-ZA',
  'user_id': 'BALidQIfm4es2c6uZI0Uhw',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 4.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': '$1 oysters on Sunday from 8pm onward!',
  'date': '2013-08-05 21:00:49'},
 {'review_id': 'eB1zCBbshI5s_1oQjQuXzg',
  'user_id': 'ehlaQ48Vw2sMOR1Ll8mdvQ',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Really great selection of oysters, the whole kitchen is visible to all the customers including shucking the oysters, everything is really fresh and a nice Presentation with all the plates. Best of Boston!',
  'date': '2012-08-18 17:27:35'},
 {'review_id': 'mG-AM4fhLCzncSbS_7ottg',
  'user_id': 'vF4NusaRF9Mnpv4j4m1YEw',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 4.0,
  'useful': 2,
  'funny': 0,
  'cool': 0,
  'text': "First time checking out this place last Thursday while on vacation. \n\nOrdered King Crab and the 1.5lb lobster tail, and crab cakes for an appetizer. (You're on vacation so why not enjoy!) I felt the food was expensive but it was very good. Fresh seafood is amazing and you get what you pay for. We waited for an hour for a table on the patio because they were very busy. Service was good, not bad and nothing spectacular. I would love to come back and try more off their menu!",
  'date': '2015-05-13 13:26:24'},
 {'review_id': '3Lv2Qmovg9-1r_jsfCatLw',
  'user_id': 'T3k8yd4k66U2BtaebW05lw',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Came here and tried the poutine, mini chicken and waffles, and the dueling chickens (herb roasted chicken). Everything was delicious! I wasn't a huge fan of the poutine mainly because of the gravy, but the chunks of cheese were on point. I LOVED the green monster dipping sauce - it had the perfect amount of spiciness. Will definitely be back.",
  'date': '2014-01-19 20:22:16'},
 {'review_id': 'xxlh2iETZ_L7qIlMRBgR4A',
  'user_id': 'qqsAznx0uUoD1sFyaFsmxg',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "FOOD: 3 of us shared 5 dishes: pork and chive dumplings, yu-choy, pork belly buns, duck wraps, and lamb noodles. The dumplings were perfectly crispy, juicy, and flavorful. The yu-choy was on point, I couldn't get enough, and the baos were also super tasty. The noodles were alright, the lamb was tender but the overall dish wasn't too memorable. The other dish I was meh about was the duck wraps; the duck was delicious but the actual pancakes for the wrap were pretty dry and tasteless even when the herbs and sauce was added. They should add on the menu medium spicy level especially for people that don't know that nam prik pao is chile paste. There's tons of options on the menu so I'm excited to come back and try other items on the menu!\n\nDRINKS: We got the pimm's punch, pineapple express, and the drink with papaya and they were all awesome\n\nSERVICE: Everyone for the hostess, the chefs (kitchen was open so you can see everything being made) and our waiter were super friendly! Our waiter suggested adding a pork bun to the order since it usually comes with 2 so the 3 of us could have 1 each which was helpful.\n\nAMBIANCE: Small space so reservations are definitely helpful. Came Thursday night at 6:30pm and pretty popping. The restaurant vibe was trendy and complemented the menu well with the asian inspired decor and chinese newspaper place mats.",
  'date': '2015-12-06 02:34:39'},
 {'review_id': 'CT6p5vnhePhTPgRN5-5WkA',
  'user_id': 'zTheDIJdMrd4tq9LScgwdg',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 3.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': "So this would have been a great restaurant because the sandwiches are fantastic, but I have to let you know about the mouse slaughter I saw here.\n\nEating lunch. A mouse runs out and a group of business men start shrieking and all leave. Then the servers CRUSH THE MOUSE TO DEATH WITH A CHAIR. \n\nSo naturally, I was a little upset. The crushing took a long time, and there was a smear to clean up. I feel like there were better ways to handle this mouse situation. Like any other way.\n\nThe servers apologized and offered us free dessert. No thanks. I wasn't hungry.",
  'date': '2015-01-20 01:31:26'},
 {'review_id': 'bbHLoZgsS4ZDN5Ymsiwcxg',
  'user_id': 'BZ9s2qX077wWs7HIOer5xg',
  'business_id': 'VImbIWfxODVsiRHebSQePw',
  'stars': 5.0,
  'useful': 5,
  'funny': 2,
  'cool': 3,
  'text': "One of my favorite adventures during my Boston trip!!\n\nFree tour, free beer tasting, and a great witty tour guide showing you behind the scenes of the brewery. We got to taste the individual ingredients and thought process that goes into each beer, and I definitely left with a newfound appreciation for the taste of beer. Plus, you get to keep the cup!!\n\nA shuttle will then take you to Doyle's, a historic bar not too far from the brewery, where the clam chowder and lobster roll were absolutely out of this world delicious and fresh!!! When you get a Samuel Adams beer here, they let you keep the special glass (your tour guide will explain to you why this glass is quite special) \n\nAny donations go to local charities!!! \n\nVery easy to get to via the T. Only a few blocks from the station.",
  'date': '2013-07-15 05:40:56'},
 {'review_id': 'Y69wW2Cf5HT44Pd2GbtmAA',
  'user_id': '-nBLpS6wiyMLu_qN-eRbiw',
  'business_id': 'nqKL5PbJbwwoCK_Xon31kA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Went here with 3 of my girlfriends for my belated birthday. We started out with stuffed mushrooms , spinach artichoke dip, & mussels (lemon butter) and it was soo good! (thanks to the yelpers!)  The portions was decent and by the time our entrees came out, we were already full!. I got the Beef tenderloin medallions that's come with creamy mash portatoes topped with crispy onions & sauteed mushrooms and a side a asparagus. It was pretty flavorful and I couldn't finished it so I took it to go. The environment is nice, definitely good for family dinners/get together. I wanna try the all u can eat that's $30 per person next time!",
  'date': '2012-04-30 17:50:51'},
 {'review_id': '3sG4dVmGWYU8Z-Tzic9sIg',
  'user_id': 'evzzR3OdUpShuytD_1wsQQ',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 4.0,
  'useful': 1,
  'funny': 0,
  'cool': 1,
  'text': "Had a very good dining experience at Myers + Chang. Pork belly buns are out of this world. Can't stop thinking about those). My friend loved the noodles.\nThe decor is very inviting and relaxing. Excellent place to get a drink.",
  'date': '2013-03-19 22:43:39'},
 {'review_id': 'VpFZdswZL0WzgyNEfdrLaQ',
  'user_id': 'HJ2u8bYxkVDRnFdlFzuZdw',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 1,
  'text': "Just had a very pleasant dinner here. I'm in town by myself, and by getting there a little before 6 PM didn't have to wait at all for a table. The hostess was pleasant and my waiter was great - very friendly and outgoing without going over the top.\n\nI had the New England Fish Cakes, which came with a very yummy cream corn sauce of sorts and garlic mashed potatoes. Perfect portion. Only minor thing is that the potatoes were only warm. Not a big deal, and not worth doing anything about, but they were good anyway. I found the prices to be quite reasonable for the unique and creative dishes.\n\nDuring the 30-45 minutes I was there, it really got crowded and a line started to form waiting for a table. I suspect the place really gets hopping as it gets later, so maybe keep that in mind: if you don't want to wait, go for an early dinner.\n\nI suspect I'll be back there again on this trip. There were a few other interesting things on the menu!",
  'date': '2013-04-12 23:03:31'},
 {'review_id': 'z3jLre-GN8PbW7dAIQ3qRA',
  'user_id': '8gxNYDZmexw9f7nZz345iw',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 4.0,
  'useful': 1,
  'funny': 0,
  'cool': 1,
  'text': "This is always on my list of places to go while in Boston. I am lucky enough to have an office right down the channel from here and get to come here often. First, the 4 stars - its not a date place or a destination restaurant to go for a fancy meal!!!!! The Barking Crab is a great casual place for sitting outside, having a few cold drinks and satisfying your appetite! They have a huge outside area with standing room area's as well as large picnic tables if you want to grab a bite to eat. They also have indoor seating. Also, late nite is pretty cool - they have live music outside. \nWe usually get the crab cakes (either the app or the sandwich), clam chowder (NE of course) or the steamed shrimp - all are very good and don't break the bank. On this last visit, i steered off and got the lunch version of the mahi mahi - it was really good and had a very fresh mango chutney as a topping. \nAgain, while in Boston, if you need an attitude adjustment after work or while touring...or you just want to sit outside and get a decent casual drink/meal - this is the place!",
  'date': '2011-05-29 20:07:13'},
 {'review_id': 'iyTlVyFZrs1KsrmZvsloJA',
  'user_id': 'B9rBamGky99I1eZK9MWffQ',
  'business_id': 'MjpH-uP90jTUp_KqAOiBJg',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'My recommendation is the pasta bolengnese. The bread is amazing and the service at the bar is pure entertainment. \n\nIts a great place to spend a fun weekend late evening with friends. There is always a great crowd there and the loud music and dim lights makes it even more appealing. \n\nDefinitely a place for a more mature crowd which is totally more my scene.',
  'date': '2010-01-14 03:16:06'},
 {'review_id': '9Q8BNKr-kK1WCSm1d4z0qg',
  'user_id': '0khVyYZ2GBALQ0_ZXrxv7g',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'This restauranta was yhe perfect spot for taking my family to a tasty lunch.  From reserving our table, which was done with the reservation tool on yelp, to getting there with ease and being sat down right away when we called out our reservation, this surely was a no hassle experience.  We ordered some oysters and calamari for an app and then four of us shared two bacon lobster rolls.  The proportions were perfect since I definitely could not finish the who roll by myself.  The fries and coleslaw were to die for and the lobster roll itself was light, unlike the overly mayonnaised roll I was expecting to receive, although the roll could have used a little bit more flavor.  All in all, great food and fantastic location!',
  'date': '2013-11-16 14:33:15'},
 {'review_id': 'Aw7tWzAhhlX-apFZtmR_Tg',
  'user_id': 'u2hUREwMahrsAGOlM7tA4A',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Ok fine, I'm a pizza snob and I usually get super judgmental when people tell me some place has good pizza.\n\nHowever, this was so good. It was different than most pizza I have had. We ordered take out during lunch hours and it was good even at sort of room temperature when it got back to our office. \n\nI don't know how to explain the difference in the pizza but the cheese on the top (I am assuming parmesan) is good. I would definitely recommend!",
  'date': '2014-10-28 02:54:14'},
 {'review_id': '2pA1SuAdsCOHL9XTK8SniA',
  'user_id': 'qzBOmz2KrI2tSrwkzICsnQ',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 4.0,
  'useful': 0,
  'funny': 1,
  'cool': 0,
  'text': "everything in this place is so frikken good. especially the salted caramel waffle...nom nom nom! \n\nservice excellent, food great, good price! \n\nthe only reason 4 stars is that they really need a bigger space to accommodate seating...but it's a small boston eatery so they manage just fine. just don't expect to get a table right away at lunch time! :)",
  'date': '2011-12-02 15:52:45'},
 {'review_id': 'y-au6yzeBuXh3GjyTkBMQA',
  'user_id': 'GmRcci8kgknB0cfq2cAOCw',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 2.0,
  'useful': 0,
  'funny': 3,
  'cool': 1,
  'text': "1) I can't believe this place is still standing (but probably not for long right?). \n2) I can't believe I actually agreed to meet friends here recently.\n3) The food here is just not good.  Sadly, tourists think it is.  \n4) How can you get a lobster roll and fries in Boston and have it not be good? I have no idea but it happened to me here a few weeks ago.  \n5) They used to be able to make up for the crappy food with decent atmosphere and view but now they don't even have that. I won't be sad to see it go.",
  'date': '2014-07-28 18:32:56'},
 {'review_id': 'RwkVSDIuvIishO1KbT3orA',
  'user_id': '9tFvWg_EbrIYPieMrScUyg',
  'business_id': '6fF-nAA2AWTPYF2vlOzqtg',
  'stars': 3.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': 'i hate to be the first to give this place a less than stellar review, but...\n\nmaybe it is because i heard such wonderful things about this place my expectations were too high?  the pizza is ok...i have had better.  i was told the steak tips were dry, and more fatty than meaty.  the service...not so hot.  the atmosphere...like a diner, but not in the kitchy, cutey way.\n\noverall, not some place i would go out of my way to eat at.',
  'date': '2006-01-23 04:13:42'},
 {'review_id': '5Oo0xIq2xscLA3TeYr939Q',
  'user_id': 'KiyPUvmz2HDmZvP8yiiuxw',
  'business_id': '5HMXgD_gui5n0Tc_hadesg',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'I really really liked the poutine, which is pretty much the only reason this doesn\'t get three stars. We got the out of control poutine and it was absolutely delicious. So big though and probably could have been our whole meal since there was pork on it. My parents both ordered the carpet burger and I got the pulled pork burger. Word of warning, the "burgers" are all burgers. I am used to when something says "fried oyster burger" or "pulled pork burger" that just means it\'s going to be served on a bun like a burger and we were in for a big surprised when the fried oysters were on top of a beef burger and my pulled pork was on top of a beef burger. My mom is allergic to beef so this was a big problem for us and the waitress kind of treated us like we were idiots for not knowing what their version of a burger was. Maybe we were naive but I\'ve been to many places where our version is right! Anyway, they replaced my mom\'s meal with just a pulled pork "sandwich." I will highlight that the portions at this place are CRAZY. Way too much food I\'d say. But when it\'s delicious, who\'s going to complain?',
  'date': '2015-08-20 15:09:36'},
 {'review_id': 'eiiUtLK4Lkgu1nBv8lA7WA',
  'user_id': 'pIyaIJdVa_Kl8vebITpIew',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 1.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Worse service ever!!! We were on vacation and we showed up at 9pm when the sign clearer says 10pm and on the website says 10pm we tried to get a seat and the hostess spoke to someone and they said they were closed?!?\n\nClosed I'm pretty sure we had an extra hour to be seated and to eat before it was going to be close.\n\nWe were so appalled by the service and why would you kick out people when they want to pay for your services???\n\nAnd it wasn't just us the people right behind us got kicked out too that's just crazy!!\n\nWill never come here when I come back for vacation!",
  'date': '2014-09-11 16:09:34'},
 {'review_id': 'h09i2bbkkCudQWMCUufFyg',
  'user_id': 'Ec80MhfFrAQgVqHeDnHZFA',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 3.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Every time I am in Boston I have to stop by the Barking Crab to enjoy their lobster rolls. It's a generous serving for a reasonable price and most important delicious.\n\nFrom the dining area you have a nice view over Fort Point Channel and Boston's Downtown/Financial District on the other side. The restaurant is super casual and you are seated on long tables with other guests next to you or in the pub/bar.\n\nIf you are looking for a private dining experience don't come here. It is noisy (I think they increase the sound volume each year) and packed. Moreover, be aware that the hosts are not the friendliest people around. However, the waiters and waitresses make up for that.\n\n+ Delicious and inexpensive lobster rolls,\n- noisy, \n- hosts have attitudes.\n\nAverage entree: $18-25.",
  'date': '2013-10-04 19:36:42'},
 {'review_id': 'xbnY7PsmZ9hZeR-JF_2zNw',
  'user_id': 'HXOBfK10HDsZz4HYpZyeuA',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 5.0,
  'useful': 0,
  'funny': 1,
  'cool': 0,
  'text': "A delicious take on a baan mi at a little American cafe not far from newbury? Yes!\n\nThe bread was crusty, the sauce perfect and the chicken cooked beautifully amid the wonderfully sliced veggies. Who knew grilled chicken could taste so good wrapped up in Thai basil?\n\nThe salad was good but had maybe a bit too much fish sauce.\n\nIt's all okay because the sandwich was $13.50 right next to Boston common. Amazing! Definitely worth a stop.",
  'date': '2017-01-16 19:45:16'},
 {'review_id': 'PV431W0jRXvXyN8mD18Qvg',
  'user_id': 'VBZazxDBOPqQWFSmUnBaDg',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 5.0,
  'useful': 1,
  'funny': 1,
  'cool': 1,
  'text': 'Waffles with nutella and caramel.  \n\nDo I really need to say more?\nGreat staff, great idea, cool little spot.\n\nProbably my spot for a quick snack in Boston from now on.\nOh yeah, the waffles are AWESOME!',
  'date': '2011-04-25 18:09:36'},
 {'review_id': 'zbpuwMFFrE6TkCUoUQa6AA',
  'user_id': 'KQhLF7fFymuvUJNf7SoUoA',
  'business_id': 'J1uidHIL7nE_noUuvFXj0A',
  'stars': 5.0,
  'useful': 2,
  'funny': 0,
  'cool': 2,
  'text': "I went to Grotto during restaurant week after reading all the great reviews on Yelp. I was SO glad I did because this was the BEST meal I have had in Boston to date. We started with a bottle of Chianti which was reasonably priced and fantastic. \n\nFor appetizers, we ordered the crab ravioli and the cheese fondue with beef tenderloin. The ravioli so light it felt like you were eating air. But the cheese fondue was the one that was spectacular. We scraped up every little last bit of cheese with the bread.\n\nFor our entrees we ordered the potato gnocchi with short ribs and the bolognese. Both were perfectly cooked and delicious but the gnocchi had flavors that my mouth had never experienced. The gnocchi and the fondue are a must if you ever come here. We ended our meal with the chocolate cake and tiramisu. Both were good but can be skipped if you're full from your meal.\n\nWhile the service was not great, the food more than made up for it. And at those prices, this is a place I will come back to over and over again.\n\nP.S. If you're lazy like me and decide to drive here, make sure to check the street signs before you park. I was able to find a lot just up the hill from the restaurant for a flat fee of $10. Especially with the increases in parking ticket fees, not a bad idea to just pay the $10 and not worry about your car for the night.",
  'date': '2008-05-11 21:19:03'},
 {'review_id': 'k3L685TqsOoRP5SgpjTlmQ',
  'user_id': 'Nw1Qu1V8KcFj8fBA3H2qUw',
  'business_id': 'nqKL5PbJbwwoCK_Xon31kA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'really enjoyed it, nice meal!!!  i went there w/ my bf and had really good food and we were stuffed.  must try their seafood pasta, alot of seafood, shrimp, clams, lobster cooked in tomato sauce for under 20 bucks!!!  really good deal.  i love the fact that you can ground your own peppers and salt.  love the place, good place,good food, nice atmosphere, and good reasonable prices.',
  'date': '2011-01-10 15:40:51'},
 {'review_id': 'QTh8C0SDk9jHs9CasHFwyg',
  'user_id': '2ZSFQGYbk7ZneVOp1bU37w',
  'business_id': 'mxjVk5rvPNhzYe_vt3OSQA',
  'stars': 5.0,
  'useful': 1,
  'funny': 0,
  'cool': 0,
  'text': 'Tucked away below the street and full of great food and people.  Would definitely return here.',
  'date': '2017-06-07 14:43:55'},
 {'review_id': 'rGz-17nLQfHpHFX3NZvcnA',
  'user_id': '6NNjeaaLL1UCtBcgLUjvFA',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Fantastic service! Quality food with incredible ingredients. I loved everything about this place and highly recommend.',
  'date': '2017-03-28 23:22:17'},
 {'review_id': 'wzd2yz8abOlSlHTOB0xhRA',
  'user_id': '-tdsrQ3QIkGmmP2n6-DTeg',
  'business_id': '5HMXgD_gui5n0Tc_hadesg',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Maybe it was because I was ravenously hungry. Maybe it was partly due to the thrill of visiting Boston. Either way, the tofu burger that I had at The Gallows was one of the most delicious burgers I\'ve had in my life. \n\nWe walked in 20 minutes before the kitchen closed and scrambled to get our orders in. I love that there are a variety of vegetarian-friendly options on the menu (although I still wonder how a main of "Cornish Game Hen" can be made vegetarian...), and I opted for the classic "Our Way" burger with cheese, onions, lettuce, and pickles. The burger came out with a side of fries and two huge chunks of tofu that rendered the thing nearly impossible to eat (I had to wimp out and go the fork and knife route). Everything about the burger was delicious, from the made in house pickles to the sweet grilled onions. \n\nOne star off for a server who seemingly didn\'t know how to smile and couldn\'t be bothered to be congenial in any way. Regardless, I would add a visit to The Gallows to my itinerary if I was ever to visit Boston again!',
  'date': '2013-05-30 22:25:27'},
 {'review_id': 'TYKSLXhC1e-uPSsTy_ineg',
  'user_id': 'opPpHVIp8SKHMJg3_y5Lew',
  'business_id': 'VFvCFOYtyK9ae4Skxvf3vA',
  'stars': 2.0,
  'useful': 8,
  'funny': 5,
  'cool': 2,
  'text': "No thanks. Didn't have a good experience here. The beer was the only thing that kept my sanity here.\n\nThe poutine here is a disgrace, and shouldn't be advertised as poutine. Fries were cooked well done, and they were skimpy on the gravy. \n\nI'm now locked into this mission in finding some good poutine joints just so I can remove the memory from coming here.",
  'date': '2017-01-17 01:08:04'},
 {'review_id': 'VVf32g17bqhs9TtfLBGJYQ',
  'user_id': 'D4CHdcVnDzdC-qRuCZaLYA',
  'business_id': '_QUh5vFHSuw8R_uiFZ7XKQ',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "I can tell that Parish Cafe is accustomed to dealing with the busy downtown Boston lunch rushes, because their service was incredible. My boyfriend and I went to grab a quick dinner on a weeknight, before heading to the movie theatre and we had one of the best dinner experiences that we've had in awhile. We were a bit pressed for time, but our waitress seated us immediately, gave us a couple minutes to decide what we wanted to order, and was back and ready to take our orders the second we set our menus down. On top of that, it didn't take more than 8 minutes or so for our food to come out! And boy was it amazing...\n\nWe ordered the Le Mistral sandwich and the Rialto sandwich to share. We must have picked the best two options on the menu, because I don't think it's possible to have ordered anything better than what we had. The garden salad (with the Rialto) had the most amazing dressing on it, and was the perfect addition to the sandwich. The sandwich was simple and contained a small number of my favourite things: prosciutto, pesto and buffalo mozzeralla. How could you ever go wrong with that?! My boyfriend ordered the Le Mistral sandwich, which was the most expensive option, and rightfully so; it was incredible. Perfectly cooked beef tenderloin with mashed potatoes, cheese and crispy onions; it tasted like one of my favourite comfort meals packed into a deliciously compact sandwich. If I were to recommend one item at Parish Cafe to a friend, it would be Le Mistral. As tough as it may be to justify paying $19.50 for a sandwich, I can assure you that it's worth it.",
  'date': '2016-02-25 14:57:51'},
 {'review_id': 'jnXaSybkvEH7QYqiGiFpVQ',
  'user_id': 'THd14M7knK0DgXl6zyh-Gg',
  'business_id': 'mP1EdIafQKMuOm9O4PzAfA',
  'stars': 5.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Delicious. The croquetas and Migas were my favorites. They accommodated my party of 7 on a Tuesday night wonderfully. Great ambience.',
  'date': '2016-01-30 04:26:51'},
 {'review_id': 'DIiQxXXZo7_-zBEC6SEdBQ',
  'user_id': 'cASe2VEmuP3itXijOpAC4A',
  'business_id': 'mP1EdIafQKMuOm9O4PzAfA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Food was great and service super helpful and knowledgeable. \n\nOnly down point was that it just didn't seem to meet the same awesome vibe as the other Boston location. This one seemed a little more plain and corporate. Still, a good time was had by all.",
  'date': '2017-12-17 19:29:57'},
 {'review_id': 'tmvkERd350euhZfyDsKjYQ',
  'user_id': 'PsIDu0JlCKfDQ4SG2yPRYA',
  'business_id': '72PQGMhrEcIuWH-S44TprA',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': 'Finally after many attempts to make reservations on yelp or open table to no avail , . We just walked in for early dinner on Monday memorial weekend , and fortunately they had a table for 3 (even though when we called and tried to make reservation on line they said no availability !!!! \nWe were more than happy with the menu and our server (Marlene) was very helpful in explaining everything . Many great choices for vegan, vegetarian and carnivores. We all left very pleased and happy . Love to go back and try other menu selections',
  'date': '2018-05-29 21:30:47'},
 {'review_id': 'vWTxF3p0l4jPn0ityp3RRg',
  'user_id': 'tY6zPPYWiOu_BeT3vWLWyA',
  'business_id': 'VImbIWfxODVsiRHebSQePw',
  'stars': 4.0,
  'useful': 0,
  'funny': 0,
  'cool': 0,
  'text': "Wonderful tour! I loved the great way they did sampling at the end and allowed you to actually taste hops and barley. I have some other tours that I have enjoyed more but their sampling process at other breweries left something to be desired. Tour takes about 45 minutes. \n\nTake the trolley to Doyle's!!",
  'date': '2015-11-06 19:55:14'},
 {'review_id': 'kl_5j4I6rcEtAamrV6-RAg',
  'user_id': 'lHNwkOPku_KWjJKH6KYw7g',
  'business_id': 'oz882XuZCxajKo64Opgq_Q',
  'stars': 1.0,
  'useful': 2,
  'funny': 1,
  'cool': 0,
  'text': '03/20/2011\n\nA CATERPILAR IN MY VEGETABLES !!!\nHORRIBLE.\n\nThe manager only made a discount of $5. When I asked her if it was normal to pay for a plate with a big chenille in it, she only made our sodas for free!\nWhat a SHAME!',
  'date': '2011-03-20 23:38:30'}]
        
