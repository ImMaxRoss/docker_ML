import pandas as pd
import sqlite3

from bs4 import BeautifulSoup

from nltk.tokenize import RegexpTokenizer

import warnings
warnings.filterwarnings('ignore')


conn = sqlite3.connect('/app/data/database.db')

df_reviews = pd.read_json('./data/Movies_and_TV.json.gz', compression='gzip', lines=True)
meta_df = pd.read_json('./data/meta_Movies_and_TV.json.gz', compression='gzip', lines=True)


meta_df.rename(columns={'category': 'genre', 'brand': 'starring', 'asin': 'movie_id'}, inplace=True)
meta_df.drop_duplicates(subset='movie_id', inplace=True)
meta_df.dropna(subset='title', inplace=True)
meta_df = meta_df[meta_df['main_cat'] == 'Movies & TV']
meta_df.drop(['tech1', 'fit', 'tech2', 'similar_item',
         'date', 'price', 'imageURL', 'imageURLHighRes',
        'also_buy', 'also_view', 'feature', 'rank',
          'main_cat', 'details'], axis=1, inplace=True)


def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text(strip=True)

def clean_html(title):
    # Remove HTML entities and strip leading/trailing whitespaces
    cleaned_title = remove_html_tags(title).strip()
    # Remove extra information by splitting on " / " and taking the first part of the split
    cleaned_title = cleaned_title.split(" / ", 1)[0]
    return cleaned_title

meta_df['title'] = meta_df['title'].apply(remove_html_tags)
meta_df['title'] = meta_df['title'].apply(clean_html)


convert = ['.', '\n', '-', '--', 'Na',
           'BRIDGESTONE MULTIMEDIA', '*', 'none',
           'na', 'N/a', 'VARIOUS', 'Artist Not Provided',
           'Sinister Cinema', 'Learn more', 'Various', 'various',
           'The Ambient Collection', 'Animation', 'Standard Deviants',
          'Animated']

meta_df['starring'] = meta_df['starring'].apply(lambda x: 'Various Artists' if isinstance(x, str) and (x in convert or '\n' in x) else x)
meta_df['starring'].fillna('Various Artists', inplace=True)
meta_df['starring'].replace({'': 'Various Artists'}, inplace=True)


#removing 'Movies & TV' from the beginning of each genre list
meta_df['genre'] = [x[1:] if len(x) > 1 and x[0] == 'Movies & TV' else x for x in meta_df['genre']]

#removing 'Exercise & Fitness' videos
meta_df = meta_df[~meta_df['genre'].apply(lambda x: 'Exercise & Fitness' in x)]

#exctracting Art House & International and the language origin of the film
meta_df.loc[meta_df['genre'].apply(lambda x: isinstance(x, list) and len(x) > 2 and x[0] == 'Art House & International'), 'genre'] = meta_df['genre'].apply(lambda x: [x[0] + ' ' + x[2]] if len(x) > 2 else x)

#combining Art House with it's language origin
meta_df['genre'] = meta_df['genre'].apply(lambda x: x[:1] + x[2:] if isinstance(x, list) and len(x) > 2 and x[0] == 'Art House & International' and len(x) > 2 else x)

#joining all of the lists so they are now one string value
meta_df['genre'] = meta_df['genre'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

#replacing empy sting values with unknown
meta_df['genre'].replace({'': 'unknown'}, inplace=True)


meta_df['description'] = meta_df['description'].apply(lambda x: x[0] if x else 'unknown')
meta_df['description'] = meta_df['description'].apply(remove_html_tags)
meta_df['description'] = meta_df['description'].apply(clean_html)
meta_df.drop_duplicates(subset='description', inplace=True)



tokenizer = RegexpTokenizer('\w+')


sw = ['genre','for','featured','categories',
      'independently','distributed','for','studio',
     'home', 'warner', 'specials', 'all', 'hbo',
      'titles', 'pictures', 'entertainment' 'blue',
      'ray', 'dvd', 'vhs', 'lionsgate', 'mod',
      'createspace', 'video', 'a', 'e', '20th', 'fox',
      'universal', 'mgm', 'entertainment', 'specials',
      'bbc', 'boxed', 'sets', 'walt', 'general',
      'paramount', 'loaded', 'dvds', 'fully', 'blu',
      'sony', 'studios', 'pbs', 'television', 'dts',
      'miramax', 'history', 'series', 'movies',
      'criterion','collection','century', 'top',
      'sellers', 'first', 'to', 'know', 'disney'
     ]


def tokenize_sw(text):
    
    #converting all letters to lowercase
    text = text.lower()
    
    #tokenizing words so that I can isolate words and remove unecessary labels
    words = tokenizer.tokenize(text)

    #removing unecessary genre labels found in my stopwords list
    words = [word for word in words if word not in sw]
    
    return words


meta_df['genre'] = meta_df['genre'].apply(tokenize_sw)
meta_df['genre'] = meta_df['genre'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
meta_df['genre'].replace({'': 'unknown'}, inplace=True)


remove_genre_str = ['tv', 'special editions'] 

def remove_substrings(s, word_list):
    for word in word_list:
        s = s.replace(word, '')
    return s

mask = ~meta_df['genre'].isin(remove_genre_str)
meta_df = meta_df[mask]
meta_df['genre'] = meta_df['genre'].apply(remove_substrings, word_list=remove_genre_str).str.strip()
meta_df['genre'].replace({'': 'unknown'}, inplace=True)



df_reviews.drop(['image', 'reviewTime', 'reviewerName', 'summary',
              'vote', 'unixReviewTime'], axis=1, inplace=True)

df_reviews.rename(columns={'overall': 'rating', 'asin': 'movie_id',
                          'reviewerID': 'user_id','reviewText':'reviews'}, inplace=True)


df_reviews = df_reviews[df_reviews['verified'] == True]
df_reviews.drop(columns = 'verified', inplace=True)


# Handle non-dictionary types in the 'style' column
rm_format = df_reviews['style'].apply(lambda x: isinstance(x, dict) and 'Format:' in x)
df_reviews = df_reviews[rm_format]

# Continue with extraction and filtering as before
df_reviews['style'] = df_reviews['style'].apply(lambda x: x['Format:'].strip())
format_counts = df_reviews['style'].value_counts()
df_reviews = df_reviews[df_reviews['style'].isin(format_counts[format_counts >= 25000].index)]


# unique movie_ids from the meta dataframe and convert them to a list
all_vid = meta_df['movie_id'].unique().tolist()

# Keep only rows in df_collab where 'movie_id' is in all_vid (for inference)
df_collab = df_reviews[df_reviews['movie_id'].isin(all_vid)]

# removing entries where a user reviewed a movie multiple times in order to 
# preserve the values of the ratings
df_collab.drop_duplicates(subset=['user_id', 'movie_id'], keep='first', inplace=True)

#dropping style as I will be assuming that all formats will be reviewed soely based on their content
df_collab.drop(columns='style', inplace=True)

# Removing user_id's that have less than 4 reviews in order to have collaborative
# .... filtered recommendations for similar users (may want to filter more depending on RMSE)
df_collab = df_collab[df_collab['user_id'].isin(df_collab['user_id'].value_counts()[df_collab['user_id'].value_counts() >= 4].index)]


all_vid2 = df_collab['movie_id'].unique().tolist()
col_meta = meta_df[meta_df['movie_id'].isin(all_vid2)]



revtext_merged_df = pd.merge(df_collab, col_meta, on="movie_id", how="left")
revtext_merged_df.to_sql('revtext_merged_table', conn, if_exists='replace', index=False)
conn.close()