# Databricks notebook source
# MAGIC %md
# MAGIC # Reading in the data
# MAGIC
# MAGIC Header: brewery_id,brewery_name,review_time,review_overall,review_aroma,review_appearance,review_profilename,beer_style,review_palate,review_taste,beer_name,beer_abv,beer_beerid

# COMMAND ----------

df = spark.read.load("/FileStore/tables/beer_reviews.csv", format= "csv", sep=",", inferSchema="true", header="true")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC I only want to deel with overall reviews of the data, so I'm selecting the profile names, beer name, and overall review.

# COMMAND ----------

overall_reviews_df = df['review_profilename', 'beer_beerid', 'review_overall']
overall_reviews_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Here I just finding some helpful statistics about the data, as described by the comments.
# MAGIC
# MAGIC Note that beer name and beer id are different, so we can't identify the beers by name. 

# COMMAND ----------

# Number of reviews total
print("Total reviews: ", df.count())

# Number of unique reviwers
print("Total Reviewers: ", df.select("review_profilename").distinct().count())

# Number of unique beers, checking that against beer ids to make sure no same names
print("Unique beer names: ", df.select("beer_name").distinct().count())
print("Unique beers: ", df.select("beer_beerid").distinct().count())

# Number of reviews per reviwer
df.select(["beer_beerid", "review_profilename"]).groupby("review_profilename").count().show()

# Number of reviews per beer, only shows beer id
df.select(["beer_beerid", "review_profilename"]).groupby("beer_beerid").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Splitting dateset up
# MAGIC I am skipping a validation set at this time, will be adding later.
# MAGIC
# MAGIC **Note**: 'cold start' problem is possible here, testing set may have beers or reviewers unseen in training set, future attempts at recommender systems with this dataset should prevent/reduce this with the inclusion of more 
# MAGIC data such as type of beers and breweries. 
# MAGIC

# COMMAND ----------

test, train = overall_reviews_df.randomSplit([.8, .2])

# COMMAND ----------

# MAGIC %md
# MAGIC # Stochastic Gradient Descent 

# COMMAND ----------


