# Databricks notebook source
# MAGIC %md
# MAGIC # Reading in the data
# MAGIC
# MAGIC Header: 
# MAGIC brewery_id,
# MAGIC brewery_name,
# MAGIC review_time,
# MAGIC review_overall,
# MAGIC review_aroma,
# MAGIC review_appearance,
# MAGIC review_profilename,
# MAGIC beer_style,
# MAGIC review_palate,
# MAGIC review_taste,
# MAGIC beer_name,
# MAGIC beer_abv,
# MAGIC beer_beerid

# COMMAND ----------

df = spark.read.load("/FileStore/tables/beer_reviews.csv", format= "csv", sep=",", inferSchema="true", header="true")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Manipulation and Exploration
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Creating unique ids for the review profile names for ease of use and so I can convert the dataframe into a matrix using Spark

# COMMAND ----------

all_usernames = df.select("review_profilename").distinct()

all_usernames = all_usernames.rdd.zipWithIndex()
# return back to dataframe
all_usernames = all_usernames.toDF()


all_usernames.show()

# COMMAND ----------

all_usernames.createOrReplaceTempView("profile_ids")
df.createOrReplaceTempView("all_data")

spark.sql("DESCRIBE TABLE all_data;").show()
spark.sql("DESCRIBE TABLE profile_ids;").show()


# COMMAND ----------

replaced_names = spark.sql("""
    SELECT * 
    FROM all_data 
    JOIN profile_ids 
    ON all_data.review_profilename = profile_ids._1.review_profilename
""")

mod_df = replaced_names.drop("_1").withColumnRenamed("_2", "review_userid") 
mod_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Here I just finding some helpful statistics about the data, as described by the comments.
# MAGIC
# MAGIC The number beer name and beer id are different, so we can't identify the beers by name. 

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

# Checking that the number of unique profile names and unique ids are the same (ensuring I did earlier SQL correctly)
print("Total number of unique users: ", mod_df.select("review_profilename").distinct().count())
print("Total number of unique user ids: ", mod_df.select("review_userid").distinct().count())
print("Total reviews: ", mod_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Come back too!
# MAGIC Somehow while adding id's, several reviews and a reviewer has gone missing. Trying to track down what went wrong here.

# COMMAND ----------

df.createOrReplaceTempView("all_data")
mod_df.createOrReplaceTempView("mod_data")

spark.sql("""
    SELECT DISTINCT all_data.review_profilename
    FROM all_data, mod_data
    WHERE all_data.review_profilename != mod_data.review_profilename
    """).show()

spark.sql("""
  SELECT COUNT(beer_beerid)
  FROM mod_data
  GROUP BY mod_data.review_profilename
  """).show()

# COMMAND ----------

# MAGIC %md
# MAGIC I only want to deal with overall reviews of the data, so I'm selecting the profile ids, beer ids, and overall review.

# COMMAND ----------

overall_reviews_df = mod_df['review_userid', 'beer_beerid', 'review_overall']
overall_reviews_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Splitting dateset up
# MAGIC I am skipping a validation set at this time, will be adding later.
# MAGIC
# MAGIC **Note**: 'cold start' problem is possible here, testing set may have beers or reviewers unseen in training set, future attempts at recommender systems with this dataset should prevent/reduce this with the inclusion of more 
# MAGIC data such as type of beers and breweries. 
# MAGIC

# COMMAND ----------

train, test = overall_reviews_df.randomSplit([.8, .2])

# COMMAND ----------

# MAGIC %md
# MAGIC Setting up inital info about the User-Item matrix

# COMMAND ----------

user_n = train.select("review_userid").distinct().count()
item_n = train.select("beer_beerid").distinct().count()

print("Matrix: ", user_n, " x ", item_n)


# COMMAND ----------

col_ptrs = [0]
row_indices = []
values = []

for row in mod_df:
    row_indices.append(row["review_userid"])
    values.append(row["review_overall"])

    col_ptrs.append(col_ptrs[-1] + 1)

col_ptrs = col_ptrs.tolist()

sparse_matrix = SparseMatrix(num_users, num_items, col_ptrs, row_indices, values)

print("Sparse Matrix:", sparse_matrix)


# COMMAND ----------

# MAGIC %md
# MAGIC # The Model
# MAGIC
# MAGIC Using stacastic gradient descent to minimize errors on predicted U and I feature vectors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC ratings - a sparse matrix
# MAGIC ## Hyperparameters:
# MAGIC n_factors - the number of factors used in the matrix factorization
# MAGIC
# MAGIC l_rate - the step size during gradient descent
# MAGIC
# MAGIC alpha - regularization parameter
# MAGIC
# MAGIC n_iter - the number of iterations

# COMMAND ----------

class MatrixFactorization1():
  def init(self, ratings, n_factors=100, l_rate=0.01, alpha=0.01, n_iter=100):
    self.ratings = ratings
    self.n_factors = n_factors
    self.l_rate = l_rate
    self.alpha = alpha
    self.n_iter = n_iter

    


# COMMAND ----------


