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

all_beerids = df.select("beer_beerid").distinct().sort('beer_beerid')

all_beerids = all_beerids.rdd.zipWithIndex()
# return back to dataframe
all_beerids = all_beerids.toDF()


all_beerids.show()

# COMMAND ----------

all_usernames.createOrReplaceTempView("profile_ids")
df.createOrReplaceTempView("all_data")

spark.sql("DESCRIBE TABLE all_data;").show()
spark.sql("DESCRIBE TABLE profile_ids;").show()

replaced_names = spark.sql("""
    SELECT * 
    FROM all_data 
    JOIN profile_ids 
    ON all_data.review_profilename = profile_ids._1.review_profilename
""")

mod_df = replaced_names.drop("_1").withColumnRenamed("_2", "review_userid") 
mod_df.show()


# COMMAND ----------

mod_df.printSchema()

# COMMAND ----------

all_beerids.createOrReplaceTempView("beer_ids")
mod_df.createOrReplaceTempView("mod_data")

spark.sql("DESCRIBE TABLE mod_data;").show()
spark.sql("DESCRIBE TABLE beer_ids;").show()

replaced_beerids = spark.sql("""
    SELECT * 
    FROM mod_data 
    JOIN beer_ids 
    ON mod_data.beer_beerid = beer_ids._1.beer_beerid
""")

mod_df = replaced_beerids.drop("_1").withColumnRenamed("_2", "mod_beerid") 
mod_df.printSchema()

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

# Checking that the number of beer ids remained the same
print("Total number of unique beers: ", mod_df.select("beer_beerid").distinct().count())
print("Total number of modified beer ids: ", mod_df.select("mod_beerid").distinct().count())



# COMMAND ----------

from pyspark.sql.functions import min, max

#Checking the range of beer ids, need them to be 0-max # of beers
print(mod_df.agg(min('mod_beerid'), max('mod_beerid')).collect()[0])


# COMMAND ----------

# MAGIC %md
# MAGIC **If you're checking outputs might notice a discrepency between distinct count of modified beer ids and the max id. Idk what thats about**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Come back too!
# MAGIC Somehow while adding id's, several reviews and a reviewer has gone missing. Trying to track down what went wrong here.
# MAGIC
# MAGIC **2/11 Note: Profile numbers line up now**

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

overall_reviews_df = mod_df['review_userid', 'mod_beerid', 'review_overall']
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
item_n = train.select("mod_beerid").distinct().count()

print("Matrix: ", user_n, " x ", item_n)


# COMMAND ----------

# MAGIC %md
# MAGIC Though about converting to coordinate matrix but realized the way I was implementing the model need custom tranformations, which are easier to apply to an RDD

# COMMAND ----------

"""
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

matrix_entries = overall_reviews_df.rdd.map(lambda x: MatrixEntry(x["review_userid"], x["mod_beerid"], x["review_overall"]))

coord_matrix = CoordinateMatrix(matrix_entries)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # The Model
# MAGIC
# MAGIC Using stacastic gradient descent to minimize errors on predicted U and I feature vectors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC ratings - a RDD with columns: user index, item index, review
# MAGIC   With all indices starting at 0 and consecutive
# MAGIC ## Hyperparameters
# MAGIC n_factors - the number of factors used in the matrix factorization
# MAGIC
# MAGIC l_rate - the step size during gradient descent
# MAGIC
# MAGIC alpha - regularization parameter
# MAGIC
# MAGIC n_iter - the number of iterations
# MAGIC ## Methods
# MAGIC initalize(): Here I'm initalizing the user and beer embeddings with just random normal distribution, there are other methods for initalization though. 

# COMMAND ----------

# Libraries involved in the model
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC **Note: Need to make sure id's work out, also parallelizing perdict task!!**

# COMMAND ----------

class MatrixFactorization1():
  def __init__(self, ratings, n_users, n_items, n_factors=100, l_rate=0.01, alpha=0.01, n_iter=100):
    self.ratings = ratings
    self.n_factors = n_factors
    self.l_rate = l_rate
    self.alpha = alpha
    self.n_iter = n_iter

    self.n_users = n_users
    self.n_items = n_items

  def initalize(self, ):
    self.user_vecs = self.createEmbeddings(n_users)
    self.item_vecs = self.createEmbeddings(n_items)

    self.evaluate(0)
  
  def evaluate(self, epoch):
    total_sq_err = 0 
    predictions = self.predict()
    sq_err = self.ratings.map(lambda review: review[2]).zip(predictions).map(lambda pair: (pair[0] - pair[1])**2)
    total_sq_errs = sq_err.sum()
    mse = total_sq_errs / self.ratings.count()

    print(f"---> Epoch {epoch}")
    print("MSE: ", mse)
  
  def predict(self, ):
    predictions = self.ratings.map(lambda review: numpy.dot(self.user_vecs[review[0]], self.item_vecs[review[1]]))
    return predictions
  
  # Will want to split into smaller functions! And probably clean up 
  def update(self, error):
    seqFunc = (lambda x, y: x + y)
    combFunc = (lambda x, y: x + y)

    # review[1] = error of the review 
    # review[0][x] = (user id, item id, real rating)
    user_vec_gradients = self.l_rate * (self.ratings.zip(error).map(lambda review: review[1] * self.item_vecs[review[0][1]] - self.alpha * self.user_vecs[review[0][0]]))
    
    # Combining the gradients for each vector
    agg_u_grads = self.ratings.map(lambda review: review[0]).zip(user_vec_gradients).aggregateByKey(0, seqFunc, combFunc)

    updated_u = self.user_vecs.zip(agg_u_grads).map(lambda user_vecs: user_vecs[0] + user_vecs[1][1])

    # Will probably need to check closer
    item_vec_gradients = self.l_rate * (self.ratings.zip(error).map(lambda review: review[1] * self.user_vecs[review[0][0]] - self.alpha * self.item_vecs[review[0][1]]))
    
    # Combining the gradients for each vector
    agg_i_grads = self.ratings.map(lambda review: review[1]).zip(item_vec_gradients).aggregateByKey(0, seqFunc, combFunc)
    
    updated_i = self.item_vecs.zip(agg_i_grads).map(lambda item_vecs: item_vecs[0] + item_vecs[1][1])

  
  def fit(self, ):
    self.initalize()
    for epoch in range(0, self.n_iter):
      prediction = self.predict()
      err = self.ratings.map(lambda review: review[2]).zip(predictions).map(lambda pair: (pair[0] - pair[1]))
      self.update(err)
      self.evaluate(epoch)

    
  def createEmbeddings(self, n_rows):
    embedding_rdd = sc.parallelize([1] * n_rows * self.n_factors).map(lambda x: np.random.normal(scale = 1/np.sqrt(self.n_factors), size = self.n_factors))


# COMMAND ----------

# MAGIC %md
# MAGIC # Testing

# COMMAND ----------

import numpy as np
from pyspark.mllib.linalg.distributed import DenseMatrix

data = [(1, 1), (1, 1), (2, 1), (3, 1), (3, 1)]
rdd = sc.parallelize(data)

keys = [1, 0, 3]
key_rdd = sc.parallelize(keys)

print(rdd.aggregateByKey(0, lambda x,y: x + y, lambda x,y: x + y).take(3))

rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 2)])
seqFunc = (lambda x, y: x + y)
combFunc = (lambda x, y: x + y)
print(sorted(rdd.aggregateByKey(0, seqFunc, combFunc).collect()))

# COMMAND ----------

spark = SparkSession.builder.appName("CustomClassExample").getOrCreate()
    
# Using the custom class in a map transformation
result_rdd = overall_reviews_df.map(lambda x: MatrixFactorization(x).process())
    
print(result_rdd.collect())
    
spark.stop()
