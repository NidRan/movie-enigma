
# coding: utf-8

# In[3]:


from pyspark.mllib.recommendation import ALS
from numpy import array
rating_raw = sc.textFile("data/ratings.csv")
rating_raw_header=rating_raw.take(1)[0]
print rating_raw_header


# In[7]:


rating_=rating_raw.filter(lambda line: line!=rating_raw_header)    .map(lambda line: line.split(",")).map(lambda tokens:(tokens[0], tokens[1], tokens[2])).cache()
     


# In[8]:


print rating_.take(3)


# In[9]:


movies_raw=sc.textFile("data/movies.csv")
movies_raw_header=movies_raw.take(1)[0]
print movies_raw_header


# In[11]:


movies_=movies_raw.filter(lambda line: line!=movies_raw_header)    .map(lambda line: line.split(",")).map(lambda tokens:(tokens[0], tokens[1], tokens[2])).cache()
print movies_.take(3)


# In[13]:


training_RDD, validation_RDD, test_RDD = rating_.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


# In[14]:


from pyspark.mllib.recommendation import ALS
import math


# In[15]:


seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02


# In[16]:


min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank


# In[17]:


predictions.take(3)


# In[18]:


rates_and_preds.take(3)


# In[19]:


model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print 'For testing data the RMSE is %s' % (error)


# In[ ]:




