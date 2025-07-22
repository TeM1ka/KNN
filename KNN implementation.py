from scipy import stats

class KNN:

  def __init__(self,k=3,metric='euclidean',p=None):
    self.k = k
    self.metric = metric
    self.p = p

  def euclidean(self,v1,v2):
    return np.sqrt(np.sum((v1-v2)**2))

  def manhattan(self,v1,v2):
    return np.sum(np.abs(v1-v2))

  def minkowski(self,v1,v2,p=2):
    return np.sum(np.abs(v1-v2)**p)**(1/p)

  def fit(self,X_train,y_train):
    self.X_train = X_train
    self.y_train = y_train
  
  def predict(self,X_test):
    preds = []
    for test_row in X_test:
      nearest_neighbours = self.get_neighbours(test_row)
      majority = stats.mode(nearest_neighbours)[0]
      preds.append(majority)
    return np.array(preds)

  def get_neighbours(self,test_row):
    distances = []

    for (train_row,train_class) in zip(self.X_train,self.y_train):
      if self.metric == 'euclidean':
        distance = self.euclidean(train_row,test_row)
      elif self.metric == 'manhattan':
        distance = self.manhattan(train_row,test_row)
      elif self.metric == 'minkowski':
        distance = self.minkowski(train_row,test_row,self.p)
      else:
        raise NameError("Name error")
      distances.append((distance,train_class))

    distances.sort(key=lambda x: x[0])
    neighbours = list()
    for i in range(self.k):
      neighbours.append(distances[i][1])
    
    return neighbours
  


