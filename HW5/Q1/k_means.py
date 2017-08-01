import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import CV as cv

#lower bound of distance between initial centroids
separation = 9
lambda_coe = 0.00005

def initialpts(num_cluster,training):
    matrix = np.zeros((num_cluster,training.shape[1]))
    centroid_idx = np.random.randint(0,training.shape[0])
    matrix[0,:] = training[centroid_idx,:]
    for num_added in range(1,num_cluster):
        continuing = 1
        while continuing == 1:
            continuing = 0
            centroid_idx = np.random.randint(0,training.shape[0])
            centroid = training[centroid_idx,:]
            temp = np.sum(np.abs(np.subtract(matrix,centroid)),axis = 1)
            for i in temp:
                if i < separation:
                    continuing = 1
                    break
        matrix[num_added,:] = centroid
        
    return matrix
    
def L1norm(a):
    return np.sum(np.abs(a))

def distance(a,b):
    #return np.linalg.norm(a-b)
    return L1norm(a - b)

num_genre = 20
num_cluster = 2
iteration = 200

#read training data
df_user_cate = pd.read_csv("user_category.csv")
df_user_cate_trim = df_user_cate.drop('user_id',axis=1)

#read labels
df_labels = pd.read_csv("labels.csv")['labels']

#split data into training and validation
total_data = df_user_cate_trim.as_matrix()
    
#k-means
#initilizize centroids
centroids = initialpts(num_cluster,total_data)


labels = np.zeros(total_data.shape[0])
#k-means
for iter in range(0,iteration):
    #print(iter)
    temp_labels = np.zeros(total_data.shape[0],dtype=np.int32)
    #calculate which cluster belongs to
    for user in range(0,total_data.shape[0]):
        user_vec = total_data[user,:]
        min_label = -1
        min_dist = np.inf
        for cluster in range(0,num_cluster):
            dist = distance(user_vec, centroids[cluster,:])
            if dist < min_dist:
                min_label = cluster
                min_dist = dist
                
        temp_labels[user] = min_label
        
    #if stable, leave
    if np.array_equal(labels, temp_labels):
        break
    else:
        labels = temp_labels

        
    #renew centroids
    temp_centroids = np.zeros((num_cluster,total_data.shape[1]))
    centroids_count = np.zeros(num_cluster)
    #sum up point coordinates
    for user in range(0,total_data.shape[0]):
        temp_centroids[labels[user],:] += total_data[user,:]
        centroids_count[labels[user]] += 1
    
    for cluster_idx in range(0,num_cluster):
        centroids[cluster_idx,:] = temp_centroids[cluster_idx,:] / centroids_count[cluster_idx]
            
df_cluster = pd.DataFrame(0,index=np.arange(0,df_user_cate.shape[0]),columns=['user_id','cluster_id'])
df_cluster['user_id'] = df_user_cate['user_id']
df_cluster['cluster_id'] = labels
df_cluster.to_csv("output_cluster.csv",index=False)
