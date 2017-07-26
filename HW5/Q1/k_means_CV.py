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

def schwarz_cri(num_cluster,data):
    return lambda_coe * num_cluster * data.shape[1] * np.log(data.shape[0])

def entropy(labels,num_labels,cluster,num_cluster):
    num_data = len(labels)
    cluster_label = np.zeros((num_cluster,num_labels))
    for i in range(0,num_data):
        cluster_label[cluster[i],labels[i]] += 1
    
    for i in range(0,cluster_label.shape[0]):
        sum = np.sum(cluster_label[i,])
        for j in range(0,cluster_label.shape[1]):
            if cluster_label[i,j] != 0:
                proba = cluster_label[i,j] / sum
                cluster_label[i,j] = proba * np.log(proba) * -1
    return np.sum(cluster_label) / num_cluster

def purity(labels,num_labels,cluster,num_cluster):
    num_data = len(labels)
    cluster_label = np.zeros((num_cluster,num_labels))
    for i in range(0,num_data):
        cluster_label[cluster[i],labels[i]] += 1
        
    max = np.sum(cluster_label.max(axis=1))
    return max / np.sum(cluster_label) 
    

num_genre = 20
num_training = 1000
num_cluster_set = np.arange(2,15)
iteration = 200

#read training data
df_user_cate = pd.read_csv("user_category.csv")
df_user_cate_trim = df_user_cate.drop('user_id',axis=1)

#read labels
df_labels = pd.read_csv("labels.csv")['labels']

#split data into training and validation
total_data = df_user_cate_trim.iloc[:num_training].as_matrix()
total_labels = df_labels.iloc[:num_training].values

kfold = 10

output = open("k_means_CV.txt",'w')

schwarz_criterion = np.zeros(len(num_cluster_set))
num_cluster_idx = 0
for num_cluster in num_cluster_set:
    
    #k-fold cross validation
    indice = cv.KFold(total_data, kfold)
    shuffled_data, shuffled_labels = cv.shuffle(total_data,total_labels)
    
    #shuffled_data = total_data
    #shuffled_labels = total_labels
    
    validation_error_set = np.zeros(kfold)
    
    for kfold_idx in range(0,kfold):
        valid_indice = np.arange(indice[kfold_idx],indice[kfold_idx+1])
        train_indice = np.delete(np.arange(0,total_data.shape[0]),valid_indice)
        
        data_train = shuffled_data[train_indice]
        data_validation = shuffled_data[valid_indice]
        
        target_validation = shuffled_labels[valid_indice]
    
        #initilizize centroids
        centroids = initialpts(num_cluster,data_train)

        #print(centroids)
        labels = np.zeros(data_train.shape[0])
        #k-means
        for iter in range(0,iteration):
            #print(iter)
            temp_labels = np.zeros(data_train.shape[0],dtype=np.int32)
            #calculate which cluster belongs to
            for user in range(0,data_train.shape[0]):
                user_vec = data_train[user,:]
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
            temp_centroids = np.zeros((num_cluster,data_train.shape[1]))
            centroids_count = np.zeros(num_cluster)
            #sum up point coordinates
            for user in range(0,data_train.shape[0]):
                temp_centroids[labels[user],:] += data_train[user,:]
                centroids_count[labels[user]] += 1
            
            for cluster_idx in range(0,num_cluster):
                centroids[cluster_idx,:] = temp_centroids[cluster_idx,:] / centroids_count[cluster_idx]
        
        #validation        
        validation_error = 0
        validation_labels = np.zeros(data_validation.shape[0],dtype=np.int32)
        for user in range(0,data_validation.shape[0]):
            user_vec = data_validation[user,:]
            
            min_label = -1
            min_dist = np.inf
            for cluster in range(0,num_cluster):
                dist = distance(user_vec, centroids[cluster,:])
                if dist < min_dist:
                    min_label = cluster
                    min_dist = dist
            validation_labels[user] = min_label
        
        #validation_error = entropy(target_validation,20,validation_labels,num_cluster)
        validation_error = purity(target_validation,20,validation_labels,num_cluster)
        validation_error_set[kfold_idx] = validation_error
    
    schwarz_criterion[num_cluster_idx] = 1 - np.mean(validation_error_set)#+schwarz_cri(num_cluster,data_validation)
    #schwarz_criterion[num_cluster_idx] = np.mean(validation_error_set)
    #output error
    print("# of clusters = %d, Schwarz Cri = %f" %(num_cluster,schwarz_criterion[num_cluster_idx]))
    output.write("# of clusters = %d, Schwarz Cri. = %f" %(num_cluster,schwarz_criterion[num_cluster_idx]))
    num_cluster_idx += 1

fig, ax = plt.subplots()
plt.plot(num_cluster_set, schwarz_criterion,label='Schwarz Criterion')
plt.legend(loc='best')
plt.xlabel('Number of Clusters')
plt.ylabel('Schwarz Criterion(1/Purity+lambda*(#parameters)*(#data)')
plt.title('Number of Clusters vs. Schwarz Criterion')
 
plt.savefig("k-mean_CV.png")
