#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd


# In[47]:


df = pd.read_csv('loan_data_2007_2014.csv')


# Data Understanding

# In[48]:


df.head()


# In[49]:


df.shape


# In[50]:


df.columns


# In[96]:


import seaborn as sns
sns.heatmap(df.corr());


# Data Preprocessing

# > Handling Missing Value

# In[61]:


import missingno as msno

def cekNull(df):
    missing = []
    persen = []

    for i in df.columns:
        missing.append(df[i].isnull().sum())
        persen.append(df[i].isnull().sum()/len(df[i])*100)

    proporsi_hilang = pd.DataFrame({
            'kolom' : df.columns,
            'missing' : missing,
            'persen_missing' : persen
    })
    
    msno.matrix(df)
    
    delete_columns = proporsi_hilang.drop(proporsi_hilang[proporsi_hilang.persen_missing < 50].index)
    #print(delete_columns)
    
    impute_columns = proporsi_hilang.drop(proporsi_hilang[(proporsi_hilang.persen_missing >= 50) | (proporsi_hilang.persen_missing <= 0)].index)
    #print(impute_columns)

    return delete_columns, impute_columns


# In[62]:


delete_columns, impute_columns = cekNull(df)


# In[54]:


delete_columns


# In[55]:


df[delete_columns['kolom']]


# In[56]:


df = df.drop(delete_columns['kolom'], axis=1)


# In[63]:


impute_columns


# In[64]:


df[impute_columns['kolom']]


# In[68]:


df_obj = df[impute_columns['kolom']].select_dtypes(include = 'object')
df_obj.fillna('Tidak Diketahui', inplace=True)
df_obj


# In[70]:


df[df_obj.columns] = df_obj
df


# In[75]:


df_num = df[impute_columns['kolom']].select_dtypes(include = 'number')
df_num.fillna(0, inplace=True)
df_num


# In[76]:


df[df_num.columns] = df_num
df


# In[78]:


delete_columns, impute_columns = cekNull(df)


# > Ordinal Encoder

# In[85]:


df_obj = df.select_dtypes(include = 'object')


# In[88]:


from sklearn.preprocessing import OrdinalEncoder
import numpy as np

oe = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=np.nan)
X_obj_transform = oe.fit_transform(df_obj)

X_obj_transform = pd.DataFrame(X_obj_transform)
X_obj_transform.columns = df_obj.columns
X_obj_transform


# In[89]:


df_num = df.select_dtypes(include = 'number')


# In[94]:


df_concat = pd.concat([X_obj_transform,df_num], axis=1)
df_concat


# > Scaler

# In[95]:


from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
df_transform = minmax_scaler.fit_transform(df_concat)

df_transform = pd.DataFrame(df_transform)
df_transform.columns = df_concat.columns
df_transform


# Make label with clustering

# > PCA

# In[97]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(df_transform)


# In[99]:


import matplotlib.pyplot as plt

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[100]:


np.cumsum(pca.explained_variance_ratio_)


# In[101]:


n_components=2
pca_final = PCA(n_components=n_components)
pca_Data = pca_final.fit_transform(df_transform)


# In[109]:


a = {'PC1':pca_final.components_[0], 'PC2':pca_final.components_[1]}
b = pd.DataFrame(a)
b.index = df_transform.columns
b


# In[102]:


plt.scatter(pca_Data[:,0], pca_Data[:,1])


# In[103]:


for i in np.arange(n_components):
    index =  np.argmax(np.absolute(pca_final.get_covariance()[i]))
    max_cov = pca_final.get_covariance()[i][index]
    column = df_transform.columns[index]
    print("Principal Component", i+1, "maximum covariance :", "{:.2f}".format(max_cov), "from column", column)


# > KMeans

# In[105]:


from sklearn.cluster import KMeans

sse = {}
n_clust = np.arange(2,11)

for i in n_clust:
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(pca_Data)
    sse[i] = kmeans.inertia_


# In[106]:


plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum-of-Squares")
plt.show()


# In[125]:


kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(pca_Data)


# In[112]:


plt.scatter(pca_Data[:,0], pca_Data[:,1],
            c = KMeans(n_clusters = 4).fit_predict(pca_Data),
            cmap = plt.cm.summer)
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show() 


# In[121]:


centroids = kmeans.cluster_centers_
for i in np.arange(len(centroids)):
    print("Center of Cluster", i+1, ":", centroids[i])


# In[129]:


df_transform['label'] = kmeans.labels_


# In[130]:


df_transform


# Classification with Random Forest

# > Split Data

# In[131]:


y = df_transform['label']
x = df_transform.drop(['label'], axis=1)


# In[132]:


# Data Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

print("Jumlah Data Train : {0}".format(len(X_train)))
print("Jumlah Data Test : {0}".format(len(X_test)))


# > Make Modelling

# In[133]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# > Make Evaluation Result

# In[134]:


y_pred_test = rf.predict(X_test)


# In[136]:


from sklearn.metrics import classification_report

print('Result of Testing Datast with optimized parameter')
print(classification_report(y_test, y_pred_test, labels=[3,2,1,0]))


# In[ ]:




