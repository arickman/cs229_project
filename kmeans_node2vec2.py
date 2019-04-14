from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#import snap
import sys
import pdb

color1 = (0,0,0.5)
color2 = (0,0.5,0.1)
color3 = (0.9,0.4,0.3)
color4 = (0.5,0.5,0.1)
color5 = (0.2,0.2,0.2)
color6 =(0,0.5,0.5)
colorset = [color1,color2,color3,color4,color5,color6]

def readEmbedding(infile):
    nodes = 0
    dim = 0
    features = {}
    with open(infile,'r') as f:
        for line in f:
            dat = line.split()
            
            # 1st line
            if nodes == 0 and dim ==0:
                nodes = dat[0]
                dim = dat[1]
            else:
                floatarray = []
                for i in range(1,len(dat)):
                    floatarray.append(float(dat[i]))
                name = int(dat[0])
                features[name] = floatarray
    return features

def clusterFeatures(features):
    X = np.array(list(features.values()))
    db = DBSCAN(eps=.3, min_samples=3).fit(X)
    return db.labels_,None

    #Now we have a new set of labels for the given list of features, so update features by taking out anomalies

#def removeAnoms(features, labels):
    
def euclidDist(a,b):
    A = np.array(a)
    B = np.array(b)
    C = B-A
    return np.sqrt(C.T.dot(C))


def findDistances(features,labels,centers):
    keys = features.keys()
    dists = {}
    radii = {}
    size = {}
    for j in range(0,len(centers)):
        radii[j] = 0
        size[j] = 0
    for i in range(0,len(keys)):
        k = keys[i]
        point = features[k]
        center = centers[labels[i]]
#        print "K: ",k," at ",point," from c=",center
        d = euclidDist(point,center)
        radii[labels[i]]+=d
        size[labels[i]]+=1
        print("k=",k," is ",d," from center ",labels[i]) 
        dists[k] = d
    for j in range(0,len(centers)):
        if size[j]>0:
            radii[j] = radii[j]/size[j]
    return dists,radii,size

def printClusterInfo(radii,sizes):
    for k in radii.keys():
        print( "cluster k: ",sizes[k]," nodes, ",radii[k]," avg dist")


def outputAnomalies(features,distances,outfile):
    of = open(outfile,'w')
    truthfile = str(args[5])    
    for k in distances.keys():
        if distances[k]> 0.21:
            of.write(str(k))
            of.write("\n")
    of.close()

def computeModularities(nodes,labels,graphfile,nCenters):
    Graph = snap.LoadEdgeList(snap.PUNGraph, graphfile)
    print( "graph read: ",Graph.GetNodes()," nodes with ",Graph.GetEdges()," edges.")
    for c in range(0,nCenters):
        Nodes = snap.TIntV()
        for i in range(0,len(labels)):
            if labels[i] == c:
                Nodes.Add(nodes[i])
        mc = snap.GetModularity(Graph, Nodes)
        print( "c = ",c," modularity: ",mc)


def readTruths(infile):
    labels = {}
    with open(infile,'r') as f:
        for line in f:
            dat = line.split()
            labels[dat[0]] = dat[1]
    return labels

args  = sys.argv
print(args)
infile = str(args[1])
outfile = str(args[2])
plotdir = str(args[3])
graphfile=str(args[4])
truthfile = str(args[5])


features = readEmbedding(infile)
truths = readTruths(truthfile)

X = np.array(list(features.values()))

#pdb.set_trace()
#Run DBSCAN
#db = DBSCAN(eps=.3, min_samples=5).fit(X)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.labels_
#labels = db.labels_
print('Num labels: ', len(set(labels)))


#Reduce dimensionality to 3 for visualization
pca = PCA(n_components = 3).fit(X)
X_transform = pca.transform(X)

#Create plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
fig.set_size_inches(24,24)
color1 = (0,0,0.5)
color2 = (0,0.5,0.1)
color3 = (0.9,0.4,0.3)
color4 = (0.5,0.5,0.1)
color5 = (0.2,0.2,0.2)
color6 =(0,0.5,0.5)
color7 =(0.3,0.4,0.9)
color8 = (0.1, 0.2, 0.8)
color9 = (0.8, 0.4,0.8)
color10 = (0, 0, 0)
colorset = [color1,color2,color3,color4,color5,color6,color7,color8,color9]

keys = list(features.keys())

xs = X_transform[:,0]
ys = X_transform[:,1]
zs = X_transform[:,2]

ax.scatter(xs, ys, zs, color = 'grey', s = 1)

for i in range(0,len(keys)):
    name = truths[str(keys[i])]
    ax.text(xs[i],ys[i],zs[i],name,size =6,zorder =1,color = colorset[labels[i]])

plt.show()

