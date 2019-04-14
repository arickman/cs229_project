from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snap
import sys

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


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def remove_anoms(features, labels, outfile):
	index_list = []
	new_features = features
	feat_vec = new_features.values()
	for i in range(0,len(feat_vec)):
		#if labels[i] != -1: return_vec.append(feat_vec[i])
		if labels[i] == -1: index_list.append(i)
	np.array(index_list)
	of = open(outfile,'w')
	for k in index_list:
		of.write(str(k))
		of.write("\n")
	#np.save(outfile, index_list)
	keys = new_features.keys()
	for index in index_list:
		del new_features[keys[index]]
	return new_features


def plotFeatures(features,axes,outfile,labels,centers,truths):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	fig.set_size_inches(24,24)

	keys = features.keys()
	if len(labels)!=len(keys):
		labels = [0]*len(keys)
	xs = [features[keys[i]][axes[0]] for i in range(0,len(keys))]
	ys = [features[keys[i]][axes[1]] for i in range(0,len(keys))]
	zs = [features[keys[i]][axes[2]] for i in range(0,len(keys))]

	ax.scatter(xs, ys, zs,color='grey')
	ax.set_xlabel(axes[0])
	ax.set_ylabel(axes[1])
	ax.set_zlabel(axes[2])
	for i in range(0,len(keys)):
		name = unicode(truths[str(keys[i])], "utf-8")
               #  # str(keys[i])->truths[keys[i])
               # if "General" in name or "general" in name:
               #     ax.text(xs[i],ys[i],zs[i],name,size =6,zorder =1,color='r')		
               # # else:
		ax.text(xs[i],ys[i],zs[i],name,size =6,zorder =1,color =colorset[labels[i]])		
    
	xc = [centers[i][axes[0]] for i in range(0,len(centers))]
	yc = [centers[i][axes[1]] for i in range(0,len(centers))]
	zc = [centers[i][axes[2]] for i in range(0,len(centers))]
	print("xc,yc,zc: ",xc,yc,zc)
	ax.scatter(xc,yc,zc,color = 'r')

	for i in range(0,len(centers)):
		ci = centers[i]	
		ax.text(ci[axes[0]],ci[axes[1]],ci[axes[2]],"c"+str(i),color=(0.5,0,0))
		if len(outfile)>0:
			plt.savefig(outfile)
			plt.close()
		else:
			plt.show()


def clusterFeatures(features):
	X = np.array(features.values())
	kmeans = 1
	if kmeans ==1:
		kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
		return kmeans.labels_,kmeans.cluster_centers_

	else:
		db = DBSCAN(eps=0.3, min_samples=8).fit(X)
		return db.labels_

	#Now we have a new set of labels for the given list of features, so update features by taking out anomalies
	


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
#		print "K: ",k," at ",point," from c=",center
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
#print(args)
infile = str(args[1])
outfile = str(args[2])
plotdir = str(args[3])
graphfile=str(args[4])
truthfile = str(args[5])

features = readEmbedding(infile)
truths = readTruths(truthfile)
labels,centers = clusterFeatures(features)
labels = clusterFeatures(features)
#print( "labels: ",labels," features: ",features)



# for i in range(0,10):
# 	for j in range(0,10):
# 		for k in range(0,10):
			#print( "i j k: ",i,j,k)
			#plotFeatures(features,[i,j,k],plotdir+"/axesset_"+str(i)+str(j)+str(k)+".png",labels,centers,truths)


distances,radii,sizes = findDistances(features,labels,centers)
#printClusterInfo(radii,sizes)
outputAnomalies(features,distances,outfile)  
#removed_anoms_dict = remove_anoms(features, labels, outfile) 

#computeModularities(features.keys(),labels,graphfile,len(centers))




