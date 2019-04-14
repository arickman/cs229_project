from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snap
import sys



args = sys.argv
graphfile = str(args[1])

Graph = snap.LoadEdgeList(snap.PUNGraph, graphfile)
print("nodes: "+str(Graph.GetNodes())+" edges: "+str(Graph.GetEdges()))
