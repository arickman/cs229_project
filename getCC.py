from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snap
import sys



args = sys.argv
graphfile = str(args[1])
nodefile = str(args[4])
outfilename = str(args[2])
nodefilename = str(args[3])

Graph = snap.LoadEdgeList(snap.PUNGraph, graphfile)
print("nodes: "+str(Graph.GetNodes())+" "+str(Graph.GetEdges()))
MxWcc = snap.GetMxWcc(Graph)
outfile = open(outfilename,'w')
for EI in MxWcc.Edges():
    print("Edge")
    outfile.write(str(EI.GetSrcNId()))
    outfile.write(" ")
    outfile.write(str(EI.GetDstNId()))
    outfile.write("\n")

outfile.close()
nodelabels = {}
with open(nodefile,'r') as f:
    for line in f:
        print(line)
        dat = line.split()
        nodelabels[dat[0]] = dat[1]



nodefile = open(nodefilename,'w')
for N in MxWcc.Nodes():
    nodefile.write(str(N.GetId()))
    print(N.GetId())
    nodefile.write(" ")
    nodefile.write(str(nodelabels[str(N.GetId())]))
    nodefile.write("\n")
nodefile.close()
