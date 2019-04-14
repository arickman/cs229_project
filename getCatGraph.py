import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snap
import sys



def writeGroup(catGroup,of):
    for i in range(0,len(catGroup)):
        for j in range(i+1,len(catGroup)):
            of.write(str(catGroup[i])+" "+str(catGroup[j])+"\n")
def addEdges(catGroup,G):
    for i in range(0,len(catGroup)):
        for j in range(i+1,len(catGroup)):
            if G.IsEdge(catGroup[i],catGroup[j]):
                e = G.GetEI(catGroup[i],catGroup[j])
                weight = G.GetIntAttrDatE(e,"w")+1
                G.AddIntAttrDatE(e,weight,"w")
            else:
                G.AddEdge(catGroup[i],catGroup[j])
                e = G.GetEI(catGroup[i],catGroup[j])
                G.AddIntAttrDatE(e,1,"w")

def printEdges(G,of):
    for EI in G.Edges():
        w = G.GetIntAttrDatE(EI,"w")
        of.write(str(EI.GetSrcNId())+" "+str(EI.GetDstNId())+" "+str(w)+"\n")

def readInfoFile(infile,outfile,nodefile):
    state = 0
    of = open(outfile,'w')
    nf = open(nodefile,'w')
    ids = {}
    maxnid = 0
    catGroup = []
    G = snap.TNEANet.New()
    with open(infile,'r') as f:
        for line in f:
                if "<newbook>" in line:
                    addEdges(catGroup,G)
                    #writeGroup(catGroup,of)
                    catGroup = []
                else:
                    name = line.strip()
                    nid = -1
                    if name in ids.keys():
                        nid = ids[name]
                    else:
                        nid = maxnid+1
                        ids[name] = nid
                        G.AddNode(nid)
                        nf.write(str(nid)+" "+name+"\n")
                        maxnid+=1
                    catGroup.append(nid)   
    print "Edges: ",G.GetEdges()," nodes: ",G.GetNodes() 
    printEdges(G,of)
    of.close()
    nf.close()

args  = sys.argv
infile = str(args[1])
outfile = str(args[2])
nodefile = str(args[3])
readInfoFile(infile,outfile,nodefile)
