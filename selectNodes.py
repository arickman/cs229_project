import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snap
import sys


def readInfoFile(infile):
    gmap = {}
    gmap["Book"] = 0
    gmap["Video"] = 1
    gmap["Music"] = 2
    gmap["DVD"] = 3
    nodes = [[],[],[],[]]
    currentId = -1#safety check that have new, real id
    with open(infile,'r') as f:
        for line in f:
            if "Id: " in line:
                currentId = line.split()[1]
            if "group:" in line:
                foundgroup = line.split()[1]
                if foundgroup in gmap.keys() and currentId !=-1:
                    nodes[gmap[foundgroup]].append(currentId)
                    currentId = -1
    print "gsizes: "
    for i in range(0,4):
        print "i: ",i,len(nodes[i])
    return nodes

def writeSubset(nodes,catsize,outfile):
    #shuffle nodes[0...3]
    shufflednodes = []
    for i in range(0,len(nodes)):
        shufflednodes.append(nodes[i]) #todo: actually shuffle

    #write first catsize of each
    with open(outfile,'w') as f:
        for i in range(0,len(nodes)):
            for j in range(0,catsize[i]):
                f.write(str(shufflednodes[i][j]))
                f.write(" ")
                f.write(str(i))
                f.write('\n')
    



args  = sys.argv
infile = str(args[1])
outfile = str(args[2])


nodes = readInfoFile(infile)
catsize = [393561,0,0,0]
writeSubset(nodes,catsize,outfile)
