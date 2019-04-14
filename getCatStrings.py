import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snap
import sys


def readInfoFile(infile,outfile):
    state = 0
    of = open(outfile,'w')
    with open(infile,'r') as f:
        for line in f:
            if "categories" in line and state == 0:
                state = 1
                #coin = 1#np.random.uniform(0,1)
                #if coin<2:
                 #   state = 1
               # else:
                #    state = -1 #ignore this book
            elif "salesrank" in line:
                rank = int(line.split()[1])
                #if rank > 2370677:
                if rank < 2236:
                    state = 0
                else:
                    state = -2
            elif "reviews" in line and state == 1: #state =1-> did a book
                state = -2
                of.write("<newbook>\n")
            elif state == 1:
#                print "line: ",line
                cats = line.strip().split("|")
                for c in cats:
                    if len(c)>3:
                        of.write(c)
                        of.write("\n")
 #           elif state ==-1:
  #              print "ignored line: ",line
    of.close()

args  = sys.argv
infile = str(args[1])
outfile = str(args[2])
readInfoFile(infile,outfile)
