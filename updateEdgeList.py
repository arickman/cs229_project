import sys


def fileToList(filename):
    outlist = []
    with open(filename,'r') as f:
        for line in f:
            dat = line.split()[0]
            outlist.append(dat)
        print "outlist: "
        print outlist
        return outlist

def writeFilteredGraph(infile,outfile,anoms,direction):
    of = open(outfile,'w')
    with open(infile,'r') as f:
        for line in f:
            edge = line.split()
            keep = 0
            if edge[0] in anoms or edge[1] in anoms:
                    #either is in anoms and we are rejecting
                    if direction<0:
                        keep = -1
                    else: #>0
                        if edge[0] in anoms and edge[1] in anoms:
                            keep = 1
                        else:
                            keep = -1
            else:#neither in anoms
                if direction<0:
                    keep = 1
                else:
                    keep = -1
            if keep == 0:
                print "err!"
            if keep ==1:            
                of.write(edge[0])
                of.write(" ")
                of.write(edge[1])
                of.write(" ")
                of.write(edge[2])
                of.write("\n")

    of.close()







args = sys.argv
infile = str(args[1])
anomfile = str(args[2])
outfile = str(args[3])

direction = int(args[4])
anoms = fileToList(anomfile);
writeFilteredGraph(infile,outfile,anoms,direction)


