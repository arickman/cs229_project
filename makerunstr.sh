#!/bin/bash

#ingraph="node2vec/graph/karate.edgelist"
ingraph="data/updated_amazon-meta-catlist-rankmin-edges.txt"
name="amazon-meta-min10krank"
anomlist=$name"anom.out"
anomlist2=$name"anom2.out"
newgraph=$name".edgelist.new"
plotdir=$name"plots"
plotdirnew=$name"plots_iterated"
updatedEmbed="new"$name"features.emd"

mkdir $plotdir
mkdir $plotdirnew
d=10
w=10
l=10
p=0.1
q=1

outemd="results/"updated_$name"_"$d"_"$w"_"$l"_"$p"_"$q".emd"
labels="data/amazon-meta-catlist-rankmin-nodes.txt"
n2vpath="./Snap-4.1/examples/node2vec/node2vec"
runstr="$n2vpath -i:$ingraph -o:$outemd -d:$d -l:$l  -r:$w -p:$p -q:$q -w"
echo $runstr > runtemp.txt
#python node2vec/src/main.py --input $ingraph --output $outemd --dimension $d --num-walks $w --p 1 --q 1 --walk-length $l #--weighted
##python kmeans_node2vec.py $outemd $anomlist $plotdir $ingraph $labels
#python updateEdgeList.py $ingraph $anomlist $newgraph -1

#python node2vec/src/main.py --input $newgraph --output $updatedEmbed
#python kmeans_node2vec.py $updatedEmbed $anomlist2 $plotdirnew $newgraph
