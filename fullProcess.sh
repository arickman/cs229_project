#ingraph="node2vec/graph/karate.edgelist"
ingraph="data/amazon-meta-catlist-rankmax-edges.txt"
name="amazon-meta-rankmax_5k"
anomlist=$name"anom.out"
anomlist2=$name"anom2.out"
newgraph=$name".edgelist.new"
plotdir=$name"plots"
plotdirnew=$name"plots_iterated"
updatedEmbed="new"$name"features.emd"
outemd="node2vec/emb/"$name".emd"
mkdir $plotdir
mkdir $plotdirnew
d=10
w=10
l=10
p=0.1
q=1
labels="data/amazon-meta-catlist-rankmax-nodes.txt"
/Users/alex/Research/Snap-4.1/examples/node2vec/node2vec -i:$ingraph -o:$outemd -d:$d -l:$l -r:$w -p:$p -q:$q -w
#python node2vec/src/main.py --input $ingraph --output $outemd --dimension $d --num-walks $w --p 1 --q 1 --walk-length $l #--weighted
python kmeans_node2vec.py $outemd $anomlist $plotdir $ingraph $labels
#python updateEdgeList.py $ingraph $anomlist $newgraph -1

#python node2vec/src/main.py --input $newgraph --output $updatedEmbed
#python kmeans_node2vec.py $updatedEmbed $anomlist2 $plotdirnew $newgraph
