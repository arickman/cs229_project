mm="min"
d=128
w=10
l=10
p=0.1
q=1
name="amazon-meta-"$mm"10krank"
#outemd="results/amazon-meta-min10krank_10_10_10_0.1_1.emd"
outemd = "./amazon-meta-min10krank_128_10_10_1_1.emd"
#anomlist="anoms.out"
anomlist = "anoms_from_kmeans.txt"
plotdir="catlist-rank"$mm"_"$name"_"$d"_"$w"_"$l"_"$p"_"$q
#plotdir = "rt_catlist"
mkdir $plotdir
ingraph="data/amazon-meta-catlist-rank"$mm"-edges.txt"
labels="data/amazon-meta-catlist-rank"$mm"-nodes.txt"
#python kmeans_node2vec2.py ./amazon-meta-min10krank_128_10_10_1_1.emd anoms_from_kmeans.txt $plotdir $ingraph $labels
#python kmeans_node2vec.py ./amazon-meta-min10krank_128_10_10_1_1.emd anoms_from_kmeans.txt $plotdir $ingraph $labels
python anomaly_removal.py ./amazon-meta-min10krank_10_10_10_0.1_1.emd anoms_from_kmeans.txt "catlist-rank"$mm"_"$name"_"$d"_"$w"_"$l"_"$p"_"$q "data/amazon-meta-catlist-rank"$mm"-edges.txt" "data/amazon-meta-catlist-rank"$mm"-nodes.txt"

python updateEdgeList.py data/amazon-meta-catlist-rank"$mm"-edges.txt anoms_from_kmeans.txt data/updated_amazon-meta-catlist-rank"$mm"-edges.txt -1
