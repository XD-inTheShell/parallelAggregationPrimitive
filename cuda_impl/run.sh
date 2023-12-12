# python3 generator.py 
cd ../seq_impl
pwd
./seqAggregate
cd ../cuda_impl
pwd
./cudaAggregate
echo "shout"
diff out/shout.txt ../seq_impl/out.txt
echo "loout"
diff out/loout.txt ../seq_impl/out.txt 
echo "lsout"
diff out/lsout.txt ../seq_impl/out.txt
echo "sout"
diff out/sout.txt ../seq_impl/out.txt
echo "cuout"
diff out/cuout.txt ../seq_impl/out.txt 
echo "lcout"
diff out/lcout.txt ../seq_impl/out.txt 
echo "lscout"
diff out/lscout.txt ../seq_impl/out.txt 
echo "scout"
diff out/scout.txt ../seq_impl/out.txt 