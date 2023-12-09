python3 generator.py 
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
echo "cuout"
diff out/cuout.txt ../seq_impl/out.txt 
echo "lcout"
diff out/lcout.txt ../seq_impl/out.txt 
