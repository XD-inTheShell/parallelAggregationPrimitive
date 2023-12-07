python3 generator.py 
cd ../seq_impl
pwd
./seqAggregate
cd ../cuda_impl
pwd
./cudaAggregate
diff out/shout.txt ../seq_impl/out.txt
diff out/loout.txt ../seq_impl/out.txt 
diff out/cuout.txt ../seq_impl/out.txt 
