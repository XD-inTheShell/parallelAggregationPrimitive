python3 generator.py 
cd ../seq_impl
pwd
./seqAggregate
cd ../cuda_impl
pwd
./cudaAggregate
diff out.txt ../seq_impl/out.txt 
