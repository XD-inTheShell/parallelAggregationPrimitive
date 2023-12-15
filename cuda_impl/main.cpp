#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <unordered_map> 
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include "../common.h"

unsigned int computestep;

double simpleHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
double localHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
double localHashnSharedAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap, int debug
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
double cucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
double localncucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
int localnsharedHashcucoAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
int sharedHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);
int sharedHashcucoAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap
                            , std::map<std::string, std::map<unsigned int, double>> &perf);

int test();
int readFile(std::string fileName, 
                std::vector<Key> &ukeys, std::vector<Value> &uvalues,
                std::vector<Key> &okeys, std::vector<Value> &ovalues,
                unsigned int rsize){
    std::ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        return false;
    }
    std::string line;
    
    std::vector<std::pair<Key, Value>> entries;
    unsigned int read = 1;
    while (std::getline(inFile, line)) {
        std::stringstream sstream(line);
        std::string str;
        std::pair<Key, Value> entry;
        std::getline(sstream, str, ' ');
        // Key key         = (Key)atoi(str.c_str());
        Key key         = 1;
        ukeys.push_back(key);
        entry.first = key;
        std::getline(sstream, str, ' ');
        #ifdef VALUEINT
            Value value    = (Value)atoi(str.c_str());
        #else
            Value value    = (double)atof(str.c_str());
        #endif
        uvalues.push_back(value);
        entry.second = value;
        entries.push_back(entry);
        if(read==rsize){
            break;
        }
        read ++;
    }

    std::sort(entries.begin(), entries.end());
    for (auto entry : entries){
        okeys.push_back(entry.first);
        ovalues.push_back(entry.second);
        // printf("%u %u \n", entry.first, entry.second);
    }
    inFile.close();
    return 0;
}
int writeFile(std::string fileName, std::unordered_map<Key, Value> &umap){
    std::map<Key, Value> res(umap.begin(), umap.end());

    std::ofstream file(fileName);
    if (!file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return 0;
    }
    
    for (auto d : res) {
        file << d.first << " " << d.second << std::endl;
    }
    file.close();
    if (!file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    // std::ofstream file(fileName);
    // if (!file) {
    //     std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    //     return 0;
    // }
    // file << std::setprecision(10);
    // for (long unsigned int i=0; i<keys.size();i++) {
    //     file << keys[i] << " " << values[i] << std::endl;
    // }
    // file.close();
    // if (!file)
    //     std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    return true;
}

int genInputs(std::vector<Key> &ukeys, std::vector<Value> &uvalues,
                std::vector<Key> &okeys, std::vector<Value> &ovalues,
                unsigned int rsize, unsigned int keynum){
     unsigned int factor = rsize / keynum;    
    if(rsize % keynum){
        factor +=1;
    }
    unsigned int ukey, okey;
    printf("factor %d", factor);
    for(int i=0; i<rsize; i++){
        ukey = i % keynum;
        ukeys.push_back(ukey);
        okey = i / factor;
        // okeys.push_back(okey);
        okeys.push_back(1);
        // unsigned int value = 1;
        uvalues.push_back(1);
        ovalues.push_back(1);

        // printf("%u %u\n", ukey, okey);
    }
    return 0;
}

void run(std::vector<Key> &keys, std::vector<Value> &values, std::map<std::string, std::map<unsigned int, double>> &perf){

    std::unordered_map<Key, Value> shumap, loumap, lsumap, sumap, cuumap, lcumap, lscumap, scumap;

cudaDeviceReset();

    simpleHashAggregate(keys, values, shumap, perf);
    writeFile("out/shout.txt", shumap);

cudaDeviceReset();

    localHashAggregate(keys, values, loumap, perf);
    writeFile("out/loout.txt", loumap);

cudaDeviceReset();

    localHashnSharedAggregate(keys, values, lsumap,0, perf);
    writeFile("out/lsout.txt", lsumap);
    
cudaDeviceReset();

    sharedHashAggregate(keys, values, sumap, perf);
    writeFile("out/sout.txt", sumap);
    
cudaDeviceReset();

    cucoHashAggregate(keys, values, cuumap, perf);
    writeFile("out/cuout.txt", cuumap);
    
cudaDeviceReset();

    localncucoHashAggregate(keys, values, lcumap, perf);
    writeFile("out/lcout.txt", lcumap);
    
cudaDeviceReset();

    localnsharedHashcucoAggregate(keys, values, lscumap, perf);
    writeFile("out/lscout.txt", lscumap);
    
cudaDeviceReset();
    
    sharedHashcucoAggregate(keys, values, scumap, perf);
    writeFile("out/scout.txt", scumap);
}

void benchmark(std::vector<Key> &keys, std::vector<Value> &values, std::map<std::string, std::map<unsigned int, double>> &perf){

    std::unordered_map<Key, Value> shumap, loumap, lsumap, sumap, cuumap, lcumap, lscumap, scumap;

cudaDeviceReset();

    simpleHashAggregate(keys, values, shumap, perf);


// cudaDeviceReset();

//     localHashAggregate(keys, values, loumap, perf);


// cudaDeviceReset();

//     localHashnSharedAggregate(keys, values, lsumap,0, perf);

    
cudaDeviceReset();

    sharedHashAggregate(keys, values, sumap, perf);

    
cudaDeviceReset();

    cucoHashAggregate(keys, values, cuumap, perf);

    
// cudaDeviceReset();

//     localncucoHashAggregate(keys, values, lcumap, perf);

    
// cudaDeviceReset();

//     localnsharedHashcucoAggregate(keys, values, lscumap, perf);

    
// cudaDeviceReset();
    
//     sharedHashcucoAggregate(keys, values, scumap, perf);

}

int main(int argc, char** argv)
{
    printf("HI\n");

    std::vector<Key> ukeys, okeys;
    std::vector<Value> uvalues, ovalues;
    

    // std::map<std::string, std::map<unsigned int, double>> uperf, operf;

    std::map<unsigned int , std::map<std::string, std::map<unsigned int, double>>> uperf, operf;


    // readFile("../testcases/inputs/in.txt", ukeys, uvalues, okeys, ovalues, 20000000);

    // computestep = ukeys.size()/(GRIDSIZE*BLOCKSIZE)+1;
    // printf("step is %u\n", computestep);


    // printf("----------------------Unsorted Keys----------------------- \n");
    // run(ukeys, uvalues, perf);

    // printf("----------------------Sorted Keys----------------------- \n");
    // run(okeys, ovalues, perf);

    // int keycases = 
    // long unsigned int keysizes[keycases] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 121}

    int incases = 6;
    // In K
    long unsigned int insizes[incases]= {5,50,500,5000,50000,500000};
    for (int i=0; i<incases; i++){
        insizes[i] *= 1000; 
    }
    
    // unsigned int keysizes [9] = {25,50,60,70,80,85,90,95,99};
    unsigned int keysizes [2] = {25,99};
    for(int i=0; i<2; i++){
        keysizes[i] = keysizes[i] * KEYSIZE / 100;
    }
    keysizes [1] = 1023;
    
    // for(int i=0; i<11; i++){
    //     printf("%d:%u ", i, keysizes[i]);
    // }

    for(auto size:insizes){
        for(auto percent:keysizes){
            ukeys.clear();
            uvalues.clear();
            okeys.clear();
            ovalues.clear();
            
            // unsigned int keynum = percent * KEYSIZE / 100;
            genInputs(ukeys, uvalues,
                okeys, ovalues,
                size, percent);
            
            printf("size is %u", size);

    // cudaDeviceReset();
    // size_t free, total;
    // printf("\n");
    // cudaMemGetInfo(&free,&total);   
    // printf("%d B free of total %d B\n",free,total);
            computestep = ukeys.size()/(GRIDSIZE*BLOCKSIZE)+1;
            printf("step is %u\n", computestep);

            benchmark(ukeys, uvalues, uperf[percent]);
            benchmark(okeys, ovalues, operf[percent]);            
        }

    }

    std::string name;
    std::string fpath = "perf/";
    std::string uname = "1key_uperf.txt";
    std::string oname = "1key_operf.txt";

    FILE * pFile;
    name = fpath + std::to_string(KEYSIZE) + uname;
    pFile = fopen (name.c_str(),"w");
    for(auto keynum:uperf){
        fprintf(pFile, "KEYSIZE: %u\n", keynum.first);
        for(auto impl:keynum.second){
            fprintf(pFile, "\t%s: \n", impl.first.c_str());
            for(auto size:impl.second){
                fprintf(pFile, "\tinput size: %uK: \t%f kpair\n", size.first/1000, size.first/(size.second*1000000));
            }
        }
        fprintf(pFile, "\n");
    }
    
    name = fpath + std::to_string(KEYSIZE) + oname;
    pFile = fopen (name.c_str(),"w");
    for(auto keynum:operf){
        fprintf(pFile, "KEYSIZE: %u\n", keynum.first);
        for(auto impl:keynum.second){
            fprintf(pFile, "\t%s: \n", impl.first.c_str());
            for(auto size:impl.second){
                fprintf(pFile, "\tinput size: %uK: \t%f kpair\n", size.first/1000, size.first/(size.second*1000000));
        }
        fprintf(pFile, "\n");
    }  
    }
      
    

    // shumap.clear(); 
    // loumap.clear(); 
    // lsumap.clear(); 
    // cuumap.clear(); 
    // lcumap.clear();
    // simpleHashAggregate(okeys, ovalues, shumap, perf);
    // writeFile("out/shout.txt", shumap);

    // localHashAggregate(okeys, ovalues, loumap, perf);
    // writeFile("out/loout.txt", loumap);

    // localHashnSharedAggregate(okeys, ovalues, lsumap,1, perf);
    // writeFile("out/lsout.txt", lsumap);

    // cucoHashAggregate(okeys, ovalues, cuumap, perf);

    // writeFile("out/cuout.txt", cuumap);

    // localncucoHashAggregate(okeys, ovalues, lcumap, perf);
    // writeFile("out/lcout.txt", lcumap);

    // FILE * pFile;
    // pFile = fopen ("uperf.txt","w");
    // for(auto impl:uperf){
    //     fprintf(pFile, "%s: \n", impl.first.c_str());
    //     for(auto size:impl.second){
    //         fprintf(pFile, "input size: %uM: \t%f kpair\n", size.first/1000000, size.first/(size.second*1000000));
    //     }
    // }

    // pFile = fopen ("operf.txt","w");
    // for(auto impl:operf){
    //     fprintf(pFile, "%s: \n", impl.first.c_str());
    //     for(auto size:impl.second){
    //         fprintf(pFile, "input size: %uM: \t%f kpair\n", size.first/1000000, size.first/(size.second*1000000));
    //     }
    // }    


//     for(auto size:insizes){
//         ukeys.clear();
//         uvalues.clear();
//         okeys.clear();
//         ovalues.clear();

//         readFile("../testcases/inputs/in.txt", ukeys, uvalues, okeys, ovalues, size);

// cudaDeviceReset();
// size_t free, total;
// printf("\n");
// cudaMemGetInfo(&free,&total);   
// printf("%d B free of total %d B\n",free,total);
//         computestep = ukeys.size()/(GRIDSIZE*BLOCKSIZE)+1;
//         printf("step is %u\n", computestep);

//         run(ukeys, uvalues, uperf);
//         run(okeys, ovalues, operf);
//     }
    

//     // shumap.clear(); 
//     // loumap.clear(); 
//     // lsumap.clear(); 
//     // cuumap.clear(); 
//     // lcumap.clear();
//     // simpleHashAggregate(okeys, ovalues, shumap, perf);
//     // writeFile("out/shout.txt", shumap);

//     // localHashAggregate(okeys, ovalues, loumap, perf);
//     // writeFile("out/loout.txt", loumap);

//     // localHashnSharedAggregate(okeys, ovalues, lsumap,1, perf);
//     // writeFile("out/lsout.txt", lsumap);

//     // cucoHashAggregate(okeys, ovalues, cuumap, perf);

//     // writeFile("out/cuout.txt", cuumap);

//     // localncucoHashAggregate(okeys, ovalues, lcumap, perf);
//     // writeFile("out/lcout.txt", lcumap);

//     FILE * pFile;
//     pFile = fopen ("uperf.txt","w");
//     for(auto impl:uperf){
//         fprintf(pFile, "%s: \n", impl.first.c_str());
//         for(auto size:impl.second){
//             fprintf(pFile, "input size: %uM: \t%f kpair\n", size.first/1000000, size.first/(size.second*1000000));
//         }
//     }

//     pFile = fopen ("operf.txt","w");
//     for(auto impl:operf){
//         fprintf(pFile, "%s: \n", impl.first.c_str());
//         for(auto size:impl.second){
//             fprintf(pFile, "input size: %uM: \t%f kpair\n", size.first/1000000, size.first/(size.second*1000000));
//         }
//     }
    
    return 0;
}