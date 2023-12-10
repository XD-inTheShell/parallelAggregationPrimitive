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

int simpleHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int localHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int localHashnSharedAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap, int debug);
int cucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int localncucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int test(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int readFile(std::string fileName, 
                std::vector<Key> &ukeys, std::vector<Value> &uvalues,
                std::vector<Key> &okeys, std::vector<Value> &ovalues){
    std::ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        return false;
    }
    std::string line;
    
    std::vector<std::pair<Key, Value>> entries;
    while (std::getline(inFile, line)) {
        std::stringstream sstream(line);
        std::string str;
        std::pair<Key, Value> entry;
        std::getline(sstream, str, ' ');
        Key key         = (Key)atoi(str.c_str());
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



int main(int argc, char** argv)
{
    printf("HI\n");

    std::vector<Key> ukeys, okeys;
    std::vector<Value> uvalues, ovalues;
    std::unordered_map<Key, Value> shumap, loumap, lsumap, cuumap, lcumap;
    readFile("../testcases/inputs/in.txt", ukeys, uvalues, okeys, ovalues);
    printf("----------------------Unsorted Keys----------------------- \n");
    simpleHashAggregate(ukeys, uvalues, shumap);
    // writeFile("out/shout.txt", shumap);

    localHashAggregate(ukeys, uvalues, loumap);
    // writeFile("out/loout.txt", loumap);

    localHashnSharedAggregate(ukeys, uvalues, lsumap,0);
    // writeFile("out/lsout.txt", lsumap);

    cucoHashAggregate(ukeys, uvalues, cuumap);
    // test(ukeys, uvalues, umap);
    // writeFile("out/cuout.txt", cuumap);

    localncucoHashAggregate(ukeys, uvalues, lcumap);
    // writeFile("out/lcout.txt", lcumap);

    printf("----------------------Sorted Keys----------------------- \n");
    shumap.clear(); 
    loumap.clear(); 
    lsumap.clear(); 
    cuumap.clear(); 
    lcumap.clear();
    simpleHashAggregate(okeys, ovalues, shumap);
    writeFile("out/shout.txt", shumap);

    localHashAggregate(okeys, ovalues, loumap);
    writeFile("out/loout.txt", loumap);

    localHashnSharedAggregate(okeys, ovalues, lsumap,1);
    writeFile("out/lsout.txt", lsumap);

    cucoHashAggregate(okeys, ovalues, cuumap);
    // test(okeys, ovalues, umap);
    writeFile("out/cuout.txt", cuumap);

    localncucoHashAggregate(okeys, ovalues, lcumap);
    writeFile("out/lcout.txt", lcumap);
    
    return 0;
}