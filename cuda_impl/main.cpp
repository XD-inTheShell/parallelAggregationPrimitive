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
int cucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int test(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap);
int readFile(std::string fileName, std::vector<Key> &keys, std::vector<Value> &values){
    std::ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        return false;
    }
    std::string line;
    while (std::getline(inFile, line)) {
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        Key key         = (Key)atoi(str.c_str());
        keys.push_back(key);
        std::getline(sstream, str, ' ');
        #ifdef VALUEINT
            Value value    = (Value)atoi(str.c_str());
        #else
            Value value    = (double)atof(str.c_str());
        #endif
        values.push_back(value);
    }
    inFile.close();
    return 0;
}
int writeFile(std::string fileName, std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap){
    std::map<Key, Value> res(umap.begin(), umap.end());

    fileName = "out.txt";
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

    std::vector<Key> keys;
    std::vector<Value> values;
    std::unordered_map<Key, Value> umap;
    readFile("../testcases/inputs/in.txt", keys, values);
    // simpleHashAggregate(keys, values, umap);
    cucoHashAggregate(keys, values, umap);
    test(keys, values, umap);
    writeFile("out.txt", keys, values, umap);
    return 0;
}