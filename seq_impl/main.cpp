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
#include "../common.h"

int seq_impl(std::string fileName) {
    std::ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        return false;
    }

    std::string line;

    std::unordered_map<Key, Value> umap;
    while (std::getline(inFile, line)) {
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        Key key         = (Key)atoi(str.c_str());
        std::getline(sstream, str, ' ');
        #ifdef VALUEINT
            Value value    = (Key)atoi(str.c_str());
        #else
            Value value    = (double)atof(str.c_str());
        #endif

        auto search = umap.find(key);
        if ( search != umap.end())
            search->second += value;
        else
            umap[key] = value;
    }
    inFile.close();

    // Finished addition, write to file
    std::map<Key, Value> res(umap.begin(), umap.end());

    fileName = "out.txt";
    std::ofstream file(fileName);
    if (!file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return 0;
    }
    file << std::setprecision(9);
    for (auto d : res) {
        file << d.first << " " << d.second << std::endl;
    }
    file.close();
    if (!file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    return true;
    
}

int main(int argc, char** argv)
{
    seq_impl("../testcases/inputs/in.txt");
    return 0;
}

