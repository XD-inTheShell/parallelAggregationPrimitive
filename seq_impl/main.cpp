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

int seq_impl(std::string fileName) {
    std::ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        return false;
    }

    std::string line;

    std::unordered_map<int, double> umap;
    while (std::getline(inFile, line)) {
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        int key         = (int)atoi(str.c_str());
        std::getline(sstream, str, ' ');
        double value    = (double)atof(str.c_str());

        auto search = umap.find(key);
        if ( search != umap.end())
            search->second += value;
        else
            umap[key] = value;
    }
    inFile.close();

    // Finished addition, write to file
    std::map<int, double> res(umap.begin(), umap.end());

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

