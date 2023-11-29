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

int cudaAggregate(std::vector<int> &keys, std::vector<double> &values, std::unordered_map<int, double> &umap);
int readFile(std::string fileName, std::vector<int> &keys, std::vector<double> &values){
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
        int key         = (int)atoi(str.c_str());
        keys.push_back(key);
        std::getline(sstream, str, ' ');
        double value    = (double)atof(str.c_str());
        values.push_back(value);
    }
    inFile.close();
    return 0;
}
int writeFile(std::string fileName, std::vector<int> &keys, std::vector<double> &values, std::unordered_map<int, double> &umap){
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
    
    std::vector<int> keys;
    std::vector<double> values;
    std::unordered_map<int, double> umap;
    readFile("../testcases/inputs/in.txt", keys, values);
    cudaAggregate(keys, values, umap);
    writeFile("out.txt", keys, values, umap);
    return 0;
}