#ifndef LOOKUP_MAPPER_H
#define LOOKUP_MAPPER_H
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../util/file_graph.h"

class LookupMapper {
    public:
        //variable
        int size, dimension;
        std::vector<std::vector<double>> embedding;

        // embedding function
        std::vector<double> textgcn_embedding(std::vector<long>& indexes);

        // constructor
        LookupMapper(int size, int dimension);

        // load pretrain
        void load_pretrain(std::string file_name, n2iHash node2index);

        // update function
        void update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda);

        // save function
        void save_gcn_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append);

        // overload operator
        std::vector<double>& operator[](long index);
};
#endif
