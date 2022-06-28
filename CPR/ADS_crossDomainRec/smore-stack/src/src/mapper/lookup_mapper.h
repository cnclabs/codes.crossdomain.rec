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
        int size, relation_size, dimension;
        std::vector<std::vector<double>> embedding;
        std::vector<std::vector<double>> relation_embedding;

        // embedding function
        std::vector<double> avg_embedding(std::vector<long>& indexes);
        std::vector<double> textgcn_embedding(std::vector<long>& indexes);
        std::vector<double> textgcn_embedding_with_neg(std::vector<long>& indexes, std::vector<long>& neg_indexes);
        std::vector<double> meta_gcn_embedding(std::vector<long>& indexes);
        std::vector<double> meta_avg_embedding(std::vector<long>& indexes);
        //std::vector<double> LookupMapper::TransE_transform(long index, long relation_index);
        //std::vector<double> LookupMapper::RotatE_transform(long index, long relation_index);

        // constructor
        LookupMapper(int size, int dimension);
        void add_relation_embeddings(int size, int dimension);

        // load pretrain
        void load_pretrain(std::string file_name, n2iHash node2index);

        // update function
        void update(long index, std::vector<double>& loss_vector, double alpha);
        void update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda);
        void update_relation_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda);

        // save function
        void save_to_file(std::vector<char*>& index2vertex, std::string file_name);
        void save_relation_to_file(std::string *index2relation, std::string file_name);
        void save_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append);
        void save_trans_to_file(FileGraph* file_graph, std::string file_name);
        void save_gcn_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append);
        void save_avg_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append);
        void save_meta_gcn_to_file(FileGraph* main_graph,
                                   std::vector<FileGraph>& meta_graphs,
                                   std::vector<long> indexes,
                                   std::string file_name,
                                   int append);
        void save_meta_avg_to_file(FileGraph* main_graph,
                                   std::vector<FileGraph>& meta_graphs,
                                   std::vector<long> indexes,
                                   std::string file_name,
                                   int append);

        // overload operator
        std::vector<double>& operator[](long index);
};
#endif
