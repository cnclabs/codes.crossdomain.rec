#include "lookup_mapper.h"

LookupMapper::LookupMapper(int size, int dimension) {
    this->embedding.resize(size);
    this->size = size;
    this->dimension = dimension;
    for (long index=0; index<size; ++index)
    {
        this->embedding[index].resize(dimension);
        for (int d=0; d<dimension; ++d)
        {
            this->embedding[index][d] = (rand()/(double)RAND_MAX - 0.5) / dimension;
        }
    }
}

void LookupMapper::add_relation_embeddings(int size, int dimension){
    this->relation_embedding.resize(size);
    this->relation_size = size;
    this->dimension = dimension;
    for (long index=0; index<size; ++index)
    {
        this->relation_embedding[index].resize(dimension);
        for (int d=0; d<dimension; ++d)
        {
            this->relation_embedding[index][d] = (rand()/(double)RAND_MAX - 0.5) / dimension;
        }
    }
}

void LookupMapper::load_pretrain(std::string file_name, n2iHash node2index) {
    std::ifstream embedding_file;
    std::string line, node, value;
    long index, counter=0;

    std::cout << "Load Pre-train Embeddings:" << std::endl;
    embedding_file.open(file_name);
    if(!embedding_file.is_open()) {
        throw("embedding file error");
    }
    while(std::getline(embedding_file, line)) {
        std::istringstream line_stream(line);
        std::getline(line_stream, node, '\t');
        index = node2index.search_key(strdup(node.c_str()));
        if (index != -1)
        {
            counter++;
            for (int d=0; d<dimension; ++d)
            {
                line_stream >> value;
                this->embedding[index][d] = std::stod(value);
            }
        }
    }
    std::cout << "\t# detections:\t" << counter << std::endl;
}

void LookupMapper::update(long index, std::vector<double>& loss_vector, double alpha) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*loss_vector[d];
    }
}

void LookupMapper::update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*(loss_vector[d] - lambda*embedding[index][d]);
    }
}

void LookupMapper::save_gcn_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append) {
    std::vector<double> fused_embedding(this->dimension, 0.0);
    long branch, from_index, to_index;
    double weight, weight_sum;
    std::ofstream embedding_file;
    if (append)
    {
        std::cout << "Append Mapper:" << std::endl;
        embedding_file.open(file_name, std::ios_base::app);
    }
    else
    {
        std::cout << "Save Mapper:" << std::endl;
        embedding_file.open(file_name);
    }

    if (embedding_file)
    {
        for (auto from_index: indexes)
        {
            branch = file_graph->index_graph[from_index].size();
            if (branch > 0)
            {
                embedding_file << file_graph->index2node[from_index];
                fused_embedding.assign(this->dimension, 0.0);
                weight_sum = 0.0;
                for (auto it: file_graph->index_graph[from_index])
                {
                    to_index = it.first;
                    weight = file_graph->index_graph[from_index][to_index];
                    weight_sum += weight;
                    for (int dim=0; dim!=this->dimension; dim++)
                    {
                        fused_embedding[dim] += this->embedding[to_index][dim]*weight;
                    }
                }
                embedding_file << "\t" << (this->embedding[from_index][0]+fused_embedding[0]/weight_sum)/2.0;
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << (this->embedding[from_index][dim]+fused_embedding[dim]/weight_sum)/2.0;
                }
                embedding_file << std::endl;
            }
            else
            {
                embedding_file << file_graph->index2node[from_index];
                embedding_file << "\t" << this->embedding[from_index][0];
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << this->embedding[from_index][dim];
                }
                embedding_file << std::endl;
            }
        }
        std::cout << "\tSave to <" << file_name << ">" << std::endl;
    }
    else
    {
        std::cout << "\tfail to open file" << std::endl;
    }
    embedding_file.close();
}

std::vector<double>& LookupMapper::operator[](long index) {
    return this->embedding[index];
}

std::vector<double> LookupMapper::avg_embedding(std::vector<long>& indexes) {
    double size = indexes.size();
    std::vector<double> avg_embedding(this->dimension, 0.0);
    for (auto index: indexes)
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[index][d];
    for (int d=0; d<this->dimension; d++)
        avg_embedding[d] /= size;
    return avg_embedding;
}

std::vector<double> LookupMapper::textgcn_embedding(std::vector<long>& indexes) {
    std::vector<double> avg_embedding(this->dimension, 0.0);
    double size = indexes.size()-1;
    if (size)
    {
        for (auto it=++indexes.begin(); it!=indexes.end(); it++)
            for (int d=0; d<this->dimension; d++)
                avg_embedding[d] += this->embedding[*(it)][d];
        for (int d=0; d<this->dimension; d++)
        {
            avg_embedding[d] = (this->embedding[indexes[0]][d] + avg_embedding[d]/size)/2.0;
        }
    }
    else
    {
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[indexes[0]][d];
    }
    return avg_embedding;
}

std::vector<double> LookupMapper::textgcn_embedding_with_neg(std::vector<long>& indexes, std::vector<long>& neg_indexes) {
    //has to use center for its "sampling" aggregation
    //std::cout << "positive aggregation ..." << std::endl;
    std::vector<double> avg_embedding(this->dimension, 0.0);
    double size = indexes.size()-1;
    //std::cout << "negative aggregation ..." << std::endl;
    std::vector<double> neg_avg_embedding(this->dimension, 0.0);
    double neg_size = neg_indexes.size();
    //double num_of_embed_type = 3;
    int final_vec_length = 1;

    // calculate embedding difference

    if (size)
    {
        //double agg_num = 0;
        for (auto it=++indexes.begin(); it!=indexes.end(); it++)
            //agg_num++;
            for (int d=0; d<this->dimension; d++)
                avg_embedding[d] += this->embedding[*(it)][d];
        for (int d=0; d<this->dimension; d++)
        {
            avg_embedding[d] = (this->embedding[indexes[0]][d] + avg_embedding[d]/size);
            //avg_embedding[d] = (this->embedding[indexes[0]][d] + avg_embedding[d])/num_of_embed_type;
        }
        final_vec_length += 1;
    }
    else
    {
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[indexes[0]][d];
    }

    if (neg_size)
    {
        //double neg_agg_num=0;
        //std::cout << "neg agg size: " << neg_size << std::endl;
        //std::cout << "aggregate negative nodes" << std::endl;
        //std::cout << "begin: " << *(neg_indexes.begin()) << "end: " << *(neg_indexes.end()) << std::endl;
        for (auto it=++neg_indexes.begin(); it!=neg_indexes.end(); it++)
            //neg_agg_num++;
            for (int d=0; d<this->dimension; d++)
                neg_avg_embedding[d] += this->embedding[*(it)][d];
        //std::cout << "average negative aggregations" << std::endl;
        for (int d=0; d<this->dimension; d++)
        {
            neg_avg_embedding[d] = neg_avg_embedding[d]/neg_size;
        }
        final_vec_length += 1;
    }
    //std::cout << "merge with positive aggregations" << std::endl;
    for (int d=0; d<this->dimension; d++)
        avg_embedding[d] = avg_embedding[d] - neg_avg_embedding[d]/neg_size;

    for (int d=0; d<this->dimension; d++)
        avg_embedding[d] = avg_embedding[d]/final_vec_length;
 
    return avg_embedding;
}

std::vector<double> LookupMapper::meta_gcn_embedding(std::vector<long>& indexes) {
    std::vector<double> meta_embedding(this->dimension, 0.0);
    double observed=0;
    for (auto it=++indexes.begin(); it!=indexes.end(); it++)
    {
        if (*(it)!=-1)
        {
            for (int d=0; d<this->dimension; d++)
                meta_embedding[d] += this->embedding[*(it)][d];
            observed++;
        }
    }

    if (observed)
    {
        for (int d=0; d<this->dimension; d++)
        {
            meta_embedding[d] = this->embedding[indexes[0]][d] + meta_embedding[d]/observed;
        }
    }
    else
    {
        for (int d=0; d<this->dimension; d++)
            meta_embedding[d] += this->embedding[indexes[0]][d];
    }
    return meta_embedding;
}

std::vector<double> LookupMapper::meta_avg_embedding(std::vector<long>& indexes) {
    std::vector<double> meta_embedding(this->dimension, 0.0);
    double observed=0;
    for (auto it=indexes.begin(); it!=indexes.end(); it++)
    {
        if (*(it)!=-1)
        {
            for (int d=0; d<this->dimension; d++)
                meta_embedding[d] += this->embedding[*(it)][d];
            observed++;
        }
    }
    for (int d=0; d<this->dimension; d++)
        meta_embedding[d] /= observed;
    return meta_embedding;
}

