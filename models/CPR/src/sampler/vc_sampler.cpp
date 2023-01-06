#include "vc_sampler.h"

VCSampler::VCSampler(FileGraph* file_graph) {
    /* See Figure 3. in https://arxiv.org/abs/1711.00227
     */
    this->vertex_size = file_graph->index2node.size();
    this->context_size = file_graph->index_graph.size();

    std::cout << "Build VC Sampler:" << std::endl;
    long from_index, to_index, to_offset;
    double weight;
    std::vector<double> vertex_distribution, context_distribution, neg_distribution;
    std::vector<double> vertex_uniform_distribution, context_uniform_distribution;
    vertex_distribution.resize(this->vertex_size, 0.0);
    vertex_uniform_distribution.resize(this->vertex_size, 0.0);
    context_uniform_distribution.resize(this->vertex_size, 0.0);
    neg_distribution.resize(this->vertex_size, 0.0);

    std::cout << "\tBuild Alias Methods" << std::endl;
    for (int from_index=0; from_index<this->vertex_size; from_index++)
    {
        context_distribution.clear();
        for (auto it: file_graph->index_graph[from_index])
        {
            to_index = it.first;
            weight = it.second;
            vertex_distribution[from_index] += weight;
            context_distribution.push_back(weight);
            neg_distribution[to_index] += weight;
            vertex_uniform_distribution[from_index] = 1.0;
            context_uniform_distribution[to_index] = 1.0;
            this->contexts.push_back(to_index);
        }
        this->context_sampler.append(context_distribution, 1.0);
    }
    this->vertex_sampler.append(vertex_distribution, 1.0);
    this->vertex_uniform_sampler.append(vertex_uniform_distribution, 1.0);
    this->context_uniform_sampler.append(context_uniform_distribution, 1.0);
    this->negative_sampler.append(neg_distribution, 0.75);
    std::cout << "\tDone" << std::endl;
}

long VCSampler::draw_a_vertex() {
    return this->vertex_sampler.draw();
}

long VCSampler::draw_a_context(long vertex_index) {
    return this->contexts[this->context_sampler.draw(vertex_index)];
}

long VCSampler::draw_a_context_safely(long vertex_index) {
    long context_index = this->context_sampler.draw_safely(vertex_index);
    if (context_index != -1)
        return this->contexts[context_index];
    return -1;
}
long VCSampler::draw_a_context_uniformly() {
    return this->context_uniform_sampler.draw();
}
