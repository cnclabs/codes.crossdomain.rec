#ifndef VC_SAMPLER_H
#define VC_SAMPLER_H

#include <cmath>
#include <unordered_map>
#include <vector>
#include "../util/file_graph.h"
#include "../util/random.h"
#include "alias_methods.h"

class VCSampler {
    /* VCSampler performs vertex-context-style sampling
     */
    public:
        VCSampler(FileGraph*);

        // variables
        long vertex_size=0, context_size=0;
        AliasMethods vertex_sampler, context_sampler, negative_sampler;
        AliasMethods vertex_uniform_sampler, context_uniform_sampler;
        std::vector<long> contexts; // context ref.
        //std::unordered_map<long, std::vector<long>> adjacency; // context ref.

        // functions
        long draw_a_vertex();
        long draw_a_context(long vertex);
        long draw_a_context_uniformly();
        long draw_a_context_safely(long vertex);
};
#endif
