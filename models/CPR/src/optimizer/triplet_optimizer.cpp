#include "triplet_optimizer.h"

TripletOptimizer::TripletOptimizer() {
    // pre-compute sigmoid func
    this->cached_sigmoid.resize(SIGMOID_TABLE_SIZE);
    for (int i = 0; i != SIGMOID_TABLE_SIZE + 1; i++)
    {
        double x = i * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        this->cached_sigmoid[i] = 1.0 / (1.0 + exp(-x));
    }
}

double TripletOptimizer::fast_sigmoid(double value) {
    if (value < -MAX_SIGMOID)
    {
        return 0.0;
    }
    else if (value > MAX_SIGMOID)
    {
        return 1.0;
    }
    else
    {
        return this->cached_sigmoid[ int((value + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2) ];
    }
}

int TripletOptimizer::feed_margin_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& to_embedding_pos, std::vector<double>& to_embedding_neg, double margin, int dimension, std::vector<double>& from_loss, std::vector<double>& to_loss_pos, std::vector<double>& to_loss_neg) {

    std::vector<double> diff_to_embedding;
    diff_to_embedding.resize(dimension, 0.0);

    double gradient, prediction=-margin;

    for (int d=0; d<dimension;d++)
    {
        diff_to_embedding[d] = to_embedding_pos[d] - to_embedding_neg[d];
        prediction += from_embedding[d] * ( diff_to_embedding[d] );
    }

    gradient = this->fast_sigmoid(0.0-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * diff_to_embedding[d];
        to_loss_pos[d] += gradient * from_embedding[d];
        to_loss_neg[d] -= gradient * from_embedding[d];
    }
    return 1;
}
