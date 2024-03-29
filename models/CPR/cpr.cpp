#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include "./src/util/util.h"                       // arguments
#include "./src/util/file_graph.h"                 // graph
#include "./src/sampler/vc_sampler.h"              // sampler
#include "./src/mapper/lookup_mapper.h"            // mapper
#include "./src/optimizer/triplet_optimizer.h"     // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_ut_path = arg_parser.get_str("-train_ut", "", "input user-target graph path");
    std::string train_us_path = arg_parser.get_str("-train_us", "", "input user-source graph path");
    std::string save_name = arg_parser.get_str("-save", "cse.embed", "path for saving mapper");
    std::string pre_train_path =  arg_parser.get_str("-pre-train", "", "pre-train graph path, keep it blank if not to use");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*user_amount)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.1, "init learning rate");
    double user_reg = arg_parser.get_double("-user_reg", 0.01, "l2 regularization");
    double item_reg = arg_parser.get_double("-item_reg", 0.01, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // main
    // 0. [FileGraph] read graph
    std::cout << "(UT-Graph)" << std::endl;
    FileGraph ut_file_graph(train_ut_path, 0);
    std::cout << "(US-Graph)" << std::endl;
    FileGraph us_file_graph(train_us_path, 0, ut_file_graph.index2node);

    // 1. [Sampler] determine what sampler to be used
    VCSampler ut_sampler(&ut_file_graph);
    VCSampler us_sampler(&us_file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(us_sampler.vertex_size, dimension);
    if(pre_train_path != "")
    {
	mapper.load_pretrain(pre_train_path, us_file_graph.node2index);
    }
    //LookupMapper i_mapper(us_sampler.vertex_size, dimension);

    // 3. [Optimizer] claim the optimizer
    TripletOptimizer optimizer;

    // 4. building the blocks [MF]
    std::cout << "Start Training:" << std::endl;
    int edge_amount = ut_file_graph.edge_size;;
    std::cout << "EDGE AMOUNT:" << edge_amount << "!" << std::endl;
    unsigned long long total_update_times = (unsigned long long)update_times*edge_amount;
    unsigned long long worker_update_times = total_update_times/worker;
    unsigned long long finished_update_times = 0;
    Monitor monitor(total_update_times);

    omp_set_num_threads(worker);
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        int step;
        long user, source, target, item_pos, item_neg;
        std::vector<double> user_embed(dimension, 0.0);
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> item_loss_pos(dimension, 0.0);
        std::vector<double> item_loss_neg(dimension, 0.0);
        std::vector<long> user2items;
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            user = ut_sampler.draw_a_vertex();
            target = ut_sampler.draw_a_context_safely(user);
            source = us_sampler.draw_a_context_safely(user);

            // [user + target]
            user2items.clear();
            user2items.push_back(user);
            user2items.push_back(target);
            if (source!=-1)
                user2items.push_back(source);
            user_embed = mapper.textgcn_embedding(user2items);

            item_pos = ut_sampler.draw_a_context(user);
            for (int b=0; b<num_negative; b++)
            {
                //item_pos = ut_sampler.draw_a_context(user);
                item_neg = ut_sampler.draw_a_context_uniformly();

                optimizer.feed_margin_bpr_loss(user_embed,
                                               mapper[item_pos],
                                               mapper[item_neg],
                                               8.0,
                                               dimension,
                                               user_loss,
                                               item_loss_pos,
                                               item_loss_neg);
                //mapper.update_with_l2(item_pos, item_loss_pos, alpha, item_reg);
                mapper.update_with_l2(item_neg, item_loss_neg, alpha, item_reg);

                //item_loss_pos.assign(dimension, 0.0);
                item_loss_neg.assign(dimension, 0.0);
            }
            mapper.update_with_l2(item_pos, item_loss_pos, alpha, item_reg);
            item_loss_pos.assign(dimension, 0.0);
            mapper.update_with_l2(user, user_loss, alpha, user_reg);
            mapper.update_with_l2(target, user_loss, alpha, user_reg);
            if (source!=-1)
                mapper.update_with_l2(source, user_loss, alpha, user_reg);
            user_loss.assign(dimension, 0.0);

            // 5. print progress
            update++;
            if (update % report_period == 0) {
                alpha = init_alpha* ( 1.0 - (double)(finished_update_times)/total_update_times );
                if (alpha < alpha_min)
                    alpha = alpha_min;
                finished_update_times += report_period;
                monitor.progress(&finished_update_times);
            }
        }
    }
    monitor.end();
    mapper.save_gcn_to_file(&us_file_graph, us_file_graph.get_all_nodes(), save_name, 0);
    return 0;
}
