#include <iostream>
#include "cnode.h"
#include "math.h"

namespace tree{

    CSearchResults::CSearchResults(){
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num){
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->search_paths.push_back(std::vector<CNode*>());
        }
    }

    CSearchResults::~CSearchResults(){}

    //*********************************************************

    CNode::CNode(){
        this->action_num = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->q_init = 0.0;
        this->use_q_init = 0;
        this->reward = 0.0;
        this->prior = 0.0;
        this->ptr_node_pool = nullptr;
        this->data_idx_0 = 0;
        this->data_idx_1 = 0;
        this->best_action = 0;
        this->to_play = -1;  // node type, 0-operator, 1-attacker
    }

    CNode::CNode(float prior, int action_num, std::vector<CNode>* ptr_node_pool){
        this->prior = prior;
        this->action_num = action_num;
        this->q_init = 0.0;
        this->use_q_init = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->reward = 0.0;
        this->ptr_node_pool = ptr_node_pool;
        this->data_idx_0 = 0;
        this->data_idx_1 = 0;
        this->best_action = 0;
        this->to_play = -1;
    }

    CNode::~CNode(){}

    void CNode::expand(float reward, int data_idx_0, int data_idx_1){
        float prior;
        this->to_play = 0;
        this->reward = reward;
        this->data_idx_0 = data_idx_0;
        this->data_idx_1 = data_idx_1;

        // Ptr_node_pool is passed down recursively.
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;

        for(int a = 0; a < action_num; ++a){
            prior = 1 / (float)(action_num);

            int index = ptr_node_pool->size();
            this->children_index.push_back(index);

            CNode child = CNode(prior, action_num, ptr_node_pool);
            ptr_node_pool->push_back(child);
        }

    }

    void CNode::expand_attacker(float reward,
                                int data_idx_0,
                                int data_idx_1,
                                const std::vector<float> &policy_logits){
//        this->to_play = to_play;
        this->data_idx_0 = data_idx_0;
        this->data_idx_1 = data_idx_1;
        this->reward = reward;
        this->to_play = 1;

        int action_num = this->action_num;
        float temp_policy;
        float policy_sum = 0.0;
        float policy[action_num];
        float policy_max = FLOAT_MIN;
        for(int a = 0; a < action_num; ++a){
            if(policy_max < policy_logits[a]){
                policy_max = policy_logits[a];
            }
        }

        for(int a = 0; a < action_num; ++a){
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;
        for(int a = 0; a < action_num; ++a){
            prior = policy[a] / policy_sum;
            int index = ptr_node_pool->size();
            this->children_index.push_back(index);

            ptr_node_pool->push_back(CNode(prior, action_num, ptr_node_pool));
        }
    }

    void CNode::expand_q_init(float reward, int data_idx_0, int data_idx_1, const std::vector<float>& q_init){
        float prior;
        this->to_play = 0;
        this->reward = reward;
        this->data_idx_0 = data_idx_0;
        this->data_idx_1 = data_idx_1;

        // Ptr_node_pool is passed down recursively.
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;

        for(int a = 0; a < action_num; ++a){
            prior = 1 / (float)(action_num);

            int index = ptr_node_pool->size();
            this->children_index.push_back(index);

            CNode child = CNode(prior, action_num, ptr_node_pool);

            child.use_q_init = 1;
            child.q_init = q_init[a];

            ptr_node_pool->push_back(child);
        }

    }


    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises){
        float noise, prior;
        for(int a = 0; a < this->action_num; ++a){
            noise = noises[a];
            CNode* child = this->get_child(a);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float CNode::get_mean_q(int isRoot, float parent_q, float discount){
        float total_unsigned_q = 0.0;
        int total_visits = 0;

        for(int a = 0; a < this->action_num; ++a){
            CNode* child = this->get_child(a);
            if(child->visit_count > 0){
                float true_reward = child->reward;
                float qsa = true_reward + discount * child->value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;

        if(isRoot && total_visits > 0){
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else{
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }

        return mean_q;
    }

    void CNode::print_out(){
        printf("*****\n");
        printf("visit count: %d \t data_idx_1: %d \t data_idx_1: %d \t reward: %f \t prior: %f \n.",
            this->visit_count, this->data_idx_0, this->data_idx_1, this->reward, this->prior
        );
        printf("children_index size: %d \t pool size: %d \n.", this->children_index.size(), this->ptr_node_pool->size());
        printf("*****\n");
    }

    int CNode::expanded(){
        int child_num = this->children_index.size();
        if(child_num > 0) {
            return 1;
        }
        else {
            return 0;
        }
    }

    float CNode::value(){
        float true_value = 0.0;
        if(this->visit_count == 0){
            // printf("visit count=0, raise value calculation error.\n");
            return 0.0;
        }
        else{
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<int> CNode::get_trajectory(){
        std::vector<int> traj;

        CNode* node = this;
        int best_action = node->best_action;
        while(best_action >= 0){
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    std::vector<int> CNode::get_children_distribution(){
        // Checked...
        std::vector<int> distribution;
        if(this->expanded()){
            for(int a = 0; a < this->action_num; ++a){
                CNode* child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode* CNode::get_child(int action){
        // Checked...
        int index = this->children_index[action];
        return &((*(this->ptr_node_pool))[index]);
    }

    //*********************************************************

    CRoots::CRoots(){
        this->root_num = 0;
        this->action_num = 0;
        this->pool_size = 0;
    }

    CRoots::CRoots(int root_num, int action_num, int pool_size){
        this->root_num = root_num;
        this->action_num = action_num;
        this->pool_size = pool_size;

        this->node_pools.reserve(root_num);
        this->roots.reserve(root_num);

        for(int i = 0; i < root_num; ++i){
            this->node_pools.push_back(std::vector<CNode>());
            this->node_pools[i].reserve(pool_size);

            this->roots.push_back(CNode(0, action_num, &this->node_pools[i]));
        }
    }

    CRoots::~CRoots(){}

//    void CRoots::prepare(float root_exploration_fraction,
//                         const std::vector<std::vector<float>> &noises,
//                         const std::vector<std::vector<float>> &q_inits,
//                         const std::vector<float> &reward_sums){
//        // Checked.
//        // Expand all the roots.
//
//        for(int i = 0; i < this->root_num; ++i){
//            this->roots[i].expand_q_init(reward_sums[i], i, 0, q_inits[i]);
//            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);
//            this->roots[i].visit_count += 1;
//        }
//    }

//    void CRoots::prepare_attacker(float root_exploration_fraction,
//                                  const std::vector<std::vector<float>> &noises,
//                                  const std::vector<float> &reward_sums,
//                                  const std::vector<std::vector<float>> &policies){
//
//        for(int i = 0; i < this->root_num; ++i){
//            this->roots[i].expand_attacker(reward_sums[i], i, 0, policies[i]);
//            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);
//            this->roots[i].visit_count += 1;
//        }
//    }


//    void CRoots::prepare_no_noise(const std::vector<float> &reward_sums,
//                                  const std::vector<std::vector<float>> &q_inits){
//        // Checked.
//        for(int i = 0; i < this->root_num; ++i){
//            this->roots[i].expand_q_init(reward_sums[i], i, 0, q_inits[i]);
//            this->roots[i].visit_count += 1;
//        }
//    }

//    void CRoots::prepare_attacker_no_noise(const std::vector<float> &reward_sums,
//                                           const std::vector<std::vector<float>> &policies){
//        for(int i = 0; i < this->root_num; ++i){
//            this->roots[i].expand_attacker(reward_sums[i], i, 0, policies[i]);
//            this->roots[i].visit_count += 1;
//        }
//    }

    void CRoots::prepare(float root_exploration_fraction,
                         const std::vector<std::vector<float>> &noises,
                         const std::vector<std::vector<float>> &q_inits,
                         const std::vector<float> &reward_sums,
                         const std::vector<std::vector<float>> &policies,
                         const std::vector<int> &to_plays){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].to_play = to_plays[i];
            if(this->roots[i].to_play){
                this->roots[i].expand_attacker(reward_sums[i], i, 0, policies[i]);
                this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);
            }
            else{
                this->roots[i].expand_q_init(reward_sums[i], i, 0, q_inits[i]);
            }
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &reward_sums,
                                  const std::vector<std::vector<float>> &q_inits,
                                  const std::vector<std::vector<float>> &policies,
                                  const std::vector<int> &to_plays){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].to_play = to_plays[i];
            if(this->roots[i].to_play){
                this->roots[i].expand_attacker(reward_sums[i], i, 0, policies[i]);
            }
            else{
                this->roots[i].expand_q_init(reward_sums[i], i, 0, q_inits[i]);
            }
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear(){
        this->node_pools.clear();
        this->roots.clear();
    }

    std::vector<std::vector<int>> CRoots::get_trajectories(){
    // Checked.
    // Return all the search trajectory.
        std::vector<std::vector<int>> trajs;
        trajs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int>> CRoots::get_distributions(){
    // Checked.
    // Return the child distribution of all the children...
        std::vector<std::vector<int>> distributions;
        distributions.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values(){
    // Checked.
    // Return the values of all the roots..
        std::vector<float> values;
        for(int i = 0; i < this->root_num; ++i){
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    //*********************************************************

    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount){
        std::stack<CNode*> node_stack;
        node_stack.push(root);

        while(node_stack.size() > 0){
            CNode* node = node_stack.top();
            node_stack.pop();

            if(node != root){
                float qsa = node->reward + discount * node->value();
                min_max_stats.update(qsa);
            }

            for(int a = 0; a < node->action_num; ++a){
                CNode* child = node->get_child(a);
                if(child->expanded()){
                    node_stack.push(child);
                }
            }
        }
    }

    void cback_propagate(std::vector<CNode*> &search_path,
                         tools::CMinMaxStats &min_max_stats,
                         float value, float discount){\
        // Checked.
        // Back_propagate one search path...

        float bootstrap_value = value;
        int path_len = search_path.size();

        for(int i = path_len - 1; i >= 0; --i){
            CNode* node = search_path[i];
            node->value_sum += bootstrap_value;
            node->visit_count += 1;
            bootstrap_value = node->reward + discount * bootstrap_value;
        }
        min_max_stats.clear();
        CNode* root = search_path[0];
        update_tree_q(root, min_max_stats, discount);
    }

    void cmulti_back_propagate(int data_idx_1,
                               float discount, const std::vector<float> &rewards,
                               const std::vector<float> &values,
                               tools::CMinMaxStatsList *min_max_stats_lst,
                               CSearchResults &results,
                               const std::vector<int> &to_plays,
                               const std::vector<std::vector<float>> &policies,
                               int add_attacker){

        for(int i = 0; i < results.num; ++i){
            int leaf_to_play = 0;
            if(add_attacker == 1){
                leaf_to_play = 1 - to_plays[i];
            }
            if(leaf_to_play == 0) {
                results.nodes[i]->expand(rewards[i], i, data_idx_1);
            }
            else {
                results.nodes[i]->expand_attacker(rewards[i], i, data_idx_1, policies[i]);
            }
            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], values[i], discount);
        }
    }

    int cselect_child(CNode* root,
                      tools::CMinMaxStats &min_max_stats,
                      int pb_c_base,
                      float pb_c_init,
                      float discount,
                      float mean_q){

    // Checked.
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;
        int is_attacker = 0;
        if (root->to_play == ATTACKER) is_attacker = 1;

        for(int a = 0; a < root->action_num; ++a){
            CNode* child = root->get_child(a);
            float temp_score = cucb_score(child, min_max_stats, root->visit_count - 1, mean_q,
                                          pb_c_base, pb_c_init, discount, is_attacker);

            if(max_score < temp_score){
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if(temp_score >= max_score - epsilon){
                max_index_lst.push_back(a);
            }
        }

        int action = 0;
        if(max_index_lst.size() > 0){
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        else{
            printf("[ERROR] max action list is empty!\n");
            printf("mean_q=%f\n", mean_q);
            printf("root_action_num=%d\n", root->action_num);
            printf("root_visit_num=%d\n", root->visit_count);
            printf("root_idx_0=%d\n", root->data_idx_0);
            printf("root_idx_1=%d\n", root->data_idx_1);
        }
        return action;
    }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats,
                     float total_children_visit_counts,
                     float parent_mean_q,
                     float pb_c_base,
                     float pb_c_init,
                     float discount,
                     int is_attacker){
    // Checked.
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0){
            if (child->use_q_init){
                value_score = child->q_init;
            }
            else{
                value_score = parent_mean_q;
            }
        }

        else {
            float true_reward = child->reward;
            value_score = true_reward + discount * child->value();
        }

        if (is_attacker){
            value_score = -value_score;
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0) value_score = 0;
        if (value_score > 1) value_score = 1;

        float ucb_value = prior_score + value_score;
//        printf("%f\n", ucb_value);
        return ucb_value;
    }

    void cmulti_traverse(CRoots *roots,
                         int pb_c_base,
                         float pb_c_init,
                         float discount,
                         tools::CMinMaxStatsList *min_max_stats_lst,
                         CSearchResults &results){
        // Checked.

        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        // Start searching...
        int last_action = -1;
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        // For each tree root...
        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;

            results.search_paths[i].push_back(node);
            std::vector<CNode>* ptr_node_pool = node->ptr_node_pool;

            for(int a = 0; a < node->action_num; ++a){
                CNode* child = node->get_child(a);
                if (child->use_q_init){
                    min_max_stats_lst->stats_lst[i].update(child->q_init);
                }
            }

            while(node->expanded()){
                float mean_q = node->get_mean_q(is_root, parent_q, discount);
                is_root = 0;
                if(isnan(mean_q)){
                    printf("meanq=%f\n", mean_q);
                }
                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, mean_q);
                parent_q = mean_q;

                node->best_action = action;

                // Goto next node.
                node = node->get_child(action);
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            // The parent of the last node.
            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];
            results.data_idx_0.push_back(parent->data_idx_0);
            results.data_idx_1.push_back(parent->data_idx_1);

            // The id of the last action (done by the parent node).
            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);

            // Remember this node to expand.
            results.nodes.push_back(node);
        }
    }
}