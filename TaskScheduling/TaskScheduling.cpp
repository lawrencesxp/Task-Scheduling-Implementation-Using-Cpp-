// Based on Prof. Xue Lin's “Task Scheduling” paper
// EECE Project 2 Lawrence SXP

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>
#include <limits>    
#include <chrono>
#include <utility>
#include <cmath>
#include <unordered_map>
#include <cstdlib>  

using namespace std;


class Node {
public:
    int id;
    vector<Node*> parents;
    vector<Node*> children;
    vector<int> core_speed;
    vector<int> cloud_speed;
    int remote_execution_time;
    int local_finish_time;
    int ft = 0;
    int wireless_sending_finish_time;
    int cloud_finish_time;
    int wireless_recieving_finish_time;
    int local_ready_time;
    int wireless_sending_ready_time;
    int cloud_ready_time;
    int wireless_recieving_ready_time;
    double priority_score;
    int assignment;
    bool is_core;
    vector<int> start_time;
    int is_scheduled;

    // Constructor
    Node(int id, const vector<int>& core_speed, const vector<int>& cloud_speed,
        const vector<Node*>& parents = vector<Node*>(),
        const vector<Node*>& children = vector<Node*>(),
        int assignment = -2, int local_ready_time = -1,
        int wireless_sending_ready_time = -1, int cloud_ready_time = -1,
        int wireless_recieving_ready_time = -1)
        : id(id), parents(parents), children(children), core_speed(core_speed),
        cloud_speed(cloud_speed), remote_execution_time(accumulate(cloud_speed.begin(), cloud_speed.end(), 0)),
        local_finish_time(0), ft(0), wireless_sending_finish_time(0), cloud_finish_time(0),
        wireless_recieving_finish_time(0), local_ready_time(local_ready_time),
        wireless_sending_ready_time(wireless_sending_ready_time), cloud_ready_time(cloud_ready_time),
        wireless_recieving_ready_time(wireless_recieving_ready_time), priority_score(-1),
        assignment(assignment), is_core(false), start_time(4, -1), is_scheduled(-1) {}


    void print_info() const {
        cout << "Node ID: " << id << "\n";
        cout << "Assignment: " << assignment + 1 << "\n";
        cout << "local ready time: " << local_ready_time << "\n";
        cout << "wireless sending ready time: " << wireless_sending_ready_time << "\n";
        cout << "cloud ready time: " << cloud_ready_time << "\n";
        cout << "wireless recieving ready time: " << wireless_recieving_ready_time << "\n";
        cout << "START time: " << start_time[assignment] << "\n";
        cout << "local FINISH time: " << local_finish_time << "\n";
        cout << "wireless sending FINISH time: " << wireless_sending_finish_time << "\n";
        cout << "cloud FINISH time: " << cloud_finish_time << "\n";
        cout << "wireless recieving FINISH time: " << wireless_recieving_finish_time << "\n\n";
    }

    Node* deep_copy_node(Node* original, unordered_map<Node*, Node*>& copied_nodes) {
        if (original == nullptr) {
            return nullptr;
        }

        auto it = copied_nodes.find(original);
        if (it != copied_nodes.end()) {
            return it->second;
        }

        Node* copy = new Node(original->id, original->core_speed, original->cloud_speed);
        copied_nodes[original] = copy;


        for (Node* parent : original->parents) {
            copy->parents.push_back(deep_copy_node(parent, copied_nodes));
        }
        for (Node* child : original->children) {
            copy->children.push_back(deep_copy_node(child, copied_nodes));
        }


        copy->remote_execution_time = original->remote_execution_time;
        copy->local_finish_time = original->local_finish_time;
        copy->ft = original->ft;
        copy->wireless_sending_finish_time = original->wireless_sending_finish_time;
        copy->cloud_finish_time = original->cloud_finish_time;
        copy->wireless_recieving_finish_time = original->wireless_recieving_finish_time;
        copy->local_ready_time = original->local_ready_time;
        copy->wireless_sending_ready_time = original->wireless_sending_ready_time;
        copy->cloud_ready_time = original->cloud_ready_time;
        copy->wireless_recieving_ready_time = original->wireless_recieving_ready_time;
        copy->priority_score = original->priority_score;
        copy->assignment = original->assignment;
        copy->is_core = original->is_core;
        copy->start_time = original->start_time;
        copy->is_scheduled = original->is_scheduled;

        return copy;
    }
};

class TaskForPlotting {
public:
    int node_id;
    int assignment;
    int cloud_start_time;
    int cloud_finish_time;
    int ws_start_time;
    int ws_finish_time;
    int wr_start_time;
    int wr_finish_time;
    int local_start_time;
    int local_finish_time;
    bool is_core;

    void print() const {
        cout << "Node ID: " << node_id;
        if (is_core) {
            cout << ", Assignment: Core " << assignment << ", Local Start time: " << local_start_time << ", Local Finish time: " << local_finish_time << endl;
        }
        else {
            cout << ", Assignment: Cloud" << ", Cloud Start time: " << cloud_start_time << ", Cloud Finish time: " << cloud_finish_time
                << ", WS Start time: " << ws_start_time << ", WS Finish time: " << ws_finish_time
                << ", WR Start time: " << wr_start_time << ", WR Finish time: " << wr_finish_time << endl;
        }
    }
};

vector<Node*> deep_copy_node_list(const vector<Node*>& original_list) {
    unordered_map<Node*, Node*> copied_nodes_map;
    vector<Node*> copied_list;


    for (Node* node : original_list) {
        Node* copied_node = node->deep_copy_node(node, copied_nodes_map);
        copied_list.push_back(copied_node);
    }

    return copied_list;
}


double total_T(const vector<Node*>& nodes) {
    double total_t = 0;
    for (const auto& node : nodes) {

        if (node->children.empty()) {
            total_t = max(node->local_finish_time, node->wireless_recieving_finish_time);
        }
    }
    return total_t;
}


double total_E(const vector<Node*>& nodes, const vector<double>& core_cloud_power = { 1, 2, 4, 0.5 }) {

    double total_energy = 0;
    for (const auto& node : nodes) {
        double current_node_e = 0;
        if (node->is_core) {

            current_node_e = node->core_speed[node->assignment] * core_cloud_power[node->assignment];
        }
        else {

            current_node_e = node->cloud_speed[0] * core_cloud_power[3];
        }
        total_energy += current_node_e;
    }
    return total_energy;
}



void primary_assignment(vector<Node*>& nodes) {

    for (auto& node : nodes) {
        int t_l_min = *min_element(node->core_speed.begin(), node->core_speed.end());

        if (t_l_min > node->remote_execution_time) {
            node->is_core = true;
        }
        else {
            node->is_core = false;
        }
    }
}

double calculate_priority(Node* task, const vector<Node*>& task_graph, const vector<double>& weights, map<double, double>& priority_cache) {

    if (priority_cache.find(task->id) != priority_cache.end()) {
        return priority_cache[task->id];
    }

    if (task->children.empty()) {
        priority_cache[task->id] = weights[task->id - 1];
        return weights[task->id - 1];
    }

    double max_successor_priority = 0;
    for (const auto& successor : task->children) {
        double successor_priority = calculate_priority(successor, task_graph, weights, priority_cache);
        max_successor_priority = max(max_successor_priority, successor_priority);
    }
    double task_priority = weights[task->id - 1] + max_successor_priority;

    priority_cache[task->id] = task_priority;

    return task_priority;
}

map<double, double> calculate_all_priorities(const vector<Node*>& task_graph, const vector<double>& weights) {
    map<double, double> priority_cache;

    for (const auto& task : task_graph) {

        calculate_priority(task, task_graph, weights, priority_cache);
    }

    return priority_cache;
}



void task_prioritizing(vector<Node*>& nodes) {

    int n = nodes.size();
    vector<double> w(n, 0.0);


    for (int i = 0; i < n; ++i) {
        if (nodes[i]->is_core) {
            w[i] = nodes[i]->remote_execution_time;
        }
        else {
            double sumCoreSpeed = accumulate(nodes[i]->core_speed.begin(), nodes[i]->core_speed.end(), 0);
            w[i] = sumCoreSpeed / nodes[i]->core_speed.size();
        }
    }

    reverse(nodes.begin(), nodes.end());
    map<double, double> priorities = calculate_all_priorities(nodes, w);
    reverse(nodes.begin(), nodes.end());


    for (int i = 0; i < n; ++i) {
        nodes[i]->priority_score = priorities[nodes[i]->id];
    }
}


vector<vector<int>> execution_unit_selection(vector<Node*>& nodes) {
    int k = 3;
    int n = nodes.size();


    vector<int> core1_seq;
    vector<int> core2_seq;
    vector<int> core3_seq;
    vector<int> cloud_seq;

    vector<int> coreEarliestready(k + 1, 0);  


    vector<pair<double, int>> node_priority_list;
    for (const auto& node : nodes) {  
        node_priority_list.emplace_back(node->priority_score, node->id);
    }


    sort(node_priority_list.begin(), node_priority_list.end());

    vector<int> pri_n;
    for (const auto& item : node_priority_list) {
        pri_n.push_back(item.second); 
    }

    for (int a = n - 1; a >= 0; --a) {  
        int i = pri_n[a] - 1;  
        Node* node = nodes[i];

        if (node->parents.empty()) { 
            auto min_load_core_it = min_element(coreEarliestready.begin(), coreEarliestready.end());
            int min_load_core = distance(coreEarliestready.begin(), min_load_core_it);


            node->local_ready_time = coreEarliestready[min_load_core];
            node->wireless_sending_ready_time = coreEarliestready[min_load_core];
            node->wireless_sending_finish_time = node->wireless_sending_ready_time + node->cloud_speed[0];
            node->cloud_ready_time = node->wireless_sending_finish_time;
            coreEarliestready[min_load_core] = node->cloud_ready_time;
        }
        else { 
            int max_j_l = 0;
            int max_j_ws = 0;
            int max_j_c = 0;
            for (const auto& parent : node->parents) {
                max_j_l = max(max_j_l, max(parent->local_finish_time, parent->wireless_recieving_finish_time));
                max_j_ws = max(max_j_ws, max(parent->local_finish_time, parent->wireless_recieving_finish_time));
                max_j_c = max(max_j_c, parent->wireless_recieving_finish_time - node->cloud_speed[2]);
            }
            node->local_ready_time = max_j_l;
            node->wireless_sending_ready_time = max_j_ws;
            node->wireless_sending_finish_time = max(node->wireless_sending_ready_time, coreEarliestready[3]) + node->cloud_speed[0];
            node->cloud_ready_time = max(node->wireless_sending_finish_time, max_j_c);
        }

        if (node->is_core) {

            node->wireless_recieving_ready_time = node->cloud_ready_time + node->cloud_speed[1];
            node->wireless_recieving_finish_time = node->wireless_recieving_ready_time + node->cloud_speed[2];
            node->ft = node->wireless_recieving_finish_time;
            node->local_finish_time = 0;
            coreEarliestready[3] = node->wireless_sending_finish_time;
            node->start_time[3] = node->wireless_sending_ready_time;
            node->assignment = 3;  // Assign to cloud
            node->is_core = false;
            node->is_scheduled = 1;
        }
        else {

            double finish_time = numeric_limits<double>::infinity();
            int index = -1;
            for (int j = 0; j < k; ++j) {
                double ready_time = max(node->local_ready_time, coreEarliestready[j]);
                if (finish_time > ready_time + node->core_speed[j]) {
                    finish_time = ready_time + node->core_speed[j];
                    index = j;
                }
            }
            node->local_ready_time = finish_time - node->core_speed[index];
            node->start_time[index] = node->local_ready_time;
            node->local_finish_time = finish_time;
            node->wireless_recieving_ready_time = node->cloud_ready_time + node->cloud_speed[1];
            node->wireless_recieving_finish_time = node->wireless_recieving_ready_time + node->cloud_speed[2];

            if (node->local_finish_time <= node->wireless_recieving_finish_time) {
                node->ft = node->local_finish_time;
                node->start_time[index] = node->local_ready_time;
                node->wireless_recieving_finish_time = 0;
                coreEarliestready[index] = node->ft;
                node->assignment = index;
                node->is_core = true;
                node->is_scheduled = 1;
            }
            else {
                node->ft = node->wireless_recieving_finish_time;
                node->local_finish_time = 0;
                coreEarliestready[3] = node->ft;
                node->start_time[3] = node->wireless_sending_ready_time;
                node->assignment = 3;  
                node->is_core = false;
                node->is_scheduled = 1;
            }
        }

        if (node->assignment == 0) {
            core1_seq.push_back(node->id);
        }
        else if (node->assignment == 1) {
            core2_seq.push_back(node->id);
        }
        else if (node->assignment == 2) {
            core3_seq.push_back(node->id);
        }
        else if (node->assignment == 3) {
            cloud_seq.push_back(node->id);
        }
    }
    vector<vector<int>> seq = { core1_seq, core2_seq, core3_seq, cloud_seq };
    return seq;
}


vector<vector<int>> new_sequence(vector<Node*>& nodes, int targetNodeId, int targetLocation, vector<vector<int>>& seq) {

    map<int, int> nodeIdToIndexMap; 
    Node* target_node = nullptr;
    for (int i = 0; i < nodes.size(); ++i) {
        nodeIdToIndexMap[nodes[i]->id] = i;
  
        if (nodes[i]->id == targetNodeId) {
            target_node = nodes[i];
        }
    }

    int target_node_rt = target_node->is_core ? target_node->local_ready_time : target_node->wireless_sending_ready_time;


    auto& original_seq = seq[target_node->assignment];
    original_seq.erase(remove(original_seq.begin(), original_seq.end(), targetNodeId), original_seq.end());

    vector<int>& s_new = seq[targetLocation]; 
    vector<int> s_new_prim; 
    bool flag = false;
    for (int _node_id : s_new) {
        Node* node = nodes[nodeIdToIndexMap[_node_id]];
        if (node->start_time[targetLocation] < target_node_rt) {
            s_new_prim.push_back(node->id);
        }
        if (node->start_time[targetLocation] >= target_node_rt && !flag) {
            s_new_prim.push_back(target_node->id);
            flag = true;
        }
        if (node->start_time[targetLocation] >= target_node_rt && flag) {
            s_new_prim.push_back(node->id);
        }
    }
    if (!flag) {
     
        s_new_prim.push_back(target_node->id);
    }

    s_new = s_new_prim;

    target_node->assignment = targetLocation;

    target_node->is_core = targetLocation != 3;

    return seq;
}


tuple<vector<int>, vector<int>, vector<int>, vector<int>, vector<Node*>>
initialize_kernel(const vector<Node*>& updated_node_list, const vector<vector<int>>& updated_seq) {

    vector<int> localCorereadytimes = { 0, 0, 0 };
    vector<int> cloudStagereadytimes = { 0, 0, 0 };
    
    vector<int> dependencyReadiness(updated_node_list.size(), -1); 
    vector<int> sequenceReadiness(updated_node_list.size(), -1); 
    dependencyReadiness[updated_node_list[0]->id - 1] = 0; 
    for (const auto& each_seq : updated_seq) {
        if (!each_seq.empty()) {
            sequenceReadiness[each_seq[0] - 1] = 0; 
        }
    }

    map<int, int> node_index;
    for (int i = 0; i < updated_node_list.size(); ++i) {
        node_index[updated_node_list[i]->id] = i;

        updated_node_list[i]->local_ready_time = updated_node_list[i]->wireless_sending_ready_time =
            updated_node_list[i]->cloud_ready_time = updated_node_list[i]->wireless_recieving_ready_time = -1;
    }

    vector<Node*> stack;
    stack.push_back(updated_node_list[0]);  

    return { localCorereadytimes, cloudStagereadytimes, dependencyReadiness, sequenceReadiness, stack };
}


void calculate_and_schedule_node(Node* currentNode, vector<int>& localCorereadytimes, vector<int>& cloudStagereadytimes) {

    if (currentNode->is_core) {
        currentNode->local_ready_time = 0; 
        if (!currentNode->parents.empty()) {
    
            for (auto& parent : currentNode->parents) {
                int p_ft = max(parent->local_finish_time, parent->wireless_recieving_finish_time);
                if (p_ft > currentNode->local_ready_time) {
                    currentNode->local_ready_time = p_ft;
                }
            }
        }
    }

    if (currentNode->assignment >= 0 && currentNode->assignment <= 2) { 
        currentNode->start_time = vector<int>(4, -1);
        int core_index = currentNode->assignment;
        currentNode->start_time[core_index] = max(localCorereadytimes[core_index], currentNode->local_ready_time);
        currentNode->local_finish_time = currentNode->start_time[core_index] + currentNode->core_speed[core_index];

        currentNode->wireless_sending_finish_time = currentNode->cloud_finish_time = currentNode->wireless_recieving_finish_time = -1;
        localCorereadytimes[core_index] = currentNode->local_finish_time; 
    }

    if (currentNode->assignment == 3) { 

        currentNode->wireless_sending_ready_time = 0;
        if (!currentNode->parents.empty()) {
            for (auto& parent : currentNode->parents) {
                int p_ws = max(parent->local_finish_time, parent->wireless_sending_finish_time);
                if (p_ws > currentNode->wireless_sending_ready_time) {
                    currentNode->wireless_sending_ready_time = p_ws;
                }
            }
        }
        currentNode->wireless_sending_finish_time = max(cloudStagereadytimes[0], currentNode->wireless_sending_ready_time) + currentNode->cloud_speed[0];
        currentNode->start_time[3] = max(cloudStagereadytimes[0], currentNode->wireless_sending_ready_time);
        cloudStagereadytimes[0] = currentNode->wireless_sending_finish_time;

        int p_max_ft_c = 0;
        for (auto& parent : currentNode->parents) {
            p_max_ft_c = max(p_max_ft_c, parent->cloud_finish_time);
        }
        currentNode->cloud_ready_time = max(currentNode->wireless_sending_finish_time, p_max_ft_c);
        currentNode->cloud_finish_time = max(cloudStagereadytimes[1], currentNode->cloud_ready_time) + currentNode->cloud_speed[1];
        cloudStagereadytimes[1] = currentNode->cloud_finish_time;

        currentNode->wireless_recieving_ready_time = currentNode->cloud_finish_time;
        currentNode->wireless_recieving_finish_time = max(cloudStagereadytimes[2], currentNode->wireless_recieving_ready_time) + currentNode->cloud_speed[2];
        currentNode->local_finish_time = -1; 
        cloudStagereadytimes[2] = currentNode->wireless_recieving_finish_time;
    }
}

void update_readiness_and_stack(Node* currentNode, vector<Node*>& updated_node_list,
    const vector<vector<int>>& updated_seq,
    vector<int>& dependencyReadiness, vector<int>& sequenceReadiness,
    vector<Node*>& stack) {

    const auto& corresponding_seq = updated_seq[currentNode->assignment];
    auto it = find(corresponding_seq.begin(), corresponding_seq.end(), currentNode->id);
    int currentNode_index = distance(corresponding_seq.begin(), it);
    int next_node_id = (currentNode_index != corresponding_seq.size() - 1) ? corresponding_seq[currentNode_index + 1] : -1;

    for (auto& node : updated_node_list) {
        int flag = count_if(node->parents.begin(), node->parents.end(), [](Node* parent) {
            return parent->is_scheduled != 2;
            });
        dependencyReadiness[node->id - 1] = flag;
        if (node->id == next_node_id) {
            sequenceReadiness[node->id - 1] = 0;
        }
    }

    for (auto& node : updated_node_list) {
        auto stack_it = find(stack.begin(), stack.end(), node);
        if (dependencyReadiness[node->id - 1] == 0 && sequenceReadiness[node->id - 1] == 0
            && node->is_scheduled != 2 && stack_it == stack.end()) {
            stack.push_back(node);
        }
    }
}

vector<Node*> kernel_algorithm(vector<Node*>& updated_node_list, const vector<vector<int>>& updated_seq) {
 
    vector<int> localCorereadytimes, cloudStagereadytimes, dependencyReadiness, sequenceReadiness;
    vector<Node*> stack;
    tie(localCorereadytimes, cloudStagereadytimes, dependencyReadiness, sequenceReadiness, stack) = initialize_kernel(updated_node_list, updated_seq);


    while (stack.size() != 0) {
        Node* currentNode = stack.back();
        stack.pop_back();  // Pop the last node from the stack
        currentNode->is_scheduled = 2;  // Mark the node as scheduled
        calculate_and_schedule_node(currentNode, localCorereadytimes, cloudStagereadytimes);
        update_readiness_and_stack(currentNode, updated_node_list, updated_seq, dependencyReadiness, sequenceReadiness, stack);
    }

    for (auto& node : updated_node_list) {
        node->is_scheduled = -1;
    }

    return updated_node_list;
}

int main() {
    // initialize nodes with specific IDs, parents, children, core and cloud speeds

    //// Test 1
    // Node node10(10, {7, 4, 2}, {3, 1, 1});
    // Node node9(9, {5, 3, 2}, {3, 1, 1});
    // Node node8(8, {6, 4, 2}, {3, 1, 1});
    // Node node7(7, {8, 5, 3}, {3, 1, 1});
    // Node node6(6, {7, 6, 4}, {3, 1, 1});
    // Node node5(5, {5, 4, 2}, {3, 1, 1});
    // Node node4(4, {7, 5, 3}, {3, 1, 1});
    // Node node3(3, {6, 5, 4}, {3, 1, 1});
    // Node node2(2, {8, 6, 5}, {3, 1, 1});
    // Node node1(1, {9, 7, 5}, {3, 1, 1});
    // node1.children = {&node2, &node3, &node4, &node5, &node6};
    // node2.parents = {&node1}; node2.children = {&node8, &node9};
    // node3.parents = {&node1}; node3.children = {&node7};
    // node4.parents = {&node1}; node4.children = {&node8, &node9};
    // node5.parents = {&node1}; node5.children = {&node9};
    // node6.parents = {&node1}; node6.children = {&node8};
    // node7.parents = {&node3}; node7.children = {&node10};
    // node8.parents = {&node2, &node4, &node6}; node8.children = {&node10};
    // node9.parents = {&node2, &node4, &node5}; node9.children = {&node10};
    // node10.parents = {&node7, &node8, &node9};

    // vector<Node*> node_list = {&node1, &node2, &node3, &node4, &node5, &node6, &node7, &node8, &node9, &node10};

    // Test 2
     Node node10(10, {7, 4, 2}, {3, 1, 1});
     Node node9(9, {5, 3, 2}, {3, 1, 1});
     Node node8(8, {6, 4, 2}, {3, 1, 1});
     Node node7(7, {8, 5, 3}, {3, 1, 1});
     Node node6(6, {7, 6, 4}, {3, 1, 1});
     Node node5(5, {5, 4, 2}, {3, 1, 1});
     Node node4(4, {7, 5, 3}, {3, 1, 1});
     Node node3(3, {6, 5, 4}, {3, 1, 1});
     Node node2(2, {8, 6, 5}, {3, 1, 1});
     Node node1(1, {9, 7, 5}, {3, 1, 1});

     node1.parents = {}; node1.children = {&node2, &node3, &node4};
     node2.parents = {&node1};node2.children = {&node5, &node7};
     node3.parents = {&node1};node3.children = {&node7, &node8};
     node4.parents = {&node1};node4.children = {&node7, &node8};
     node5.parents = {&node2};node5.children = {&node6};
     node6.parents = {&node5};node6.children = {&node10};
     node7.parents = {&node2, &node3, &node4};node7.children = {&node9, &node10};
     node8.parents = {&node3, &node4};node8.children = {&node9};
     node9.parents = {&node7, &node8};node9.children = {&node10};
     node10.parents = {&node6, &node7, &node9};node10.children = {};

     vector<Node*> node_list = {&node1, &node2, &node3, &node4, &node5, &node6, &node7, &node8, &node9, &node10};

    // Test 3
   /*  Node node20(20, {12, 5, 4}, {3, 1, 1});
     Node node19(19, {10, 5, 3}, {3, 1, 1});
     Node node18(18, {13, 9, 2}, {3, 1, 1});
     Node node17(17, {9, 3, 3}, {3, 1, 1});
     Node node16(16, {9, 7, 3}, {3, 1, 1});
     Node node15(15, {13, 4, 2}, {3, 1, 1});
     Node node14(14, {12, 11, 4}, {3, 1, 1});
     Node node13(13, {11, 3, 2}, {3, 1, 1});
     Node node12(12, {12, 8, 4}, {3, 1, 1});
     Node node11(11, {12, 3, 3}, {3, 1, 1});
     Node node10(10, {7, 4, 2}, {3, 1, 1});
     Node node9(9, {5, 3, 2}, {3, 1, 1});
     Node node8(8, {6, 4, 2}, {3, 1, 1});
     Node node7(7, {8, 5, 3}, {3, 1, 1});
     Node node6(6, {7, 6, 4}, {3, 1, 1});
     Node node5(5, {5, 4, 2}, {3, 1, 1});
     Node node4(4, {7, 5, 3}, {3, 1, 1});
     Node node3(3, {6, 5, 4}, {3, 1, 1});
     Node node2(2, {8, 6, 5}, {3, 1, 1});
     Node node1(1, {9, 7, 5}, {3, 1, 1});
     node1.parents = {};node1.children = {&node2, &node3, &node4, &node5, &node6};
     node2.parents = {&node1};node2.children = {&node7};
     node3.parents = {&node1};node3.children = {&node7, &node8};
     node4.parents = {&node1};node4.children = {&node8, &node9};
     node5.parents = {&node1};node5.children = {&node9, &node10};
     node6.parents = {&node1};node6.children = {&node10, &node11};
     node7.parents = {&node2, &node3};node7.children = {&node12};
     node8.parents = {&node3, &node4};node8.children = {&node12, &node13};
     node9.parents = {&node4, &node5};node9.children = {&node13, &node14};
     node10.parents = {&node5, &node6};node10.children = {&node11, &node15};
     node11.parents = {&node6, &node10};node11.children = {&node15, &node16};
     node12.parents = {&node7, &node8};node12.children = {&node17};
     node13.parents = {&node8, &node9};node13.children = {&node17, &node18};
     node14.parents = {&node9, &node10};node14.children = {&node18, &node19};
     node15.parents = {&node10, &node11};node15.children = {&node19};
     node16.parents = {&node11};node16.children = {&node19};
     node17.parents = {&node12, &node13};node17.children = {&node20};
     node18.parents = {&node13, &node14};node18.children = {&node20};
     node19.parents = {&node14, &node15,&node16};node19.children = {&node20};
     node20.parents = {&node17, &node18,&node19};node20.children = {};

     vector<Node*> node_list = {&node1, &node2, &node3, &node4, &node5, &node6, &node7, &node8, &node9, &node10, &node11, &node12, &node13, &node14, &node15, &node16, &node17, &node18, &node19, &node20};*/

    // Test 4
   /*  Node node20(20, {12, 5, 4}, {3, 1, 1});
     Node node19(19, {10, 5, 3}, {3, 1, 1});
     Node node18(18, {13, 9, 2}, {3, 1, 1});
     Node node17(17, {9, 3, 3}, {3, 1, 1});
     Node node16(16, {9, 7, 3}, {3, 1, 1});
     Node node15(15, {13, 4, 2}, {3, 1, 1});
     Node node14(14, {12, 11, 4}, {3, 1, 1});
     Node node13(13, {11, 3, 2}, {3, 1, 1});
     Node node12(12, {12, 8, 4}, {3, 1, 1});
     Node node11(11, {12, 3, 3}, {3, 1, 1});
     Node node10(10, {7, 4, 2}, {3, 1, 1});
     Node node9(9, {5, 3, 2}, {3, 1, 1});
     Node node8(8, {6, 4, 2}, {3, 1, 1});
     Node node7(7, {8, 5, 3}, {3, 1, 1});
     Node node6(6, {7, 6, 4}, {3, 1, 1});
     Node node5(5, {5, 4, 2}, {3, 1, 1});
     Node node4(4, {7, 5, 3}, {3, 1, 1});
     Node node3(3, {6, 5, 4}, {3, 1, 1});
     Node node2(2, {8, 6, 5}, {3, 1, 1});
     Node node1(1, {9, 7, 5}, {3, 1, 1});
     node1.parents = {}; node1.children = {&node7};
     node2.parents = {}; node2.children = {&node7, &node8};
     node3.parents = {}; node3.children = {&node7, &node8};
     node4.parents = {}; node4.children = {&node8, &node9};
     node5.parents = {}; node5.children = {&node9, &node10};
     node6.parents = {}; node6.children = {&node10, &node11};
     node7.parents = {&node1, &node2, &node3}; node7.children = {&node12};
     node8.parents = {&node3, &node4}; node8.children = {&node12, &node13};
     node9.parents = {&node4, &node5}; node9.children = {&node13, &node14};
     node10.parents = {&node5, &node6}; node10.children = {&node11, &node15};
     node11.parents = {&node6, &node10}; node11.children = {&node15, &node16};
     node12.parents = {&node7, &node8}; node12.children = {&node17};
     node13.parents = {&node8, &node9}; node13.children = {&node17, &node18};
     node14.parents = {&node9, &node10}; node14.children = {&node18, &node19};
     node15.parents = {&node10, &node11}; node15.children = {&node19};
     node16.parents = {&node11}; node16.children = {&node19};
     node17.parents = {&node12, &node13}; node17.children = {&node20};
     node18.parents = {&node13, &node14}; node18.children = {&node20};
     node19.parents = {&node14, &node15, &node16}; node19.children = {&node20};
     node20.parents = {&node17, &node18, &node19}; node20.children = {};

     vector<Node*> node_list = {&node1, &node2, &node3, &node4, &node5, &node6, &node7, &node8, &node9, &node10, &node11, &node12, &node13, &node14, &node15, &node16, &node17, &node18, &node19, &node20};*/

    // Test 5
   /* Node node20(20, { 12, 5, 4 }, { 3, 1, 1 });
    Node node19(19, { 10, 5, 3 }, { 3, 1, 1 });
    Node node18(18, { 13, 9, 2 }, { 3, 1, 1 });
    Node node17(17, { 9, 3, 3 }, { 3, 1, 1 });
    Node node16(16, { 9, 7, 3 }, { 3, 1, 1 });
    Node node15(15, { 13, 4, 2 }, { 3, 1, 1 });
    Node node14(14, { 12, 11, 4 }, { 3, 1, 1 });
    Node node13(13, { 11, 3, 2 }, { 3, 1, 1 });
    Node node12(12, { 12, 8, 4 }, { 3, 1, 1 });
    Node node11(11, { 12, 3, 3 }, { 3, 1, 1 });
    Node node10(10, { 7, 4, 2 }, { 3, 1, 1 });
    Node node9(9, { 5, 3, 2 }, { 3, 1, 1 });
    Node node8(8, { 6, 4, 2 }, { 3, 1, 1 });
    Node node7(7, { 8, 5, 3 }, { 3, 1, 1 });
    Node node6(6, { 7, 6, 4 }, { 3, 1, 1 });
    Node node5(5, { 5, 4, 2 }, { 3, 1, 1 });
    Node node4(4, { 7, 5, 3 }, { 3, 1, 1 });
    Node node3(3, { 6, 5, 4 }, { 3, 1, 1 });
    Node node2(2, { 8, 6, 5 }, { 3, 1, 1 });
    Node node1(1, { 9, 7, 5 }, { 3, 1, 1 });
    node1.parents = {}; node1.children = { &node7 };
    node2.parents = {}; node2.children = { &node7, &node8 };
    node3.parents = {}; node3.children = { &node7, &node8 };
    node4.parents = {}; node4.children = { &node8, &node9 };
    node5.parents = {}; node5.children = { &node9, &node10 };
    node6.parents = {}; node6.children = { &node10, &node11 };
    node7.parents = { &node1, &node2, &node3 }; node7.children = { &node12 };
    node8.parents = { &node3, &node4 }; node8.children = { &node12, &node13 };
    node9.parents = { &node4, &node5 }; node9.children = { &node13, &node14 };
    node10.parents = { &node5, &node6 }; node10.children = { &node11, &node15 };
    node11.parents = { &node6, &node10 }; node11.children = { &node15, &node16 };
    node12.parents = { &node7, &node8 }; node12.children = { &node17 };
    node13.parents = { &node8, &node9 }; node13.children = { &node17, &node18 };
    node14.parents = { &node9, &node10 }; node14.children = { &node18, &node19 };
    node15.parents = { &node10, &node11 }; node15.children = { &node19 };
    node16.parents = { &node11 }; node16.children = { &node19 };
    node17.parents = { &node12, &node13 }; node17.children = {};
    node18.parents = { &node13, &node14 }; node18.children = {};
    node19.parents = { &node14, &node15, &node16 }; node19.children = {};
    node20.parents = { &node12 }; node20.children = {};

    vector<Node*> node_list = { &node1, &node2, &node3, &node4, &node5, &node6, &node7, &node8, &node9, &node10, &node11, &node12, &node13, &node14, &node15, &node16, &node17, &node18, &node19, &node20 };*/


    auto start = chrono::high_resolution_clock::now();


    primary_assignment(node_list);
    task_prioritizing(node_list);
    auto sequence = execution_unit_selection(node_list);


    vector<TaskForPlotting> tasksForPlottinginitial;
    for (const auto& node : node_list) {
        TaskForPlotting task;
        task.node_id = node->id;
        task.assignment = node->assignment + 1;
        task.is_core = node->is_core;

        if (!node->is_core) {
            task.cloud_start_time = node->cloud_ready_time;
            task.cloud_finish_time = node->cloud_ready_time + node->cloud_speed[1];
            task.ws_start_time = node->wireless_sending_ready_time;
            task.ws_finish_time = node->wireless_sending_ready_time + node->cloud_speed[0];
            task.wr_start_time = node->wireless_recieving_ready_time;
            task.wr_finish_time = node->wireless_recieving_ready_time + node->cloud_speed[2];
        }
        else {
            task.local_start_time = node->start_time[node->assignment];
            task.local_finish_time = node->start_time[node->assignment] + node->core_speed[node->assignment];
        }
        tasksForPlottinginitial.push_back(task);
    }

    double T_init_pre_kernel = total_T(node_list);
    double T_init = T_init_pre_kernel;
    double E_init_pre_kernel = total_E(node_list, { 1, 2, 4, 0.5 });
    double E_init = E_init_pre_kernel;

    cout << "initial time: " << T_init_pre_kernel << endl;
    cout << "initial energy: " << E_init_pre_kernel << endl;

    for (const auto& task : tasksForPlottinginitial) {
        task.print();
    }


    int iter_num = 0;

    double T_max_constraint = T_init_pre_kernel * 1.5;

    while (iter_num < 100) {

        cout << string(80, '-') << endl;
        cout << "iter: " << iter_num << endl;


        double T_init = total_T(node_list);
        double E_init = total_E(node_list, { 1, 2, 4, 0.5 });
        cout << "initial time: " << T_init << endl;
        cout << "initial energy: " << E_init << endl;
        cout << string(80, '-') << endl;


        vector<vector<int>> migeff_ratio_choice(node_list.size(), vector<int>(4, 0));
        for (size_t i = 0; i < node_list.size(); ++i) {

            if (node_list[i]->assignment == 3) {

                fill(migeff_ratio_choice[i].begin(), migeff_ratio_choice[i].end(), 1);
            }
            else {

                migeff_ratio_choice[i][node_list[i]->assignment] = 1;
            }
        }

        vector<vector<pair<double, double>>> result_table(node_list.size(), vector<pair<double, double>>(4, make_pair(-1.0, -1.0)));
        for (size_t n = 0; n < migeff_ratio_choice.size(); ++n) {
            auto& nth_row = migeff_ratio_choice[n];
            for (size_t k = 0; k < nth_row.size(); ++k) {
                if (nth_row[k] == 1) {
                    continue;
                }

                auto seq_copy = sequence;
                vector<Node*> nodes_copy = deep_copy_node_list(node_list);

                seq_copy = new_sequence(nodes_copy, n + 1, k, seq_copy);
                kernel_algorithm(nodes_copy, seq_copy);

                double current_T = total_T(nodes_copy);
                double current_E = total_E(nodes_copy);

                result_table[n][k] = make_pair(current_T, current_E);
                for (Node* node : nodes_copy) {
                    delete node;
                }
            }
        }

        int n_best = -1, k_best = -1;
        double T_best = T_init, E_best = E_init;
        double eff_ratio_best = -1;

        for (size_t i = 0; i < result_table.size(); ++i) {
            for (size_t j = 0; j < result_table[i].size(); ++j) {
                auto val = result_table[i][j];
                if (val == make_pair(-1.0, -1.0) || val.first > T_max_constraint) {
                    continue;
                }

                double eff_ratio = (E_best - val.second) / (abs(val.first - T_best) + 0.00005);
                if (eff_ratio > eff_ratio_best) {
                    eff_ratio_best = eff_ratio;
                    n_best = i;
                    k_best = j;
                }
            }
        }

        if (n_best == -1 && k_best == -1) {
            break;
        }

        n_best += 1;
        k_best += 1;
        T_best = result_table[n_best - 1][k_best - 1].first;
        E_best = result_table[n_best - 1][k_best - 1].second;
        cout << "\ncurrent migration: task: " << n_best << ", k: " << k_best
            << ", total time: " << T_best << ", total energy: " << E_best << endl;

        cout << "\nupdate after current outer loop" << endl;
        sequence = new_sequence(node_list, n_best, k_best - 1, sequence);
        kernel_algorithm(node_list, sequence);

        for (const auto& s : sequence) {
            cout << '[';
            for (const auto& i : s) {
                cout << i << ' ';
            }
            cout << ']' << endl;
        }
        double T_current = total_T(node_list);
        double E_current = total_E(node_list, { 1, 2, 4, 0.5 });

        double E_diff = E_init - E_current;
        double T_diff = abs(T_current - T_init);

        iter_num += 1;

        cout << "\npost migration time: " << T_current << endl;
        cout << "post migration energy: " << E_current << endl;

        if (E_diff <= 1) {
            break;
        }
    }

    auto elapsed = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count();

    vector<TaskForPlotting> tasksForPlottingfinal;

    cout << "\n\nRESCHEDULING FINISHED\n\n";

    for (const auto& node : node_list) {
        TaskForPlotting task;
        task.node_id = node->id;
        task.assignment = node->assignment + 1;
        task.is_core = node->is_core;

        if (node->is_core) {
            task.local_start_time = node->start_time[node->assignment];
            task.local_finish_time = node->start_time[node->assignment] + node->core_speed[node->assignment];
        }
        else {
            task.cloud_start_time = node->cloud_ready_time;
            task.cloud_finish_time = node->cloud_ready_time + node->cloud_speed[1];
            task.ws_start_time = node->start_time[3];
            task.ws_finish_time = node->start_time[3] + node->cloud_speed[0];
            task.wr_start_time = node->wireless_recieving_ready_time;
            task.wr_finish_time = node->wireless_recieving_ready_time + node->cloud_speed[2];
        }

        tasksForPlottingfinal.push_back(task);
    }

    // Printing 
    for (const auto& task : tasksForPlottingfinal) {
        task.print();
    }

    cout << "\ntime to run on machine: " << elapsed << " milliseconds" << endl;
    cout << "final sequence: " << endl;
    for (const auto& s : sequence) {
        cout << "[";
        for (const auto& i : s) {
            cout << i << " ";
        }
        cout << "]" << endl;
    }

    double T_final = total_T(node_list);
    double E_final = total_E(node_list, { 1, 2, 4, 0.5 });

    cout << "\ninitial time: " << T_init_pre_kernel << "\ninitial energy: " << E_init_pre_kernel << "\n\n";
    cout << "final time: " << T_final << "\nfinal energy: " << E_final << endl;

    return 0;
}