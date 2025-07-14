import json
from utilis import get_test_data, get_saved_model_details, get_model
import time
import numpy as np
from utilis import create_single_rule, create_sparse_rule_query
from utilis import get_feat_dict
from rec_metrics import RecMetrics
import random
from collections import Counter
import copy
import argparse

parser = argparse.ArgumentParser(description='Testing for Rec')
parser.add_argument('--config', required=True, type=str, help='config file')


def sample_one_direction(test_info, sampling_prob, sampled_users_str):
    while True:
        temp_sample = []
        for s in test_info['sensitive_features']:
            index = list(range(len(sampling_prob[s])))
            probability_index = random.choices(index, weights=sampling_prob[s], k=1)[0]
            temp_sample.append(probability_index)

        temp_sample_str = "".join([str(x) for x in temp_sample])

        # determine whether the sample is already in the sampled_users
        if temp_sample_str not in sampled_users_str:
            sampled_users_str.append(temp_sample_str)
            break

    return temp_sample


def run_score(test_info, temp_sample):
    global query_count, query_time, eval_time

    start_query_time = time.time()
    rule = create_single_rule(test_info['sensitive_features'], np.array(temp_sample))
    rule_query = create_sparse_rule_query(rule)
    group_target = test_data.query(rule_query)
    query_count += 1
    query_time = query_time + (time.time() - start_query_time)

    if group_target['user_id'].nunique() < test_info['threshold_num']:
        return None

    start_eval_time = time.time()

    test_input = {name: group_target[name].values for name in train_features}
    y_pred = model.predict(test_input)
    g = group_target.copy()
    g.loc[:, 'predict'] = np.array(y_pred)
    metric = test_info['metric']
    if metric == 'mrr':
        score_target = RecMetrics(g).mrr_score()
    if metric == 'auc':
        score_target = RecMetrics(g).auc_score()
    if metric == 'ndcg':
        score_target = RecMetrics(g).ndcg_score()
    if metric == 'urd':
        score_target = RecMetrics(g).cate_diversity_score()
    if metric == 'urp':
        score_target = RecMetrics(g).popularity_score()
    eval_time = eval_time + (time.time() - start_eval_time)

    return score_target


def count_sensitive_one_direction(test_info, index, sampled_users):
    sensitive_count = {}

    for i in range(len(test_info['sensitive_features'])):
        sensitive_count[i] = []

    valid_users = []

    for i in index:
        valid_users.append(sampled_users[i])
        for j in range(len(test_info['sensitive_features'])):
            sensitive_count[j].append(sampled_users[i][j])

    return valid_users, sensitive_count


def local_search(test_info, feat_dict, temp_sample, score_target, direction, sampled_users_str, sampled_users,
                     sampled_score):
    """
    Local search.

    Args:
    test_info：dict, including test_data, sensitive_features,metric,threshold,test_model and train_features.
    feat_dict: dict, including the number of encodings for each training feature.
    temp_sample: list, the current sample.
    score_target: float, The effectiveness score of the current sampled user group.
    direction: str, the direction of search, 'min' or 'max'.
    sampled_users_str: list, the list of sampled users represented by str.
    sampled_users: list, the list of sampled users.
    sampled_score: list, the list of the effectiveness score of sampled users.

    Return:
    temp_sample: list, the current sample.
    score_target: float, The effectiveness score of the current sampled user group.
    """
    local_point = []
    local_point_score = []
    change_index = 0
    for s in test_info['sensitive_features']:
        # Randomly select 5 numbers from 0 to feat_dict[s] - 1
        if feat_dict[s] - 1 <= 5:
            change_value_list = [i for i in range(feat_dict[s] - 1)]
        else:
            change_value_list = random.sample(range(0, feat_dict[s] - 1), 5)

        for change_value in change_value_list:
            temp_sample_copy = copy.deepcopy(temp_sample)
            temp_sample_copy[change_index] = change_value

            temp_sample_str = "".join([str(x) for x in temp_sample_copy])

            # determine whether the sample is already in the sampled_users
            if temp_sample_str in sampled_users_str:
                continue

            sampled_users_str.append(temp_sample_str)

            score_target_copy = run_score(test_info, temp_sample_copy)
            if score_target_copy is None:
                continue
            else:
                local_point.append(temp_sample_copy)
                local_point_score.append(score_target_copy)

                sampled_score.append(score_target_copy)
                sampled_users.append(temp_sample_copy)
                if (direction == 'min' and score_target_copy < score_target) or (
                        direction == 'max' and score_target_copy > score_target):
                    temp_sample = copy.deepcopy(temp_sample_copy)
                    score_target = score_target_copy

        change_index = change_index + 1

    # print(direction)
    # print(local_point)
    # print(local_point_score)

    return temp_sample, score_target


def sample_group(test_info, sampling_prob_min, sampling_prob_max, sample_size, sampled_users, sampled_score,
                 sampled_users_str):
    global query_count, query_time, eval_time
    query_count = 0
    while len(sampled_users) < (sample_size - algorithm_params['one_step_num']):
        temp_sample_min = sample_one_direction(test_info, sampling_prob_min, sampled_users_str)

        score_target_min = run_score(test_info, temp_sample_min)
        if score_target_min is None:
            continue

        sampled_score.append(score_target_min)
        sampled_users.append(temp_sample_min)

        if score_target_min <= min(sampled_score):
            local_search(test_info, feat_dict, temp_sample_min,
                             score_target_min, 'min', sampled_users_str,
                             sampled_users, sampled_score)

    print("sample size: ", len(sampled_users))
    print("======begin search max===================================")

    while (len(sampled_users) < sample_size):
        temp_sample_max = sample_one_direction(test_info, sampling_prob_max, sampled_users_str)

        score_target_max = run_score(test_info, temp_sample_max)
        if score_target_max is None:
            continue

        sampled_score.append(score_target_max)
        sampled_users.append(temp_sample_max)

        if score_target_max >= max(sampled_score):
            local_search(test_info, feat_dict, temp_sample_max,
                             score_target_max, 'max', sampled_users_str,
                             sampled_users, sampled_score)

    print("sample size: ", len(sampled_users))
    print("query count: ", query_count)

    print("query time: ", query_time)
    print("eval time: ", eval_time)

    # get the median of the score_list, and then get the index of the value samller than the median
    print(sampled_score)
    median = np.median(sampled_score)
    print("median: ", median)
    index_min = [i for i, x in enumerate(sampled_score) if x < median]
    index_max = [i for i, x in enumerate(sampled_score) if x > median]

    valid_users_min, sensitive_count_min = count_sensitive_one_direction(test_info, index_min, sampled_users)
    valid_users_max, sensitive_count_max = count_sensitive_one_direction(test_info, index_max, sampled_users)

    print("max score: ", max(sampled_score))
    print("min score: ", min(sampled_score))

    unfairness_list.append(max(sampled_score) - min(sampled_score))

    return valid_users_min, sensitive_count_min, valid_users_max, sensitive_count_max


def update_prob_one_direction(test_info, valid_users, sensitive_count, sampling_prob):
    index = 0
    for s in test_info['sensitive_features']:
        sampled_prob = 0
        for v in set(sensitive_count[index]):
            sampled_prob = sampled_prob + sampling_prob[s][v]

        for i in range(feat_dict[s]):
            all_valid = Counter(np.array(sensitive_count[index]))[i]
            if all_valid != 0:
                sampling_prob[s][i] = (all_valid / len(valid_users)) * sampled_prob
        index = index + 1

    return sampling_prob


def update_prob(test_info, valid_users_min, sensitive_count_min, sampling_prob_min, valid_users_max,
                sensitive_count_max, sampling_prob_max):
    sampling_prob_min = update_prob_one_direction(test_info, valid_users_min, sensitive_count_min, sampling_prob_min)
    sampling_prob_max = update_prob_one_direction(test_info, valid_users_max, sensitive_count_max, sampling_prob_max)

    return sampling_prob_min, sampling_prob_max


def one_step(test_info, sampling_prob_min, sampling_prob_max, sample_size, sampled_users, sampled_score,
             sampled_users_str):
    valid_users_min, sensitive_count_min, valid_users_max, sensitive_count_max = sample_group(test_info,
                                                                                              sampling_prob_min,
                                                                                              sampling_prob_max,
                                                                                              sample_size,
                                                                                              sampled_users,
                                                                                              sampled_score,
                                                                                              sampled_users_str)

    print("begin update sampling_prob===========")

    print("sensitive_count_min", sensitive_count_min)
    print("sensitive_count_max", sensitive_count_max)

    start_update_time = time.time()

    sampling_prob_min, sampling_prob_max = update_prob(test_info, valid_users_min, sensitive_count_min,
                                                       sampling_prob_min, valid_users_max, sensitive_count_max,
                                                       sampling_prob_max)

    print("update time:", time.time() - start_update_time)

    return sampling_prob_min, sampling_prob_max


def initialize_filter(test_info, sampling_prob):
    for s in test_info['sensitive_features']:
        sampling_prob[s] = []
        if s == "gender" or s == "onehot_feat0":
            continue
        else:
            for i in range(feat_dict[s]):
                rule = create_single_rule([s], np.array([i]))
                rule_query = create_sparse_rule_query(rule)
                group_target = test_data.query(rule_query)

                if group_target['user_id'].nunique() < test_info['threshold_num']:
                    sampling_prob[s].append(0)
                else:
                    sampling_prob[s].append(1 / feat_dict[s])

    # Binary attribute filtering
    for s in test_info['sensitive_features']:
        if s == "gender" or s == "onehot_feat0":
            for i in range(feat_dict[s]):
                sampling_prob[s].append(1 / feat_dict[s])

                for two_tuple in range(1, len(test_info['sensitive_features'])):
                    s2 = test_info['sensitive_features'][two_tuple]
                    for j in range(feat_dict[s2]):
                        if sampling_prob[s2][j] == 0:
                            continue

                        rule = create_single_rule([s, s2], np.array([i, j]))
                        rule_query = create_sparse_rule_query(rule)
                        group_target = test_data.query(rule_query)

                        if group_target['user_id'].nunique() < test_info['threshold_num']:
                            sampling_prob[s2][j] = 0

    print(sampling_prob)
    return sampling_prob


if __name__ == '__main__':
    opt = parser.parse_args()
    config_file = opt.config

    # load params
    with open(config_file, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)

    sampling_params = dicts['sampling']
    model_params = sampling_params['model_params']
    data_params = sampling_params['data_params']
    algorithm_params = sampling_params['algorithm_params']

    # load data
    test_data = get_test_data(model_params, data_params)

    # load model
    check_path, train_features = get_saved_model_details(model_params, data_params)
    model = get_model(model_params, data_params)
    model_info = {
        'test_model': model,
        'train_features': train_features
    }

    # set params for testing
    test_info = sampling_params['test_info']
    test_info['test_data'] = test_data
    test_info['dataset'] = data_params['dataset']

    feat_dict = get_feat_dict(test_info['dataset'])

    test_info['threshold_num'] = feat_dict['user_id'] * test_info['threshold']

    rq1_result = []
    rq1_time = []

    rq3_result_min = []
    rq3_result_max = []

    rq2_result = []
    rq2_time = []

    for rounds in range(5):
        start_time = time.time()
        sampling_prob_min = {}
        sampling_prob_max = {}

        sampling_prob_min = initialize_filter(test_info, sampling_prob_min)
        sampling_prob_max = copy.deepcopy(sampling_prob_min)

        end_time = time.time()
        init_time = end_time - start_time
        print("initialize time: ", init_time)

        sampled_users = []
        sampled_users_str = []
        sampled_score = []

        unfairness_list = []
        time_list = []

        query_count = 0
        query_time = 0
        eval_time = 0

        for k in range(algorithm_params['iter']):
            print("Round", k + 1, "sampling")
            sampling_prob_min, sampling_prob_max = one_step(test_info, sampling_prob_min, sampling_prob_max,
                                                            algorithm_params['one_step_num'] * (k + 1) * 2,
                                                            sampled_users,
                                                            sampled_score, sampled_users_str)

            end_time = time.time()
            test_time = end_time - start_time
            print("time", test_time)

            time_list.append(test_time)

        end_time = time.time()
        test_time = end_time - start_time
        print("time", test_time)

        print(unfairness_list)
        print(time_list)

        print("========diversity=========")
        print(sampled_users)

        sampled_score = np.array(sampled_score)
        sampled_users = np.array(sampled_users)

        sorted_index = np.argsort(sampled_score)
        print(sampled_users[sorted_index])
        np.save('./results/' + model_params['model_name'] + '_' + data_params['dataset'] + '_' + test_info[
            'metric'] + '_our_rq5.npy', sampled_users[sorted_index])

        sampled_score = sorted(sampled_score)
        len_sampled_score = len(sampled_score)

        rq3_result_min.append(sampled_score[:int(len_sampled_score * 0.2)])
        rq3_result_max.append(sampled_score[-int(len_sampled_score * 0.2):])

        rq1_result.append(unfairness_list[-1])
        rq1_time.append(time_list[-1])

        rq2_result.append(unfairness_list)
        rq2_time.append(time_list)

    print('mean unfairness socre: {}'.format(np.mean(rq1_result)))
    print('mean time consumption: {}'.format(np.mean(rq1_time)))

    # Get the average at each position of all lists in a list of lists rq2_result
    rq2_result_mean = np.mean(np.array(rq2_result), axis=0)
    rq2_time_mean = np.mean(np.array(rq2_time), axis=0)

    print('unfairness score: {}'.format(rq2_result_mean))
    print('time consumption: {}'.format(rq2_time_mean))

    # save the rq2_result_mean and rq2_time_mean to csv
    np.savetxt('./results/' + model_params['model_name'] + '_' + data_params['dataset'] + '_' + test_info['metric'] + '_our_rq2_result_mean.csv', rq2_result_mean, delimiter=',')
    np.savetxt('./results/' + model_params['model_name'] + '_' + data_params['dataset'] + '_' + test_info['metric'] + '_our_rq2_time_mean.csv', rq2_time_mean, delimiter=',')

    rq3_result_min_mean = np.mean(np.array(rq3_result_min), axis=0)
    rq3_result_max_mean = np.mean(np.array(rq3_result_max), axis=0)

    # save the rq3_result_min_mean and rq3_result_max_mean
    np.save('./results/' + model_params['model_name'] + '_' + data_params['dataset'] + '_' + test_info['metric'] + '_our_rq3_min.npy', rq3_result_min_mean)
    np.save('./results/' + model_params['model_name'] + '_' + data_params['dataset'] + '_' + test_info['metric'] + '_our_rq3_max.npy', rq3_result_max_mean)
