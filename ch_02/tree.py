from math import log
import operator

def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    # 为所有可能分类创建字典
    for feature_vec in data_set:
        current_label = feature_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        # 以2为底求对数
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def split_data_set(data_set, axis, value):
    # 创建新的list
    ret_data_set = []
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            # 抽取
            reduced_feature = feature_vec[:axis]
            reduced_feature.extend(feature_vec[axis+1:])
            ret_data_set.append(reduced_feature)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    # 最后一个是分类，不是特征
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 遍历每个特征
        # 创建唯一的分类标签
        feature_list = [example[i] for example in data_set]
        unique_values = set(feature_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            # 将各个子集的熵加权平均
            prob = len(sub_data_set) / len(data_set)
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            # 得到最好的信息增益以及对应的特征
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count = 0
        class_count['vote'] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(data_set, labels):
    # 返回字符串字典
    class_list = [example[-1] for example in data_set]
    # 类别完全相同，停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征，返回出现次数最多的
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {
        best_feature_label: {
        }
    }
    # 得到列表包含的所有属性
    del labels[best_feature]
    feature_values = [example[best_feature] for example in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return my_tree


if __name__ == "__main__":
    my_data, labels = create_data_set()
    print(my_data)
    # my_data[0][-1] = 'maybe'
    print(calc_shannon_ent(my_data))
    print(split_data_set(my_data, 0, 0))
    print(choose_best_feature_to_split(my_data))
    my_tree = create_tree(my_data, labels)
    print(my_tree)