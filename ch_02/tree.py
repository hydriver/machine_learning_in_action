from math import log

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


if __name__ == "__main__":
    my_data, labels = create_data_set()
    print(my_data)
    # my_data[0][-1] = 'maybe'
    print(calc_shannon_ent(my_data))
    print(split_data_set(my_data, 0, 0))