import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    # 计算距离
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 每个元素求平方
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    # 选择距离最小的k个点
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(file_name):
    with open(file_name, encoding='utf-8') as file_obj:
        array_lines = file_obj.readlines()
        lines_num = len(array_lines)
        # 创建返回的Numpy矩阵
        return_mat = np.zeros((lines_num, 3))
        class_label_vector = []
        index = 0
        # 解析文件数据到列表
        for line in array_lines:
            line = line.strip()
            elem_list = line.split('\t')
            return_mat[index,:] = elem_list[:3]
            class_label_vector.append(int(elem_list[-1]))
            index += 1
        return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix('data/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    ranges, min_vals = min_vals, ranges
    m = norm_mat.shape[0]
    # 测试数量
    num_test_vecs = int(m*ho_ratio)
    error_count = 0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i,:], norm_mat[num_test_vecs:m,:], dating_labels[num_test_vecs:m], 3)
        print('the classifier came back with: %d, the real answer is : %d' % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print('the total error rate is %f' % (error_count / float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percenttage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dating_data_mat, dating_labels = file2matrix('data/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # ranges, min_vals = min_vals, ranges
    in_array = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_array-min_vals)/ranges, norm_mat, dating_labels, 3)
    print('You will probably like this person:', result_list[classifier_result-1])


if __name__ == "__main__":
    group, labels = create_data_set()
    print(group)
    print(labels)
    a = np.array([[1, 2], [3, 4],[5, 6]])
    print(a**2)
    print(a.sum())
    print(a.sum(axis=0))
    print(a.sum(axis=1))
    d = {
        'a': 11,
        'b': 13,
        'f': 12,
        'd': 12
    }
    sd = sorted(d.items(), key=lambda kv: (kv[1], kv[0]))
    print(sd)
    print(sd[0])
    print(classify0([0.9, 0.9], group, labels, 3))
    print(a)
    print(a[0])
    a[0] = [2, 1]
    print(a[0,:])
    dating_data_mat, dating_labels = file2matrix('data/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:,1], dating_data_mat[:,0], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    print(np.shape(a))
    print('--')
    print(a)
    print(a.min(1))
    print(a.min(0))
    print('--')
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    print(norm_mat[:7,:])
    dating_class_test()
    classify_person()