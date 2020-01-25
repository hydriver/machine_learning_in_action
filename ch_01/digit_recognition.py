import numpy as np
import kNN
from os import listdir

def img2vector(file_name):
    return_vect = np.zeros((1, 1024))
    with open(file_name, 'r', encoding='utf-8') as file_obj:
        i = 0
        for line in file_obj:
            for digit in line.strip():
                return_vect[0,i] = int(digit)
                i += 1
    return return_vect


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('data/digits/trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i,:] = img2vector('data/digits/trainingDigits/%s' % file_name_str)
    test_file_list = listdir('data/digits/testDigits')
    error_count = 0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('data/digits/testDigits/%s' % file_name_str)
        classifier_result = kNN.classify0(vector_under_test, training_mat, hw_labels, 3)
        print('the classifier came back with %d, the real answer is %d' % (classifier_result, class_num_str))
        if class_num_str != classifier_result:
            error_count += 1
    print('the total number of errors is %d' % error_count)
    print('the total error rate is %f' % (error_count / m_test))


if __name__ == "__main__":
    vec = img2vector('data/digits/trainingDigits/0_13.txt')
    print(vec[0, 32:63])
    handwriting_class_test()
