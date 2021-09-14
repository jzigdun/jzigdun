import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#################### ---------------- ####################
# define functions for calculating gini index over all subsets
#################### ---------------- ####################


def gini_index(prob_vector):

    return 1 - np.sum(prob_vector**2)


def gini_index_per_group(prob_vector,subset_size, set_size):

    gini_per_group = gini_index(prob_vector) * subset_size/set_size

    return gini_per_group


def avg_gini_index(sett):

    # inputs: set - list of classification arrays (for each given attribute)

    # calculate size of set
    subset_sizes = [len(subset) for subset in sett]
    set_size = np.sum(subset_sizes)

    # initialization
    avg_gini = 0

    for subset in sett:

        # count number of values in each class
        class_count = np.array(np.unique(subset, return_counts=True))

        # calculate conditional probability vector per subset
        cond_prob_vec = class_count[1, :] / len(subset)

        # calculate average gini over all subsets
        avg_gini += gini_index_per_group(cond_prob_vec, len(subset), set_size)

    return avg_gini

#################### ---------------- ####################
# define functions for splitting set into subsets
#################### ---------------- ####################


def sort_attribute_vals(feature_vec,class_vec):

    # obtain ids of sorted feature vector
    sorted_ids = np.argsort(feature_vec)

    return np.array(feature_vec)[sorted_ids], np.array(class_vec)[sorted_ids]


def locate_crossing_ids(class_vec):

    # define filter
    h = [-1, 1]

    # a change in class will be displayed as a jump in value for the correlation
    change = np.correlate(class_vec.ravel(), h)

    # ids will be displayed at the last index before the change in class occurs
    cross_ids = np.argwhere(np.abs(change) > 0)

    return cross_ids.ravel()


def split(base_set, optimal_id, optimal_split):

    # initialization
    subset_list = []

    # obtain dataset for LHS of optimal split value
    subset_left = base_set[base_set[:, optimal_id] < optimal_split]
    subset_list.append(subset_left)

    # obtain dataset for RHS of optimal split value
    subset_right = base_set[base_set[:, optimal_id] >= optimal_split]
    subset_list.append(subset_right)

    return subset_list


def find_best_split(base_set, base_class):

    # upper bound initialization for minimal value of gini index
    min_gini = 5

    # calculate best binary split per feature
    for feature_id in range(len(base_set.T)):

        # sort feature array
        sorted_feat_arr, sorted_class_vec = sort_attribute_vals(
                base_set[:, feature_id], base_class)

        # find last index per class before class changes value
        crossing_ids = locate_crossing_ids(sorted_class_vec)

        # filter out empty list of lists if no cross
        no_cross_list = [x for x in crossing_ids if x is not None]

        try:
            if not no_cross_list:
                raise TypeError

            for cross_id in crossing_ids:

                # (re-)initialize subset list for avg-gini calculation
                subset_list = []

                # calculate split index
                split_index = cross_id+1

                # calculate value which splits features
                split_val = (sorted_feat_arr[cross_id] + sorted_feat_arr[cross_id+1])/2

                # generate list of subsets
                subset_list.append(sorted_class_vec[:split_index])
                subset_list.append(sorted_class_vec[split_index:])

                # calculate average gini for all subsets
                avg_gini = avg_gini_index(subset_list)

                #print('Min Gini:', min_gini)
                #print('Avg Gini:', avg_gini)

                if avg_gini < min_gini:

                    min_gini = avg_gini
                    best_split = split_val
                    best_feature = feature_id

        except TypeError:

            # if we only have one class
            subset_list = [sorted_class_vec]

            # calculate average gini for all subsets
            min_gini = avg_gini_index(subset_list)
            best_split = 0
            best_feature = feature_id

            # break out of loop
            break

    return best_feature, best_split, min_gini


#################### ---------------- ####################
# define class which builds decision tree
#################### ---------------- ####################

class Node():

    min_size = 1
    #max_depth = 5 # for classical decision tree
    max_depth = 20 # symbolize no pruning for random forest

    def __init__(self, dataset, feature_ind, split_val, tree_depth):

        self.right = None
        self.left = None
        self.depth = tree_depth
        self.feature_index = feature_ind
        self.dataset = dataset
        self.node_split = split_val
        self.gini = None
        self.leaf = None

    def terminal_node(self):

        if len(self.dataset) == 0:
            terminal_node = 2

        else:
            class_labels = self.dataset[:,-1]
            label_count = np.bincount(class_labels.astype(int))

            print('Label count:', label_count)
            terminal_node = np.argmax(label_count)

        return terminal_node

    def stopping_condition(self):

        # check if dataset is of minimum length
        if len(self.dataset) <= 1:
            self.gini = 'N/A, min dataset length'
            return True

        # check if tree is at maximum depth
        if self.depth == self.max_depth:
            self.gini = 'N/A, max tree depth'
            return True

        return False

    def split_node(self):

        # if basic stopping conditions are met, classify leaf
        if self.stopping_condition():
            self.leaf = self.terminal_node()
            return None

        else:

            # assuming no termination: find best split
            feature_id, split_value, min_gini = find_best_split(
                base_set=self.dataset[:, :-1],
                base_class=self.dataset[:, -1])

            # if we have one class, classify leaf
            if min_gini == 0.0:

                self.gini = min_gini
                self.leaf = self.terminal_node()
                return None

            else:

                self.gini = min_gini

                # split dataset into subsets
                subsets = split(base_set=self.dataset,
                                optimal_id=feature_id,
                                optimal_split=split_value)

                # 1st element in subset indicates all attributes
                # of given feature less than split value,  2nd element
                # in subset indicates all attributes greater than split
                # value

                self.left = Node(subsets[0], feature_id, split_value, self.depth+1)
                self.left.split_node()

                self.right = Node(subsets[1], feature_id, split_value, self.depth+1)
                self.right.split_node()
                return None

    def print_tree(self):
        print('Tree depth: {}'.format(self.depth))
        print('Feature id: {}'.format(self.feature_index))
        print('Split value: {}'.format(self.node_split))
        print('Min gini: {}'.format(self.gini))
        print('Class: {}'.format(self.leaf))

        if self.right is not None:
            print('Right node:')
            self.right.print_tree()
        if self.left is not None:
            print('Left node:')
            self.left.print_tree()

    def predict(self, dataset_test):

        # check if we reached a leaf
        if self.leaf is not None:

            return self.leaf

        # if not, recursively advance example
        # to other nodes:
        # condition to check left node:
        elif dataset_test[self.feature_index] < self.node_split:
            pred = self.left.predict(dataset_test)

            if pred == 2:
                pred = self.right.predict(dataset_test)

        # condition to check right node:
        else:
            pred = self.right.predict(dataset_test)

        return pred

if __name__ == "__main__":

    # load file
    data = pd.read_csv('wdbc.data', header=None)

    # replace labels - benign = 0, malignant = 1
    data[1].replace({'M': 1, 'B': 0}, inplace=True)

    data = data.drop(data.columns[0], axis=1)

    data = data.to_numpy()

    x = data[:,1:]
    y = data[:,0]

    # separate data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(
                x, y, test_size=0.2, random_state=42)

    train_y = np.reshape(train_y, (train_y.shape[0],1))
    base_group = np.concatenate((train_x,train_y), axis=1)

    # generate base node
    trunk = Node(dataset=base_group,
                 feature_ind=0,
                 split_val=0,
                 tree_depth=0)

    # build tree with training set
    trunk.split_node()

    # print nodes and features
    trunk.print_tree()

    # perform prediction with test set
    prediction_vec = []
    count = 0
    accuracy = 0
    for instance in test_x:

        prediction = trunk.predict(instance)
        prediction_vec.append(prediction)

        print('Tree prediction: {}, Actual label: {}'.format(prediction, test_y[count]))
        accuracy += (prediction == int(test_y[count]))
        count += 1

    print('Accuracy: {}%'.format(accuracy*100/len(test_y)))