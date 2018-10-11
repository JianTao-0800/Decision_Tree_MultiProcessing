import math
from _node import Node
from multiprocessing import Pool
from functools import partial

# Builds the Decision Tree based on training data, attributes to train on and the outcome
def build_tree(df, attributes, outcome):
    # Get the number of positive and negative examples in the training data
    p, n = num_class(df, outcome)
    # If train data has all positive or all negative values then we have reached the end of our tree
    if p == 0 or n == 0:
	# Create a leaf node indicating it's prediction
        leaf = Node(None,None)
        leaf.leaf = True
        if p > n:
            leaf.predict = 1
        else:
            leaf.predict = 0
        return leaf
    else:
	# Determine attribute and its threshold value with the highest
	# information gain
        best_attr, threshold = choose_attr(df, attributes, outcome)
	# Create internal tree node based on attribute and it's threshold
        tree = Node(best_attr, threshold)
        sub_1 = df[df[best_attr] < threshold]
        sub_2 = df[df[best_attr] > threshold]
	# Recursively build left and right subtree
        tree.left = build_tree(sub_1, attributes, outcome)
        tree.right = build_tree(sub_2, attributes, outcome)
        return tree

# Returns the number of positive and negative data
def num_class(df, outcome):
    p_df = df[df[outcome] == 1]
    n_df = df[df[outcome] == 0]
    return p_df.shape[0], n_df.shape[0]

# Chooses the attribute and its threshold with the highest info gain from the set of attributes
def choose_attr(df, attributes, outcome):
    #max_info_gain = float("-inf")
    #best_attr = None
    #threshold = 0
    # Test each attribute (note attributes maybe be chosen more than once)
    # Use parallel computing
    pool = Pool(processes=4)
    partial_func = partial(select_thres_and_calc_ig, df=df, outcome=outcome)
    result_list = pool.map(partial_func, attributes)
    ig_list = list(zip(*result_list))[2]
    max_ig_idx = ig_list.index(max(ig_list))
    best_attr = result_list[max_ig_idx][0]
    threshold = result_list[max_ig_idx][1]
    max_info_gain = result_list[max_ig_idx][2]
    return best_attr, threshold

# New function added
def select_thres_and_calc_ig(attr, df, outcome):
    thres = select_threshold(df, attr, outcome)
    ig = info_gain(df, attr, outcome, thres)
    return attr, thres, ig

def select_threshold(df, attribute, outcome):
    # Convert dataframe column to a list and round each value
    values = df[attribute].tolist()
    values = [float(x) for x in values]
    # Remove duplicate values by converting the list to a set, then sort the set
    values = set(values)
    values = list(values)
    values.sort()
    max_ig = float("-inf")
    thres_val = 0
    # try all threshold values that are half-way between successive values in this sorted list
    for i in range(0, len(values)-1):
        thres = (values[i] + values[i+1])/2
        ig = info_gain(df, attribute, outcome, thres)
        if ig > max_ig:
            max_ig = ig
            thres_val = thres
	    # Return the threshold value that maximizes information gained

    return thres_val

# Calculate info content (entropy) of the test data
def info_entropy(df, outcome):
    # Dataframe and number of positive/negatives examples in the data
    p_df = df[df[outcome] == 1]
    n_df = df[df[outcome] == 0]
    p = float(p_df.shape[0])
    n = float(n_df.shape[0])
    # Calculate entropy
    if p  == 0 or n == 0:
        I = 0
    else:
        I = ((-1*p)/(p+n))*math.log(p/(p+n),2) + ((-1*n)/(p+n))*math.log(n/(p+n),2)
    return I

# Calculates the weighted average of the entropy after an attribute test
def remainder(df, df_subsets, outcome):
    # number of test data
    num_data = df.shape[0]
    remainder = float(0)
    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            remainder += float(df_sub.shape[0]/num_data)*info_entropy(df_sub, outcome)
    return remainder

# Calculates the information gain from the attribute test based on a given threshold
# Note: thresholds can change for the same attribute over time
def info_gain(df, attribute, outcome, threshold):
    sub_1 = df[df[attribute] < threshold]
    sub_2 = df[df[attribute] > threshold]
    # Determine information content, and subract remainder of attributes from it
    ig = info_entropy(df, outcome) - remainder(df, [sub_1, sub_2], outcome)
    return ig
