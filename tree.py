from _node import Node
#from _build_tree import build_tree
from _build_tree_parallel import build_tree

# Given a instance of a training data, make a prediction of healthy or colic based on the
# Decision Tree. Assumes all data has been cleaned (i.e. no NULL data)
def _predict(node, row_df):
    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict
    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] <= node.thres:
        return _predict(node.left, row_df)
    elif row_df[node.attr] > node.thres:
        return _predict(node.right, row_df)

# Prints the tree level starting at given level
def _print_tree(root, num=0):
    if root.leaf:
        print(str(num) + ': ' + '[leaf node] ' +str(root.predict))
    else:
        print(str(num) + ': ' + '[' + root.attr + '<' + str(root.thres) + ']'
              + ' yes=' + str(2*num+1) + ' no=' + str(2*num+2))
    if root.left:
        _print_tree(root.left, 2*num+1)
    if root.right:
        _print_tree(root.right, 2*num+2)

# Tree Class
class Tree:
    def __init__(self, df_train, attributes, outcome):
        self.dtrain = df_train
        self.X_names = attributes
        self.Y_name = outcome
        self.tree = None

    def build_tree(self):
        if not self.tree:
            self.tree = build_tree(self.dtrain, self.X_names, self.Y_name)

    def predict(self, df_test):
        # Given a set of data, make a prediction for each instance using the Decision Tree
        preds = []
        for index,row in df_test.iterrows():
            pred = _predict(self.tree, row)
            preds.append(pred)
        return preds

    def print_tree(self):
        _print_tree(self.tree)
