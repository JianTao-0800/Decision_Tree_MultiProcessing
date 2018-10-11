import time
import pandas as pd
from tree import Tree

# Cleans the input data, removes 'Diagnosis' column and adds 'Outcome' column
# where 0 means healthy and 1 means colic
def clean(csv_file_name):
    df = pd.read_csv(csv_file_name, header=None)
    df.columns = ['K', 'Na', 'CL', 'HCO', 'Endotoxin', 'Anioingap', 'PLA2', 'SDH', 'GLDH', 'TPP',
                  'Breath rate', 'PCV', 'Pulse rate', 'Fibrinogen', 'Dimer', 'FibPerDim',
                  'Diagnosis']
    # Create new column 'Outcome' that assigns healthy horses a value of 0 (negative case) and
    # horses with colic a value of 1 (positive case), this makes creating our decision tree easier
    df['Outcome'] = 0
    df.loc[df['Diagnosis'] == 'colic.', 'Outcome'] = 1
    df.drop(['Diagnosis'], axis=1, inplace=True)
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

def main():
    # An example use of 'build_tree' and 'predict'
    df_train = clean('horseTrain.txt')
    attributes =  ['K', 'Na', 'CL', 'HCO', 'Endotoxin', 'Anioingap', 'PLA2', 'SDH', 'GLDH', 'TPP',
                   'Breath rate', 'PCV', 'Pulse rate', 'Fibrinogen', 'Dimer', 'FibPerDim']
    dec_tree = Tree(df_train, attributes, 'Outcome')
    print("Building the tree...")
    time1 = time.time()
    dec_tree.build_tree()
    time2 = time.time()
    print("Time spent to build the tree %.2f seconds" % (time2 - time1))
    print("Finish building the tree...")
    time.sleep(1)
    
    print("Shape of Tree ...")
    dec_tree.print_tree()
    time.sleep(1)

    print("Accuracy of test data ... ")
    df_test = clean('horseTest.txt')
    print(dec_tree.predict(df_test))
    print(df_test['Outcome'])
    #print(str(test_predictions(root, df_test)*100.0) + '%')

if __name__ == '__main__':
    main()
