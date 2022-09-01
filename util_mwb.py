##############################################################################
# Convert all float columns that are actually integers with some NaN values
# Note than IntegerArrays are an experimental addition in Pandas 0.24. They
# allow integer columns to contain NaN fields like float columns.
#
# This is a rather brute-force technique that loops through every column
# and every row. There's likely a more efficient way to do it since it
# takes a long time (7-8 min. on ceb-bopf-vm) and uses up a lot of memory.
##############################################################################
import pandas as pd
def convert_integer(df):
    type_dict = {}
    for col in df.columns:
        intcol_flag = True
        if df[col].dtype == 'float64':  # assuming float dytpe is "float64"
            # Inner loop makes this very slow, but can't find a vectorized solution
            for val in df[col]:
                # If not NaN and the int() value is different from
                # the float value, then we have an actual float.
                if pd.notnull(val) and abs(val - int(val)) > 1e-6:
                    intcol_flag = False
                    break;
            # If not a float, change it to an Int based on size
            if intcol_flag:
                if df[col].abs().max() < 127:
                    df[col] = df[col].astype('Int8')
                elif df[col].abs().max() < 32767:
                    df[col] = df[col].astype('Int16')
                else:   # assuming no ints greater than 2147483647 
                    df[col] = df[col].astype('Int32') 
#        print(f"{col} is {df[col].dtype}")
        type_dict[col] = df[col].dtype
    return type_dict


##############################################################################
# Read a subset of a large file; can import types from a pickled mapping file
##############################################################################
# "types-file" is assumed to be a pickled Python dictionary of column names to Python datatypes.
def subset_csv(filename, rows=100, columns=10, random=False, types_file=False):
    if random:
        print("Random not yet implemented; first rows and columns used")
    if types_file:
        import pickle
        with open(types_file, 'rb') as file:
            csl_types = pickle.load(file)
        df = pd.read_csv(filename, index_col=0,  nrows=rows, usecols=range(0, columns+1),
                         header=0, skip_blank_lines=True, dtype=csl_types)
    else:
        df = pd.read_csv(filename, index_col=0,  nrows=rows, usecols=range(0, columns+1),
                         header=0, skip_blank_lines=True)
    return df


############################################################
# Read the data with "pickled" dtypes file
#    Note - removed hard-coded default path names
############################################################
def read_csl_csv(filename, types_file):
    import pickle
    print(f'types_file = {types_file}')
    if types_file is not None:
        with open(types_file, 'rb') as file:
            csl_types = pickle.load(file)
        df = pd.read_csv(filename, index_col=0, header=0, skip_blank_lines=True, dtype=csl_types)
    else:
        df = pd.read_csv(filename, index_col=0, header=0, skip_blank_lines=True)

    return df

