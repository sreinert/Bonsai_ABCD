import h5py

def read_h5(signals_file, print_key=False, key_idx=1):
    with h5py.File(signals_file, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        if print_key:
            print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[key_idx]

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
    
    return data


def read_h5_with_key(signals_file, print_key=True, key='SyncTTL'):
    with h5py.File(signals_file, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        if print_key:
            print("Keys: %s" % f.keys())

        # If key is a dataset name, 
        # this gets the dataset values and returns as a list
        if key in f.keys():
            data = list(f[key])
        else:
            KeyError("This is not a valid key.")
    
    return data