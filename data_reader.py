import numpy as np

def parse_file(file_name):
    table = np.empty([0,0])

    with open(file_name, 'r') as f:
        for line in f:
            terms = np.char.split(line.strip(), ',').tolist()
            if np.size(table) == 0:
                table = [float(x) for x in terms]
            else:
                table = np.vstack((table, [float(x) for x in terms]))

    features = table[:, :-1]
    features = np.array([np.hstack((ft, 1)) for ft in features]) # Add 1 for bias term
    labels = table[:, -1]
    labels = 2*labels - 1 # Convert from 0 and 1 to -1 and 1

    return features, labels