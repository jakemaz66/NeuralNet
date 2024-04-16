import pandas as pd

def read(path):
    path = 'data/winequality-red.csv'

    df = pd.read_csv(path, delimiter=';')
    return df

if __name__ == '__main__':
    read('data/winequality-red.csv')
