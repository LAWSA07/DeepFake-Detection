from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test
