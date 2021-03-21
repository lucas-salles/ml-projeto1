from sklearn.model_selection import train_test_split


def dataSplit(dataset):
    columns = len(dataset.columns)

    y = dataset[0]  # extrai a primeira coluna, que é o label
    X = dataset.loc[:, 1:columns-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=None, stratify=y)  # 80% treino e 20% teste

    return [X_train, X_test, y_train, y_test]
