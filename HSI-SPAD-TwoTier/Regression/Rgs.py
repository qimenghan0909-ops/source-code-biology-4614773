from Regression.ClassicRgs import Pls, Svregression, ELM, MLR


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, num=30, C=2, gamma=1e-07, kernel='linear'):
    if model == "Pls":
        Rmse, R2, Mae, y_pre ,y_train_pred= Pls(X_train, X_test, y_train, y_test, num=num)
    elif model == "SVR":
        Rmse, R2, Mae, y_pre,y_train_pred = Svregression(X_train, X_test, y_train, y_test, C=C, gamma=gamma, kernel=kernel)
    elif model == "ELM":
        Rmse, R2, Mae, y_pre,y_train_pred = ELM(X_train, X_test, y_train, y_test)
    elif model == "MLR":
        Rmse, R2, Mae, y_pre, y_train_pred = MLR(X_train, X_test, y_train, y_test)

    return Rmse, R2, Mae, y_pre ,y_train_pred



