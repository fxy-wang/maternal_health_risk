# train_model.py

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import operator

def grid_search(classifiers, params, X_train, y_train, X_test, y_test):

    result=[]
    
    # Loop through all classifiers 
    for i in range(0,len(classifiers)):
        name = type(classifiers[i]).__name__
        grid_search = GridSearchCV(estimator=classifiers[i], param_grid=params[i], cv=5, scoring='accuracy', n_jobs=-1)

        grid_search.fit(X_train,y_train)
        best_params = grid_search.best_params_
        best_clf = grid_search.best_estimator_

        best_clf.fit(X_train, y_train)

        y_pred = best_clf.predict(X_train)  

        accuracy = accuracy_score(y_train, y_pred)
        print("Tuned {} Accuracy: {:.2f}%".format(name, accuracy * 100))

        print("Best Hyperparameters:", best_params)

        print(classification_report(y_test, best_clf.predict(X_test)))

      # Storing all result
        result.append\
        (
            {
                'grid': grid_search,
                'classifier': best_clf,
                'best score': grid_search.best_score_,
                'best params': best_params,
            }
        )

        # Sort result by best score
        result = sorted(result, key=operator.itemgetter('best score'),reverse=True)

    return result