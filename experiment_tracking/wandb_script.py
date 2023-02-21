import argparse
import wandb 
import seaborn as sns
import torch



from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='', type=str)
    args = parser.parse_args()
    
    run = wandb.init(project='hello', config={'model': args.model})

    # Load dataset
    df = datasets.load_iris()
    X = df.data
    y = df.target

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    
    i = 0
    
    for model_name, model in models.items():
        # Instantiate the model
        i +=1
        b = 'hello' + str(i)
        run = wandb.init(project= b, config={'model': model_name})

        wandb.run.summary['model'] = model_name


        # Get metrics
        accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
        f1_macro = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_macro').mean()
        neg_log_loss = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_log_loss').mean()

        wandb.log({'accuracy': accuracy, 'f1_macro': f1_macro, 'neg_log_loss': neg_log_loss})

        fig = sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=df.target_names[y],
            alpha=1.0,
            edgecolor="black",
        )

        wandb.log({'data_scatter': wandb.Image(fig)})
