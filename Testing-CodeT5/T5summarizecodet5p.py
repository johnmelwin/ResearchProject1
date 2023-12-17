from transformers import AutoModel, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = "cpu"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

code = """from my_AdaBoost import my_AdaBoost
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

if __name__ == "__main__":
    #  Load training data
    data_train = pd.read_csv("../data/Iris_train.csv")
    # Separate independent variables and dependent variables
    independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm",	"PetalWidthCm"]
    X = data_train[independent]
    y = data_train["Species"]
    # Train model
    base_estimator = DecisionTreeClassifier(criterion = "entropy", max_depth = 1)
    clf = my_AdaBoost(base_estimator=base_estimator, n_estimators = 10)
    clf.fit(X, y)
    # Load testing data
    data_test = pd.read_csv("../data/Iris_test.csv")
    X_test = data_test[independent]
    # Predict
    predictions = clf.predict(X_test)
    # Predict probabilities
    probs = clf.predict_proba(X_test)
    # Print results
    for i, pred in enumerate(predictions):
        print("%s\t%f" % (pred, probs[pred][i]))"""

input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, max_length=20)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# Convert a string of SVG resources to an image.