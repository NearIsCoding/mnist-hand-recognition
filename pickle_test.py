import pickle

filename = "svm_model.pickle"

print("Loading model...")
loaded_model = pickle.load(open(filename, "rb"))
print("Model succesfully loaded!!!")