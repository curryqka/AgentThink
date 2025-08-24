import pickle

pickle_file_path = "data/tool_results/val/0a0d6b8c2e884134a3b48df43d54c36a.pkl"
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())