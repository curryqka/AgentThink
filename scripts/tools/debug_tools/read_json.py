import jsonlines
file_name = ""
with open(file_name, 'r') as f:
    for sample in jsonlines.Reader(f):
        breakpoint()
        # message = ",".join(sample["messages"])


breakpoint()
print("")