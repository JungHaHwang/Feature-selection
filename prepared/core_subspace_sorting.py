import re
from collections import Counter

def count_core_subspaces(input_file, output_file):

    pattern = re.compile(r"\(\d+, \d+, \d+, \d+\)")
    
    subspace_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            matches = pattern.findall(line)
            subspace_list.extend(matches)
    

    counter = Counter(subspace_list)
    

    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    

    with open(output_file, "w", encoding="utf-8") as f:
        for subspace, count in sorted_counts:
            f.write(f"{subspace} : {count}\n")


input_file = "core_subspaces.txt"
output_file = "core_subspace_sorting.txt"
count_core_subspaces(input_file, output_file)
