import re
from collections import defaultdict

# Initialize variables
timing_data = defaultdict(list)

# Define the pattern to match the timing lines
pattern = re.compile(r"(\w+ time): (\d+) ms")

# Read the file
with open("result.txt", "r") as file:
    for line in file:
        matches = pattern.findall(line)
        for match in matches:
            timing_name = match[0]
            timing_value = int(match[1])
            timing_data[timing_name].append(timing_value)

# Calculate the count and average
result = {}
for key, values in timing_data.items():
    count = len(values)
    average = sum(values) / count if count > 0 else 0
    result[key] = {'count': count, 'average': average}

# Display the results
for key, value in result.items():
    print(f"{key}: Count = {value['count']}, Average = {value['average']:.2f} ms")

