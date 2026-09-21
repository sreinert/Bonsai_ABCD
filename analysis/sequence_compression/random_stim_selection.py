import random

previous_random_number = 1

available_numbers = [n for n in range(1, 5) if n != previous_random_number]
session_number = random.choice(available_numbers)

print(f"Previous day's number: {previous_random_number}")
print(f"Today's number: {session_number}")

# import os
# import re
# import numpy as np
# import random
# from pathlib import Path

# basepath = Path('/Volumes/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression/Bonsai/sc_yml_scripts/cohort2')

# # Previous day's random number
# previous_random_number = 3

# # Find all .yml files, excluding macOS hidden files
# files = [
#     f for f in basepath.glob("*.yml")
#     if not f.name.startswith(".")
# ]

# # Extract the number from each filename
# available_files = []

# for f in files:
#     match = re.search(r"session-.*?-(\d+)-random", f.name)

#     if match:
#         file_number = int(match.group(1))

#         # Exclude the previous day's number
#         if file_number != previous_random_number:
#             available_files.append(f)

# # Check that there are files left to choose from
# if len(available_files) == 0:
#     raise ValueError("No files available after excluding the previous random number.")

# # Randomly select one
# selected_file = random.choice(available_files)

# # Get the number of the selected file
# match = re.search(r"session-.*?-(\d+)-random", selected_file.name)

# if match:
#     session_number = int(match.group(1))

#     print(f"Previous day's number: {previous_random_number}")
#     print(f"Selected file: {selected_file.name}")
#     print(f"File number: {session_number}")
# else:
#     raise ValueError(
#         f"Filename does not match expected pattern: {selected_file.name}"
#     )