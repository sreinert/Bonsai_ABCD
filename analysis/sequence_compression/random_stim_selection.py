import random
import numpy as np

# previous_random_number = 3

# available_numbers = [n for n in range(1, 5) if n != previous_random_number]
# session_number = random.choice(available_numbers)

# print(f"Previous day's number: {previous_random_number}")
# print(f"Today's number: {session_number}")

arr = np.arange(10)
shuffled_arr = np.random.permutation(arr)
print(shuffled_arr)

# random_stim = np.sort(np.random.choice(arr, size=4, replace=False))
# print(f"Randomly selected stimuli: {random_stim}")