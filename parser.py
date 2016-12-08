import re
import pickle
import string
#  change this to whatever body of text you want to use
file_path = "whitman-leaves.txt"

file = open(file_path, 'r')
text = file.read()
words = text.lower().split()
for word in words:
    word = filter(str.isalnum, word)
firstWords = words
length = len(words) - 1
word_pairs = []
for i in range(0, length-1):
    #  print i
    word_pairs.append((words[i], words[i+1], words[i+2]))
unique_pairs = {}
for pair in word_pairs:
    #  print pair
    if pair in unique_pairs:
        unique_pairs[pair] = unique_pairs[pair] + 1
    else:
        unique_pairs[pair] = 1
#  for p in unique_pairs:
    #  print p, unique_pairs[p]
lines = text.split("\n")
print(lines)
pickle_file = open('pairs.pkl', 'wb')
wfie = open('words.pkl', 'wb')
lfile = open('lines.pkl', 'wb')
pickle.dump(lines, lfile)
pickle.dump(unique_pairs, pickle_file)
pickle.dump(firstWords, wfie)
pickle_file.close()
file.close()
wfie.close()

print firstWords[1]
