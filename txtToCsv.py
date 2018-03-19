import csv
import re
import os
txt_file = r"train.txt"
csv_file = r"train.csv"

in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'w'))

out_csv.writerows(in_txt)

# data = open('results.csv').read()
# print(re)
# new_data = data.replace(',\n', '\n')
# print(new_data)
# #out_csv.writerows(in_txt)
