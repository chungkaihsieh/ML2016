import numpy as np
import sys

print ("hw0_1_1 start!!!")
f_load = open(sys.argv[2])
reg_0 = f_load.read().splitlines()
f_load.close()
get_column = int(sys.argv[1])
out_line = []
for i in range(len(reg_0)):
	reg_1 = reg_0[i]
	reg_2 = reg_1.split()
	out_line.append(float(reg_2[get_column]))
out_line.sort()
m=0
reg_3 = ""
hw0 = open('ans1.txt', 'w')
for i in range(len(out_line)):
    reg_3 = reg_3 + str(out_line[i])
    if i < len(out_line)-1:
        reg_3 = reg_3 + ","
hw0.write(str(reg_3))
hw0.close()
