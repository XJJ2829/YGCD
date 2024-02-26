import os
from Draw import DrawScatter

# 反复

def ReadFile(filename,TimeLine):
    with open(filename, "r") as f:
    	for line in f.readlines():
    		line = line.strip('\n')
    		strlist = line.split('_')
    		ckey = strlist[1] + '_' + strlist[2]
    		cvalue = strlist[0]
    		TimeLine[ckey].append(cvalue)

file_Paths = ['20161209.txt','20171109.txt','20181129.txt','20191114.txt','20201118.txt','20211223.txt']

TimeLine = {}
for i in range(65):
    for j in range(78):
        TimeLine['{}_{}'.format(i,j)] = []

for i in range(len(file_Paths)):
	point_List = []
	ReadFile(file_Paths[i],TimeLine)

stable_dic = {}

for i in TimeLine:
	count = {}
	for j in TimeLine[i]:
		count[j] = TimeLine[i].count(j)
	if len(count) == 1:
		stable_dic[i] = TimeLine[i]

for i in stable_dic:
	TimeLine.pop(i)
landclass = ['Farmland','Industrial','Residential','Woods','RiverLake','Rural']

continuous = {}
for item in TimeLine:
	flag = True
	for i in landclass:
		index = [j for j,x in enumerate(TimeLine[item]) if x == i]
		if index:
			if index[0] + (len(index)-1) != index[len(index)-1]:
				flag = False
				break
	if flag: continuous[item] = TimeLine[item]

for i in continuous:
	TimeLine.pop(i)

def FindLongList(classList):
    conti =[]
    contiList = []
    for i in range(len(classList)-1):
        if classList[i] == classList[i+1] - 1:
            if conti:
                conti.pop()
            conti.append(classList[i])
            conti.append(classList[i+1])
            if i == len(classList) - 2:
                contiList.append(conti)
        else:
            if conti:
                contiList.append(conti)
            conti = []
    if len(contiList) > 1:
        lens = []
        for item in contiList:
            lens.append(len(item))
        return contiList[lens.index(max(lens))]
    elif len(contiList) == 1:
        return contiList[0]
    else:
    	return [0]

def CompareList(conti_Index):
    keys = list(conti_Index.keys())
    max_index = conti_Index[keys[0]]
    max_dic = {}
    max_dic[keys[0]] = max_index

    for i in range(1,len(conti_Index)):
        if len(conti_Index[keys[i]]) > len(max_index):
            max_dic.clear()
            max_dic[keys[i]] = conti_Index[keys[i]]
            max_index = conti_Index[keys[i]]
        elif len(conti_Index[keys[i]]) == len(max_index):
            max_dic[keys[i]] = conti_Index[keys[i]]
    return max_dic

filter_All = {} # 转为一致的索引列表
continuous_dic = {}
for i in TimeLine:
	count = {}
	# print(i)
	# print(TimeLine[i])
	for j in TimeLine[i]:
		count[j] = TimeLine[i].count(j)

	flag_equal = False
	keys = list(count.keys())
	for n in range(len(keys) - 1):
		if count[keys[n]] == count[keys[n+1]]:
			flag_equal = True
		else:
			flag_equal = False
			break

	if not flag_equal:
		count_order = sorted(count.items(),key = lambda x:x[1],reverse = True)
		filter_All[i] = count_order[0][0]
	else:
		conti_Index = {}

		for c in landclass:
			index = [j for j,x in enumerate(TimeLine[i]) if x == c]
			if index:
				conti_Index[c] = FindLongList(index)
		max_dic = CompareList(conti_Index)

		if len(max_dic) == 1:
			filter_All[i] = list(max_dic.keys())[0]
		else:
			flag_zero = True
			for j in max_dic:
				if max_dic[j] != [0]:
					flag_zero = False
			if flag_zero:
				continuous_dic[i] = [TimeLine[i][0],TimeLine[i][len(TimeLine[i]) - 1]]
			else:
				max_order = sorted(max_dic.items(),key = lambda x:x[1][0],reverse = False)
				temp_list = []
				for item in max_order:
					temp_list.append(item[0])
				continuous_dic[i] = temp_list

for i in filter_All:
	temp_list = []
	for j in range(6):
		temp_list.append(filter_All[i])
	filter_All[i] = temp_list

for i in continuous_dic:
	print(continuous_dic)
	temp_list = []
	for j in range(6):
		if j < 3:
			temp_list.append(continuous_dic[i][0])
		else:
			temp_list.append(continuous_dic[i][1])
	continuous_dic[i] = temp_list

newTimeLine={}
newTimeLine.update(stable_dic)
newTimeLine.update(continuous)
newTimeLine.update(filter_All)
newTimeLine.update(continuous_dic)

# p1 = []
# p2 = []
# p3 = []
# p4 = []
# p5 = []
# p6 = []

# for i in newTimeLine:
# 	p1.append(newTimeLine[i][0] + '_' + i)
# 	p2.append(newTimeLine[i][1] + '_' + i)
# 	p3.append(newTimeLine[i][2] + '_' + i)
# 	p4.append(newTimeLine[i][3] + '_' + i)
# 	p5.append(newTimeLine[i][4] + '_' + i)
# 	p6.append(newTimeLine[i][5] + '_' + i)

# DrawScatter(p1,'20161209')
# DrawScatter(p2,'20171109')
# DrawScatter(p3,'20181129')
# DrawScatter(p4,'20191114')
# DrawScatter(p5,'20201118')
# DrawScatter(p6,'20211223')
