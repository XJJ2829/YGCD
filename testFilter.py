import os
import copy
from Draw import DrawScatter

def ReadFile(filename,TimeLine):
    with open(filename, "r") as f:
    	for line in f.readlines():
    		line = line.strip('\n')
    		strlist = line.split('_')
    		ckey = strlist[1] + '_' + strlist[2]
    		cvalue = strlist[0]
    		TimeLine[ckey].append(cvalue)

file_Paths = ['20161209.txt','20171109.txt','20181129.txt','20191114.txt','20201118.txt','20211223.txt']

# file_Paths = ['20161209.txt','20171109.txt','20181129.txt','20191114.txt','20201118.txt']

years = 6
'''
Verson 2.0 Vote Rules:
	1.筛选时间序列中完全不变的点——稳定点（stable_dic)
	2.筛选时间序列中保持连续变化的点——连续点(continuous)
	3.在剩下的不连续变化中，采取如下投票规则(continuous_dic)：
		1）对于各类地物占比不一致的点，从占比最多的地物最后出现位置向前替换
		2）对与各类地物占比一致的点，分类进行讨论：
			1】 连续子列个数>2,且最长子列唯一：按子列位置向前/向后替换，剩余投票替换，若唯一子列出现在中间，则往前替换，剩余用最后一个位置的替换
			2】 连续子列个数>2,且最长子列不唯一：按子列位置向前/向后替换
			3】 无连续子列：用首尾地物对半开

	一些问题：
		1. 最后的 “失稳” 是由【为了保留变化可能性的同时所带来的不稳定性】所引起的，暂无合适解决方法
		2. 程序目前只考虑并处理了时间序列长度为6时的情况，暂不能增加时间序列进行处理，原因是不同的最长子列组合将会变多，现有代码已经很臃肿了。
'''



#生成时间序列
TimeLine = {}

for i in range(65):
    for j in range(78):
        TimeLine['{}_{}'.format(i,j)] = []

for i in range(len(file_Paths)):
	point_List = []
	ReadFile(file_Paths[i],TimeLine)

stable_dic = {}

#筛选稳定点
for i in TimeLine:
	print(i)
	print(TimeLine[i])
	count = {}
	for j in TimeLine[i]:
		count[j] = TimeLine[i].count(j)
	if len(count) == 1:
		stable_dic[i] = TimeLine[i]

for i in stable_dic:
	TimeLine.pop(i)
landclass = ['Farmland','Industrial','Residential','Woods','RiverLake','Rural']

#筛选连续点
continuous = {}
for item in TimeLine:
	flag = True
	print(item)
	print(TimeLine[item])
	for i in landclass:
		index = [j for j,x in enumerate(TimeLine[item]) if x == i]
		if index:
			if index[0] + (len(index)-1) != index[len(index)-1]:
				flag = False
				break
	if flag: continuous[item] = TimeLine[item]

for i in continuous:
	TimeLine.pop(i)

print(len(continuous)+len(stable_dic))
#寻找最长子列
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

#比较子列长度
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

#按照占比转为连续点
continuous_dic = {}
number = 0
for i in TimeLine:
	count = {}
	for j in TimeLine[i]:
		count[j] = TimeLine[i].count(j)
	flag_equal = False
	keys = list(count.keys())
	temp_line = copy.deepcopy(TimeLine[i])

	for item in temp_line:
			index = [j for j,x in enumerate(temp_line) if x == item]

			if len(index) > 1 and index[0] + len(index) - 1 != index[len(index)-1]:
				for k in range(len(temp_line)):
					if k > index[0] and k < index[len(index) -1]:
						temp_line[k] = item
	print(temp_line)

	if temp_line.count(temp_line[0]) == years:
		stable_dic[i] = temp_line
	else:
		continuous[i] = temp_line
# 	# count_order = sorted(count.items(),key = lambda x:x[1],reverse = True)

# 	# # if len(count_order) == 3 and count_order[0][1] == count_order[1][1]:
# 	# # 	temp_line = copy.deepcopy(TimeLine[i])
# 	# # 	index1 = [j for j,x in enumerate(TimeLine[i]) if x == count_order[0][0]]
# 	# # 	index2 = [j for j,x in enumerate(TimeLine[i]) if x == count_order[1][0]]
# 	# # 	stride1 = index1[len(index1)-1] - index1[0]
# 	# # 	stride2 = index2[len(index2)-1] - index2[0]

# 	# # 	if stride1 >= stride2:
# 	# # 		for k in range(years):
# 	# # 			if k <= index1[len(index1)-1] and k >= index1[0]:
# 	# # 				temp_line[k] = count_order[0][0]
# 	# # 	else:
# 	# # 		for k in range(years):
# 	# # 			if k <= index2[len(index2)-1] and k >= index2[0]:
# 	# # 				temp_line[k] = count_order[0][0]
# 	# # 	continuous_dic[i] = temp_line
# 	# # else:
# 	# # 	temp_line = copy.deepcopy(TimeLine[i])
# 	# # 	print(i)
# 	# # 	print(TimeLine[i])
# 	# # 	print(count_order)
# 	# # 	index = [j for j,x in enumerate(TimeLine[i]) if x == count_order[0][0]]
# 	# # 	print(index)
# 	# # 	for k in range(years):
# 	# # 		if k <= index[len(index)-1]:
# 	# # 			temp_line[k] = count_order[0][0]

# 	# # 	print(temp_line)
# 	# # 	continuous_dic[i] = temp_line

	# for n in range(len(keys) - 1):
	# 	if count[keys[n]] == count[keys[n+1]]:
	# 		flag_equal = True
	# 	else:
	# 		flag_equal = False
	# 		break

	# if not flag_equal:
	# 	print(i)
	# 	print(TimeLine[i])
	# 	count_order = sorted(count.items(),key = lambda x:x[1],reverse = True)
	# 	index = [j for j,x in enumerate(TimeLine[i]) if x == count_order[0][0]]
	# 	temp_line = copy.deepcopy(TimeLine[i])

	# 	# for k in range(years):
	# 	# 	if k < index[len(index)-1]:
	# 	# 		temp_line[k] = count_order[0][0]
	# 	for item in temp_line:
	# 		index = [j for j,x in enumerate(temp_line) if x == item]

	# 		if len(index) > 1 and index[0] + len(index) - 1 != index[len(index)-1]:
	# 			for k in range(len(temp_line)):
	# 				if k > index[0] and k < index[len(index) -1]:
	# 					temp_line[k] = item
	# 	print(temp_line)

# 		if temp_line.count(temp_line[0]) == years:
# 			stable_dic[i] = temp_line
# 		else:
# 			continuous[i] = temp_line
# 	else:
# 		print(i)
# 		print(TimeLine[i])
# 		conti_Index = {}
# 		for c in landclass:
# 			index = [j for j,x in enumerate(TimeLine[i]) if x == c]
# 			if index:
# 				conti_Index[c] = FindLongList(index)

# 		flag_zero = False
# 		flag_len = True
# 		flag_more = 0
# 		current_len = len(conti_Index[list(conti_Index.keys())[0]])
# 		for j in conti_Index:
# 			if conti_Index[j] == [0]:
# 				flag_zero = True
# 			if len(conti_Index[j]) != current_len:
# 				flag_len = False
# 			if len(conti_Index[j]) > 1:
# 				flag_more += 1
# 		max_order = sorted(conti_Index.items(),key = lambda x:x[1][0],reverse = False)
# 		pre_list = max_order[0][1]
# 		pre_class = max_order[0][0]
# 		nex_class = max_order[1][0]

# 		if (not flag_zero):

# 			temp_list = []

# 			for k in range(years):
# 				if k <= pre_list[len(pre_list)-1]:
# 					temp_list.append(pre_class)
# 				else:
# 					temp_list.append(nex_class)
# 			continuous_dic[i] = temp_list
# 		elif flag_len:
# 			temp_list = []
# 			for k in range(years):
# 				if k < 3:
# 					temp_list.append(TimeLine[i][0])
# 				else:
# 					temp_list.append(TimeLine[i][len(TimeLine[i]) - 1])

# 			continuous_dic[i] = temp_list
# 		elif flag_more > 1:

# 			temp_key = []
# 			for m in range(len(conti_Index)):
# 				if conti_Index[list(conti_Index.keys())[m]] == [0]:
# 					temp_key.append(list(conti_Index.keys())[m])

# 			for n in temp_key:
# 				conti_Index.pop(n)

# 			max_order = sorted(conti_Index.items(),key = lambda x:x[1][0],reverse = False)
# 			pre_list = max_order[0][1]
# 			pre_class = max_order[0][0]
# 			nex_class = max_order[1][0]

# 			temp_list = []

# 			for k in range(years):
# 				if k <= pre_list[len(pre_list)-1]:
# 					temp_list.append(pre_class)
# 				else:
# 					temp_list.append(nex_class)
# 			continuous_dic[i] = temp_list
# 		else:
# 			max_order = sorted(conti_Index.items(),key = lambda x:x[1][0],reverse = True)
# 			pre_list = max_order[0][1]
# 			pre_class = max_order[0][0]
# 			nex_class = max_order[1][0]

# 			temp_list = []

# 			if len(conti_Index) == 2:
# 				for k in range(years):
# 					if k < 3:
# 						temp_list.append(pre_class) if pre_list[0] < 2 else temp_list.append(nex_class)
# 					else:
# 						temp_list.append(nex_class) if pre_list[0] < 2 else temp_list.append(pre_class)
# 				continuous_dic[i] = temp_list
# 			else:
# 				max_order = sorted(conti_Index.items(),key = lambda x:len(x[1]),reverse = True)

# 				temp_list = []
# 				if max_order[0][1][0] < 2:
# 					sublist = TimeLine[i][-4:-1]
# 					for k in range(years):
# 						if k < 3:
# 							temp_list.append(max_order[0][0])
# 						else:
# 							temp_count = {}
# 							for m in sublist:
# 								temp_count[m] = sublist.count(m)
# 							temp_order = sorted(temp_count.items(),key = lambda x:x[1],reverse = True)
# 							temp_list.append(temp_order[0][0])
# 					continuous_dic[i] = temp_list
# 				elif max_order[0][1] == [2,3]:
# 					for k in range(years):
# 						if k < 4:
# 							temp_list.append(max_order[0][0])
# 						else:
# 							temp_list.append(TimeLine[i][len(TimeLine[i]) - 1])
# 					continuous_dic[i] = temp_list
# 				else:
# 					sublist = TimeLine[i][0:3]
# 					for k in range(years):
# 						if k >= 3:
# 							temp_list.append(max_order[0][0])
# 						else:
# 							temp_count = {}
# 							for m in sublist:
# 								temp_count[m] = sublist.count(m)
# 							temp_order = sorted(temp_count.items(),key = lambda x:x[1],reverse = True)
# 							temp_list.append(temp_order[0][0])
# 					continuous_dic[i] = temp_list

print('稳定点:' + str(len(stable_dic)))
print('第一连续点：' + str(len(continuous)))
print('第二连续点：' + str(len(continuous_dic)))
print('总数：' + str(len(stable_dic)+len(continuous)+len(continuous_dic)))
#合并结果
newTimeLine={}
newTimeLine.update(stable_dic)
newTimeLine.update(continuous)
newTimeLine.update(continuous_dic)

p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []

#生成新时间序列
for i in newTimeLine:
	p1.append(newTimeLine[i][0] + '_' + i)
	p2.append(newTimeLine[i][1] + '_' + i)
	p3.append(newTimeLine[i][2] + '_' + i)
	p4.append(newTimeLine[i][3] + '_' + i)
	p5.append(newTimeLine[i][4] + '_' + i)
	p6.append(newTimeLine[i][5] + '_' + i)

with open("newTimeLine.txt","w") as f:
    for i in newTimeLine:
        f.writelines(i + ' {}'.format(newTimeLine[i]) + '\n')

# with open("newTimeLine.txt","w") as f:
#     for i in TimeLine:
#         f.writelines(i + ' {}'.format(TimeLine[i]) + '\n')

#绘制散点图
# DrawScatter(p1,'20161209')
# DrawScatter(p2,'20171109')
# DrawScatter(p3,'20181129')
# DrawScatter(p4,'20191114')
# DrawScatter(p5,'20201118')
# DrawScatter(p6,'20211223')
subList1 = list(set(p1)-set(p6))
subList2 = list(set(p6)-set(p1))
# DrawScatter(subList1,'20161209')
DrawScatter(subList2,'20211223')