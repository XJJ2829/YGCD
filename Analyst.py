import os
import ast
from matplotlib import pyplot as plt
from Draw import DrawScatter
import copy

newTimeLine = {}                                # 创建一个空字典

# 读取时间序列的内容，并且逐行处理
with open('newTimeLine.txt', "r") as f:         # 使用with open语句以只读模式打开newTimeLine.txt，并将其赋值给变量f
	for line in f.readlines():                  # f.readlines()读取文件中每一行，其中每一行作为一个字符串元素存储到最终返回的列表当中，逐行处理每一行内容
		line = line.strip('\n')                 # 使用strip('\n')消除返回列表的每一个字符串元素后的换行符
		lineList = line.split(' ')              # 使用split(' ')分割字符串line，以空格为分隔符，将原本一个字符串的line分成了多个字符串，前面是序号，后面是对应的时间序列
		strList = lineList[1]+lineList[2]+lineList[3]+lineList[4]+lineList[5]+lineList[6]    # 索引0代表序号，索引1-6代表了年限间土地变化
		classList = ast.literal_eval(strList)   # strList已经把一个土地变化的时间序列组合成含有6个元素的列表，现在用这个函数解析出一个与strList对应的python列表对象
		newTimeLine[lineList[0]] = classList    # 将处理后的数据按照时间序列存储到newTimeLine字典中

farmland_Ratio = []
woods_Ratio = []
industrial_Ratio = []
residential_Ratio = []
riverlake_Ratio = []
rural_Ratio = []
year = [2016,2017,2018,2019,2020,2021]

def CalcRatio(newTimeLine):   # 计算各个分类在不同年份的比例
	for i in range(6):
		Farmland = 0
		Woods = 0
		Industrial = 0
		Residential = 0
		RiverLake = 0
		Rural = 0

		for j in range(65):                   # 对应着索引0-64
			for k in range(78):               # 对应着索引0-77
				ckey = '{}_{}'.format(j,k)
				tclass = newTimeLine[ckey][i]
				if tclass == 'Farmland':      # 得到各个土地类型的计数值
					Farmland += 1
				if tclass == 'Woods':
					Woods += 1
				if tclass == 'Industrial':
					Industrial += 1
				if tclass == 'Residential':
					Residential += 1
				if tclass == 'RiverLake':
					RiverLake += 1
				if tclass == 'Rural':
					Rural += 1
		farmland_Ratio.append(Farmland / 5070 * 100)
		woods_Ratio.append(Woods / 5070 * 100)
		industrial_Ratio.append(Industrial / 5070 * 100)
		residential_Ratio.append(Residential / 5070 * 100)
		riverlake_Ratio.append(RiverLake / 5070 * 100)
		rural_Ratio.append(Rural / 5070 * 100)

		print('Farmland: {:.2f}%, Woods: {:.2f}%, Industrial: {:.2f}%, Residential: {:.2f}%, RiverLake: {:.2f}%, Rural: {:.2f}%'.format(farmland_Ratio[i],woods_Ratio[i],industrial_Ratio[i],residential_Ratio[i],riverlake_Ratio[i],rural_Ratio[i]))

p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []

for i in newTimeLine:
	p1.append(newTimeLine[i][0] + '_' + i)
	p2.append(newTimeLine[i][1] + '_' + i)
	p3.append(newTimeLine[i][2] + '_' + i)
	p4.append(newTimeLine[i][3] + '_' + i)
	p5.append(newTimeLine[i][4] + '_' + i)
	p6.append(newTimeLine[i][5] + '_' + i)

sublist1 = list(set(p1) - set(p6))
sublist2 = list(set(p6) - set(p1))
sublist3 = list(set(sublist1) | set(sublist2))

label = ['Farmland','Rural','Industrial','Residential','Woods','RiverLake']

turnList = []
turn_dic = {}
#计算土地利用动态度、土地利用转移矩阵
def CalcTurn(sublist1,sublist2):
	for i in sublist1:
		for j in sublist2:
			if i.split('_')[1] + '_' + i.split('_')[2] == j.split('_')[1] + '_' + j.split('_')[2]:
				turnList.append(i.split('_')[0] + '_' + j.split('_')[0])
				break
	for k in turnList:
		turn_dic[k] = turnList.count(k)	
	print(turn_dic)

CalcTurn(p1,p6)
# CalcTurn(p2,p3)
plist = [p1,p2,p3,p4,p5,p6]
dynamicdegree = [[0 for i in range(5)] for i in range(6)]
for i in range(len(plist)-1):
	turnList = []
	turn_dic = {}
	total = {'Farmland':0,'Rural':0,'Industrial':0,'Residential':0,'Woods':0,'RiverLake':0}

	CalcTurn(plist[i],plist[i+1])
	for j in label:
		for k in turn_dic:
			if k.split('_')[0] == j:
				total[j] += turn_dic[k]

	totalchanges = 0
	for j in total:
		totalchanges += total[j]
	
	print(totalchanges)
	print(total)
	
	headline = '{:15}'.format(' ')
	for j in label:
		headline += '{:^15}'.format(j)
	print(headline)
	
	testList = []

	for j in label:
		line = '{:15}'.format(j)
		testsubList = []
		for k in label:
			line += '{:^15.2f}'.format(turn_dic.get(j+'_'+k,0))
			testsubList.append(turn_dic.get(j+'_'+k,0))
		testList.append(testsubList)
		print(line)

	for j in range(len(label)):
		copycolList = copy.deepcopy([row[j] for row in testList])
		changein = sum(copycolList)
		print(changein,total[label[j]])
		dynamic = (changein - total[label[j]]) / total[label[j]] * 100
		dynamicdegree[j][i] = dynamic
print(dynamicdegree)

#CalcRatio(newTimeLine)
# DrawScatter(sublist1,'2016')
# DrawScatter(sublist2,'2017')
# DrawScatter(sublist3,'2016-2017')