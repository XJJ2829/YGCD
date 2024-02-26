import matplotlib.pyplot as plt
import os
import seaborn as sns

point_List = []
point_List2 = []

def ReadFile(filename,point_List):
    with open(filename, "r") as f:
    	for line in f.readlines():
    		line = line.strip('\n')  #去掉列表中每一个元素的换行符
    		point_List.append(line)

ReadFile('20161209.txt',point_List)
ReadFile('20211223.txt',point_List2)

subList1 = list(set(point_List)-set(point_List2))
subList2 = list(set(point_List2)-set(point_List))

def DrawScatter(pointList,date):
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	size = 3
	
	farmland_X = []
	farmland_Y = []
	
	industrial_X = []
	industrial_Y = []
	
	residential_X = []
	residential_Y = []
	
	riverlake_X = []
	riverlake_Y = []
	
	woods_X = []
	woods_Y = []
	
	rural_X = []
	rural_Y = []
	
	for i in range(len(pointList)):
		point = pointList[i].split('.')[0]
		point_Class = point.split('_')[0]
		point_X = abs(int(point.split('_')[1]) - 64) * 3 + 2
		point_Y = int(point.split('_')[2]) * 3 + 2
	
		# if point_Class == 'Farmland':
		# 	farmland_X.append(point_X)
		# 	farmland_Y.append(point_Y)
	
		# if point_Class == 'Industrial':
		# 	industrial_X.append(point_X)
		# 	industrial_Y.append(point_Y)
	
		if point_Class == 'Residential':
			residential_X.append(point_X)
			residential_Y.append(point_Y)
	
		# if point_Class == 'RiverLake':
		# 	riverlake_X.append(point_X)
		# 	riverlake_Y.append(point_Y)
	
		# if point_Class == 'Woods':
		# 	woods_X.append(point_X)
		# 	woods_Y.append(point_Y)
	
		# if point_Class == 'Rural':
		# 	rural_X.append(point_X)
		# 	rural_Y.append(point_Y)
	
	ax1.set_title(date)
	# ax1.scatter(rural_Y, rural_X, c='k', marker='.')
	ax1.scatter(residential_Y, residential_X, c='c', marker='.')
	# ax1.scatter(riverlake_Y, riverlake_X, c='skyblue', marker='.')
	# ax1.scatter(woods_Y, woods_X, c='g', marker='.')
	# ax1.scatter(farmland_Y, farmland_X, c='yellowgreen', marker='.')
	# ax1.scatter(industrial_Y, industrial_X, c='b', marker='.')
	

	# sns.set_style("white")
	# p1 = sns.kdeplot(x = industrial_Y, y = industrial_X,cmap="Reds", shade=True, bw=.15)
	# p1 = sns.kdeplot(x = residential_Y, y = residential_X,cmap="Reds", shade=True, bw=.15)
	# p1 = sns.kdeplot(x = woods_Y, y = woods_X,cmap="Reds", shade=True, bw=.15)
	# p1 = sns.kdeplot(x = farmland_Y, y = farmland_X,cmap="Reds", shade=True, bw=.15)
	plt.show()