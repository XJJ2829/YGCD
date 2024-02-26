# import os
# import cv2
from PIL import Image
import os
import ast
import random

# source_dir = r'E:\12class_tif'  # 源tiff图路径
# target_dir = r'E:\BaiduNetdiskDownload\12class_tif'  # 保存到的jpg路径
# dirs = os.listdir(source_dir)

# for image_dir in dirs:
# 	files = os.listdir(os.path.join(source_dir, image_dir))
# 	index = 1
# 	for image_file in files:	
# 		name = []
# 		portion = os.path.splitext(image_file)  # 把文件名拆分为名字和后缀
# 		if portion[1] == ".tif":
# 			name = image_dir + '_' + str(index)
# 			image_path = source_dir + "\\" + image_dir + "\\" + image_file
# 			image_name = target_dir + "\\" + image_dir + "\\" + name + ".jpg"
# 			print(image_path)
# 			print(image_name)
# 			img = cv2.imread(image_path)
# 			cv2.imwrite(image_name, img)  # 图片存储
# 			index += 1

# imagePath = r'E:\Research\遥感影像分类+空间点模式\Test\Data\ChangSha\errors\error_newChangSha'
# imagesList = os.listdir(imagePath)
# errorList = []
# for i in range(len(imagesList)):
# 	nameList = imagesList[i].split('_')
# 	errorList.append(nameList[0]+'_'+nameList[1])

# a={}
# for i in errorList:
# 	a[i] =errorList.count(i)
# order = sorted(a.items(),key = lambda x:x[1],reverse = True)
# print(order)

# imagePath = 'Data/NWPU_T/train'
# imageList = os.listdir(imagePath)
# for i in imageList:
# 	if i[0:10] == 'industrial':
# 		#os.remove(os.path.join(imagePath,i))
# 		image = Image.open(os.path.join(imagePath,i))
# 		image.save(os.path.join(imagePath,i.split('_')[0]+'_'+i.split('_')[2]))
# 		os.remove(os.path.join(imagePath,i))

# imagePath = r'E:\Research\遥感影像分类+空间点模式\Test\Data\ChangSha\newtest\Woods'
# targetPath = r'E:\Research\遥感影像分类+空间点模式\Test\Data\ChangSha\newtest\t'
# imageDir = os.listdir(imagePath)
# for i in range(len(imageDir)):
# 	index = random.randint(0,len(imageDir)-1)
# 	image = Image.open(os.path.join(imagePath,imageDir[index]))
# 	image.save(os.path.join(targetPath,'Woods_'+str(i+1))+'.jpg')
# 	imageDir.remove(imageDir[index]) 

# image_Paths = [r'C:\Users\xxMartrix\Desktop\CS_DATA\sort20161209',r'C:\Users\xxMartrix\Desktop\CS_DATA\sort20171219',
# r'C:\Users\xxMartrix\Desktop\CS_DATA\sort20181129',r'C:\Users\xxMartrix\Desktop\CS_DATA\sort20191114',
# r'C:\Users\xxMartrix\Desktop\CS_DATA\sort20201118',r'C:\Users\xxMartrix\Desktop\CS_DATA\sort20211113']

# # image_Paths = [r'E:\TestImage\TimeList2\sort\sort20170227',r'E:\TestImage\TimeList2\sort\sort20180212',
# # r'E:\TestImage\TimeList2\sort\sort20190123',r'E:\TestImage\TimeList2\sort\sort20200217',
# # r'E:\TestImage\TimeList2\sort\sort20210112',r'E:\TestImage\TimeList2\sort\sort20220102']

# for i in range(len(image_Paths)):
# 	image_Path = image_Paths[i]
# 	image_Dir = os.listdir(image_Path)
	
# 	for j in range(len(image_Dir)):
# 		image_Dir[j] = image_Dir[j].split('_')[0]
	
# 	ratio = {}
# 	for j in image_Dir:
# 		ratio[j] = image_Dir.count(j)
	
	
# 	keys = list(ratio.keys())
# 	for j in range(len(ratio)):
# 		k = keys[j]
# 		ratio[k] = '{:.2f}%'.format( ratio[k] / len(image_Dir) * 100)
# 	print(ratio)

###文件夹迁移
# imagePath1 = r'E:\Research\CS_DATA\sample2020\All'
# imageList1 = os.listdir(imagePath1)
# imagePath2 = r'E:\Research\CS_DATA\sample2021\All'
# imageList2 = os.listdir(imagePath2)
# savePath = r'E:\Research\CS_DATA\sample2021'

# for i in imageList1:
# 	for j in imageList2:
# 		index1 = i.split('.')[0].split('_')[1] + '_' + i.split('.')[0].split('_')[2]
# 		folderName = i.split('.')[0].split('_')[0]
# 		index2 = j.split('.')[0]
# 		print(i,j)
# 		if index1 == index2:
# 			image = Image.open(os.path.join(imagePath2,j))
# 			savePath += '\{}'.format(folderName)
# 			print(savePath)
# 			image.save(os.path.join(savePath,folderName + '_' + j))
# 			savePath = r'E:\Research\CS_DATA\sample2021'
# 			break



#均匀采样
# for i in range(9):
# 	for j in range(13):
# 		index = str(8*i) + '_' + str(6*(j+1)-3)
# 		image = Image.open(os.path.join(r'E:\Research\CS_DATA\cs20211113',index + '.jpg'))
# 		image.save(os.path.join(r'E:\Research\CS_DATA\sample2021',index + '.jpg'))

# ####计算OA
# imagePath = r'E:\Research\CS_DATA\sample2021\All'
# imageList = os.listdir(imagePath)
# point_List = []

# with open('20211113.txt','r') as f:
# 	for line in f.readlines():
#     		line = line.strip('\n')  #去掉列表中每一个元素的换行符
#     		point_List.append(line)

# newTimeLine = {}

# with open('newTimeLine.txt', "r") as f:
# 	for line in f.readlines():
# 		line = line.strip('\n')
# 		lineList = line.split(' ')
# 		strList = lineList[1]+lineList[2]+lineList[3]+lineList[4]+lineList[5]+lineList[6]
# 		classList = ast.literal_eval(strList)
# 		newTimeLine[lineList[0]] = classList

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

# count = 0
# for i in imageList:
# 	list1 = i.split('.')[0].split('_')
# 	index1 = list1[1] + '_' + list1[2]
# 	class1 = list1[0]
# 	for j in p6:
# 		list2 = j.split('_')
# 		index2 = list2[1] + '_' + list2[2]
# 		class2 = list2[0]
# 		if index1 == index2:
# 			if class1 == class2:
# 				count+=1
# 			else:
# 				print(i,j)
# 			break
# print(count)

# 生成边界

newTimeLine = {}

with open('newTimeLine.txt', "r") as f:
	for line in f.readlines():
		line = line.strip('\n')
		lineList = line.split(' ')
		strList = lineList[1]+lineList[2]+lineList[3]+lineList[4]+lineList[5]+lineList[6]
		classList = ast.literal_eval(strList)
		newTimeLine[lineList[0]] = classList

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

sublist1 = list(set(p6) - set(p1))

leftE = 112.74778
rightE = 113.25056
botN = 27.99944
upN = 28.38306

with open('p3.txt','w') as f:
	for i in sublist1:
		pointarr = i.split('_')
		if pointarr[0] == 'Rural':
			rowCoor = upN - (upN - botN) / 65 * (int(pointarr[1]) +0.5)
			colCoor = leftE + (rightE - leftE) / 78 * (int(pointarr[2]) + 0.5)
			f.writelines(pointarr[0] + ',' + '{}'.format(rowCoor) + ',' + '{}'.format(colCoor) + '\n')