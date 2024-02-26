# a = ['Farmland','Rural','Farmland','Woods','Woods','Woods']

# def Substitude(timeline):
# 	for i in timeline:
# 		index = [j for j,x in enumerate(timeline) if x == i]

# 		if len(index) > 1 and index[0] + len(index) - 1 != index[len(index)-1]:
# 			print(index)
# 			for k in range(len(timeline)):
# 				if k > index[0] and k < index[len(index) -1]:
# 					timeline[k] = i

# Substitude(a)

#a = {'Rural_Rural': 4128, 'Woods_Woods': 3656, 'Industrial_Industrial': 3557, 'Residential_Residential': 3463, 'Farmland_Farmland': 2847, 'RiverLake_RiverLake': 1465, 'Rural_Farmland': 759, 'Farmland_Rural': 716, 'Woods_Rural': 449, 'Industrial_Residential': 429, 'Rural_Woods': 349, 'Residential_Industrial': 309, 'Farmland_RiverLake': 289, 'Farmland_Industrial': 276, 'RiverLake_Farmland': 260, 'Residential_Rural': 247, 'Rural_Residential': 210, 'Rural_Industrial': 194, 'Industrial_Farmland': 192, 'Woods_Farmland': 177, 'Farmland_Woods': 151, 'Industrial_Rural': 137, 'Residential_Farmland': 131, 'Farmland_Residential': 121, 'RiverLake_Rural': 106, 'Industrial_RiverLake': 93, 'RiverLake_Industrial': 87, 'Woods_RiverLake': 85, 'Rural_RiverLake': 83, 'Woods_Industrial': 74, 'RiverLake_Woods': 69, 'Residential_RiverLake': 58, 'RiverLake_Residential': 56, 'Woods_Residential': 50, 'Residential_Woods': 45, 'Industrial_Woods': 32}
#a = {'Rural_Rural': 4426, 'Woods_Woods': 3442, 'Industrial_Industrial': 3308, 'Farmland_Farmland': 3245, 'Residential_Residential': 3207, 'RiverLake_RiverLake': 1426, 'Industrial_Residential': 134, 'Rural_Farmland': 117, 'Farmland_Rural': 114, 'Woods_Rural': 87, 'Farmland_Industrial': 87, 'Rural_Woods': 63, 'Residential_Rural': 62, 'Rural_Industrial': 60, 'Farmland_RiverLake': 49, 'RiverLake_Farmland': 44, 'Residential_Industrial': 42, 'Industrial_Farmland': 39, 'Woods_Farmland': 38, 'Farmland_Woods': 31, 'Rural_Residential': 26, 'Residential_Farmland': 26, 'Industrial_Rural': 26, 'Woods_Industrial': 25, 'Farmland_Residential': 21, 'Industrial_RiverLake': 21, 'RiverLake_Rural': 21, 'RiverLake_Industrial': 18, 'Residential_Woods': 13, 'RiverLake_Residential': 12, 'Woods_Residential': 11, 'Woods_RiverLake': 10, 'Rural_RiverLake': 8, 'Residential_RiverLake': 8, 'RiverLake_Woods': 7, 'Industrial_Woods': 6}
#a = {'Rural_Rural': 4248, 'Woods_Woods': 3591, 'Farmland_Farmland': 3295, 'Residential_Residential': 3249, 'Industrial_Industrial': 3199, 'RiverLake_RiverLake': 1429, 'Industrial_Residential': 126, 'Rural_Farmland': 114, 'Farmland_Rural': 109, 'Farmland_Industrial': 104, 'Woods_Rural': 91, 'Residential_Rural': 74, 'Rural_Industrial': 65, 'Farmland_RiverLake': 52, 'RiverLake_Farmland': 48, 'Residential_Industrial': 44, 'Rural_Woods': 43, 'Woods_Farmland': 38, 'Industrial_Farmland': 37, 'Residential_Farmland': 34, 'Rural_Residential': 33, 'Farmland_Woods': 30, 'Woods_Industrial': 29, 'RiverLake_Rural': 27, 'Industrial_Rural': 26, 'Industrial_RiverLake': 25, 'Farmland_Residential': 22, 'RiverLake_Industrial': 20, 'Woods_Residential': 13, 'RiverLake_Residential': 13, 'Residential_Woods': 13, 'Residential_RiverLake': 10, 'Woods_RiverLake': 9, 'Rural_RiverLake': 8, 'Industrial_Woods': 6, 'RiverLake_Woods': 6}
#a = {'Rural_Rural': 5223, 'Woods_Woods': 4348, 'Residential_Residential': 4082, 'Farmland_Farmland': 4013, 'Industrial_Industrial': 3950, 'RiverLake_RiverLake': 1768, 'Rural_Farmland': 200, 'Industrial_Residential': 186, 'Woods_Rural': 173, 'Farmland_Rural': 160, 'Farmland_Industrial': 145, 'Rural_Industrial': 93, 'Residential_Rural': 90, 'Farmland_RiverLake': 80, 'Residential_Industrial': 70, 'Rural_Woods': 67, 'Woods_Farmland': 65, 'Industrial_Farmland': 60, 'RiverLake_Farmland': 57, 'Rural_Residential': 56, 'Woods_Industrial': 49, 'Residential_Farmland': 45, 'Farmland_Woods': 44, 'Industrial_Rural': 41, 'Industrial_RiverLake': 37, 'Woods_RiverLake': 36, 'RiverLake_Rural': 34, 'Farmland_Residential': 32, 'RiverLake_Industrial': 32, 'Rural_RiverLake': 22, 'Woods_Residential': 20, 'Residential_Woods': 18, 'RiverLake_Residential': 17, 'RiverLake_Woods': 17, 'Residential_RiverLake': 12, 'Industrial_Woods': 8}
a = {'Farmland_Farmland': 565, 'RiverLake_RiverLake': 288, 'Woods_Woods': 705, 'Residential_Residential': 675, 'Industrial_Industrial': 564, 'Rural_Rural': 724, 'Woods_Rural': 169, 'Farmland_Industrial': 96, 'RiverLake_Farmland': 48, 'Rural_Woods': 58, 'Farmland_RiverLake': 58, 'Industrial_Farmland': 47, 'Rural_Farmland': 133, 'Industrial_Rural': 26, 'Rural_Industrial': 101, 'Residential_Rural': 74, 'Farmland_Woods': 26, 'Farmland_Rural': 136, 'Residential_Farmland': 34, 'Woods_Industrial': 38, 'Residential_Industrial': 62, 'RiverLake_Industrial': 17, 'Rural_Residential': 53, 'Industrial_Residential': 155, 'Woods_Residential': 18, 'Farmland_Residential': 27, 'Woods_RiverLake': 25, 'Woods_Farmland': 47, 'Industrial_RiverLake': 25, 'Rural_RiverLake': 17, 'RiverLake_Rural': 17, 'RiverLake_Residential': 9, 'Residential_Woods': 7, 'Residential_RiverLake': 9, 'RiverLake_Woods': 13, 'Industrial_Woods': 4}
total = {'Farmland':0,'Rural':0,'Industrial':0,'Residential':0,'Woods':0,'RiverLake':0}

label = ['Farmland','Rural','Industrial','Residential','Woods','RiverLake']
for i in label:
	for j in a:
		if j.split('_')[0] == i:
			total[i] += a[j]

totalchanges = 0
for i in total:
	totalchanges += total[i]

print(totalchanges)
print(total)

headline = '{:15}'.format(' ')
for i in label:
	headline += '{:^15}'.format(i)
print(headline)

for i in label:
	line = '{:15}'.format(i)
	for j in label:
		line += '{:^15.2f}'.format(a[i+'_'+j])
	print(line)