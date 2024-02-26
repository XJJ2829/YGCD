import cv2

img_path = 'E:/Research/images/changsha20161209.tif'
img = cv2.imread(img_path)

# for i in range(65):
# 	cv2.line(img,(0,64*(i+1)),(64*78,64*(i+1)),(0,255,0),1)

# for i in range(78):
# 	cv2.line(img,(64*(i+1),0),(64*(i+1),64*65),(0,255,0),1)

# font = cv2.FONT_HERSHEY_SIMPLEX

# for i in range(65):
# 	for j in range(78):
# 		text = str(i)+'_'+str(j)
# 		cv2.putText(img, text, (64*j, 64*i+32), font, 0.5, (0, 255, 0), 1)

for i in range(9):
	for j in range(13):
		coorX = 64*((j+1)*6 - 3) + 32
		coorY = 64*i*8 + 32
		cv2.circle(img,(coorX,coorY),20,(0,255,0),-1)
save_path = 'E:/Research/images/2016label.jpg'
cv2.imwrite(save_path, img)
