import cv2 as cv
import os
from statistics import mode
from pynput.keyboard import Key, Listener

run = True
path = "./trainImages"
images = []
className = []
myList = os.listdir(path)
orb = cv.ORB_create(nfeatures=1000)
print("total numebr of class detected are ",len(myList))
cap = cv.VideoCapture(0)

for cl in myList:
    imgCur = cv.imread(f'{path}/{cl}')
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0]) 

def findDesc(images):
    desList=[]
    for image in images:
        kp,des = orb.detectAndCompute(image,None)
        desList.append(des)
    return desList

desList = findDesc(images)

maxmode = []

def findID(img,desList,thr):
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:

        for des in desList:

            matches  = bf.knnMatch(des,des2,k=2) 
            good = []

            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])


            matchList.append(len(good))
    except:
        pass
    
    
    if len(matchList)!=0:
        if max(matchList)>thr:   
            finalVal = matchList.index(max(matchList))
            maxmode.append(finalVal)

    
    
    return finalVal,maxmode

show = -1
testcases = -1 
while run:
    success,img2  = cap.read()
    imgoriginal = img2.copy()
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    id1,id = findID(img2,desList,20)
    if id1!=-1:
        testcases += 1
        if len(id)>5:
            mod = mode(id)
            print(className[mod])
        if testcases > 15:
            show += 1  
    # if len(id)>6 and show>10:
    #     cv.putText(imgoriginal," ",(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),5)
        


    if show>0:
        cv.putText(imgoriginal,className[mod],(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),5)
    print(show,testcases)
    cv.imshow("image",imgoriginal)
    cv.waitKey(1)

    

cv.destroyAllWindows()





'''If Button == High From low Then Reset the thing'''
