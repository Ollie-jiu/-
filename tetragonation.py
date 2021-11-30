import cv2
from numpy.lib.function_base import append, select
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from shapely.geometry import Polygon, LinearRing #多边形模型，和线性环模型
import math


def IsSimplePoly(poly):
    """判断多边形poly是否为简单多边形"""
    poly_ring = poly.boundary
    if poly_ring.is_ring and list(poly.interiors) == []:
        return True
    else:
        return False
def GetPolyVex(poly):
    """得到poly的顶点序列，以numpy数组的形式返回
       :首尾点重合，该数组为n*2的数组
    """
    return np.asarray(poly.exterior)
def VexCCW(poly):
    """判断poly的顶点给出的是顺时针顺序还是逆时针顺序
       :若给出的顶点为逆时针排列则返回1，为顺时针旋转则返回-1
    """
    return 1 if LinearRing(poly.exterior).is_ccw else -1
def GetDivideVexIdx(poly):
    """得到poly中的可划分顶点的下标序列
         :返回1，无重复顶点np数组，顺序与poly.exterior中的顺序相同
         :返回2，其中可划分顶点的下标序列
         :返回3，可划分顶点的多在角的弧度序列
    """
    dividevex_idx_li = [] #存储可划分顶点的下标
    dividevex_arg_li = [] #存储可划分顶点所对应角的弧度值
    vex_arr = GetPolyVex(poly) #顶点序列
    vex_arr = vex_arr[:-1,:] #去掉最后一个回环
    nums = vex_arr.shape[0] #顶点序列的个数
    if nums <= 3: #三角形则不用再处理
        return vex_arr, dividevex_idx_li, dividevex_arg_li
    
    pm = VexCCW(poly) #poly的顺逆时针状态
    for i in range(nums):
        v = vex_arr[i,:]#当前顶点
        l = vex_arr[i-1,:]#前驱顶点
        r = vex_arr[(i+1)%nums,:]#后继顶点
        fir_vector = v - l #用有向面积法计算是否为凸顶点
        sec_vector = r - v
        A = np.array([fir_vector,sec_vector]) #判断矩阵
        if pm*np.linalg.det(A) > 0:#此时的顶点为凸顶点，在此基础上判断其是否为可划分顶点
            remainvex_arr = np.concatenate([vex_arr[:i,:],vex_arr[i+1:,:]],axis=0)
            remain_poly = Polygon(remainvex_arr)
            tri = Polygon([l,v,r])
            if (remain_poly.is_valid
                and remain_poly.intersection(tri).area < 1e-8 #为一个可调整系数
                and poly.equals(remain_poly.union(tri))):#判断一个凸顶点是否为可划分顶点的依据
                
                dividevex_idx_li.append(i) #将可划分的顶点下标压入序列
                #下面计算对应的弧度
                arc = np.arccos(-np.dot(fir_vector,sec_vector)/np.linalg.norm(fir_vector)/np.linalg.norm(sec_vector))
                dividevex_arg_li.append(arc)
    return vex_arr, dividevex_idx_li, dividevex_arg_li
def GetDivTri(poly, tris = []):
    """递归的将多边形，进行三角剖分，每次都以角度最小的可划分顶点为依据"""
    vex_arr, dv_idx_li, dv_arc_li = GetDivideVexIdx(poly)
    nums = vex_arr.shape[0]
    if nums <= 3: #三角形，则直接处理
        tris.append(poly)
        return tris
    idx = dv_idx_li[np.argmin(np.array(dv_arc_li))]#取出最小的一个可划分顶点的下标
    #idx = dv_idx_li[np.random.randint(len(dv_idx_li))]#随机取出一个下标
    v = vex_arr[idx, :]
    l = vex_arr[idx-1, :]
    r = vex_arr[(idx+1)%nums, :]
    tri = Polygon([l,v,r]) #划分出来的三角形
    tris.append(tri) #将这个处理好的三角形压入序列
    #下面为得到新序列，并转化为图形，用于递归
    remain_vex_arr = np.concatenate([vex_arr[:idx,:],vex_arr[idx+1:,:]],axis=0)
    remain_poly = Polygon(remain_vex_arr)
    GetDivTri(remain_poly,tris)
    return tris
def PolyPretreatment(poly_arr):
    """用于对poly_arr进行归一化处理"""
    temp = poly_arr - np.min(poly_arr,axis=0)
    return temp / np.max(temp)
def MinAngle(tri):
    """计算一个三角形的最小角的弧度[0,pi/2]"""
    point = np.asarray(tri.exterior)
    arc_li = []
    for i in range(3):
        j = (i+1)%3; k=(i+2)%3
        a = np.linalg.norm(point[i,:] - point[j,:])
        b = np.linalg.norm(point[j,:] - point[k,:])
        c = np.linalg.norm(point[k,:] - point[i,:])
        arc = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
        arc_li.append(arc)
    return min(arc_li)
def OptDiv(poly4_vex_arr):
    """对四边形进行优化划分，返回其最优化的两个三角形"""
    tri1 = Polygon(poly4_vex_arr[[0,1,2]])
    tri2 = Polygon(poly4_vex_arr[[0,2,3]])
    arc1 = min([MinAngle(tri1),MinAngle(tri2)])

    tri3 = Polygon(poly4_vex_arr[[0,1,3]])
    tri4 = Polygon(poly4_vex_arr[[1,2,3]])
    arc2 = min([MinAngle(tri3),MinAngle(tri4)])

    if arc1 >= arc2:
        return tri1,tri2
    else:
        return tri3,tri4
def OptAlltris(tris):
    """对已经给出的三角剖分进行进一步的优化，使得最小角最大
        :对剖分出的三角形序列进行优化
        :通常需要运行两次，才能保证充分优化
    """
    random.shuffle(tris)
    nums = len(tris)
    for i in range(nums):
        tri_i = tris[i]
        for j in range(i+1,nums):
            tri_j = tris[j]
            if tri_i.intersection(tri_j).length > 1e-10:
                u = tri_i.union(tri_j)
                vex_arr, dv_vex_li, _=GetDivideVexIdx(u)
                if len(dv_vex_li) == 4:
                    a,b = OptDiv(vex_arr)
                    flag = True
                    for idx in set(range(nums)) - {i,j}:
                        if a.intersection(tris[idx]).area > 0. or b.intersection(tris[idx]).area > 0.:
                            flag = False
                    if flag:
                        tris[i],tris[j] = a,b
    return tris



#the annotating code is my version, which is to directly split polygon,but it is not very effective for some situation. 





# def if_isJunction(rangeList, Point):
#     #if the point x,y are bigger than MaxX or MaxY or smaller than MinX or MinY return false
#     #to test whether there is a junction of lines;
#     xList=[]
#     yList=[]
#     for i in rangeList:
#         xList.append(i[0])
#         yList.append(i[1])
#     MinX=min(xList)
#     MaxX=max(xList)
#     MinY=min(yList)
#     MaxY=max(yList)
#     # print("look",Point)
#     if(Point[0]>MaxX or Point[0]<MinX or Point[1]>MaxY or Point[1]<MinY):
        
#         return False
#     # print("What:")
#     temp=(Point[0]-rangeList[1][0])*(Point[0]-rangeList[2][0])
#     temp1=(Point[1]-rangeList[1][1])*(Point[1]-rangeList[2][1])
#     if(temp+temp1<0):#different direction
#         # print("i'm here")        
#         return True
#     else:
        
#         return False
# def calDistance(m,n):
#     return np.sqrt((points[m][0]-points[n][0])**2+(points[m][1]-points[n][1])**2)
# def findCrossPoint(p1,p2,p3,p4):#to find the crossPoint
#     #!!!this algorithm need to be  fixed in order to reduce cost
#     crossPoints=[]
#     d1=(points[p2][1]-points[p1][1])/(points[p2][0]-points[p1][0]+0.000000000000000000000000000000000000000001)
#     b=(points[p1][1]*points[p2][0]-points[p1][0]*points[p2][1])/(points[p2][0]-points[p1][0]+0.000000000000000000000000000000000000000001)
#     temp1=points[p3][1]*points[p4][0]-points[p3][0]*points[p4][1]
#     temp2=points[p2][0]-points[p1][0]
#     temp3=points[p1][1]*points[p2][0]-points[p1][0]*points[p2][1]
#     temp4=points[p4][0]-points[p3][0]
#     temp5=points[p2][1]-points[p1][1]
#     temp6=points[p4][0]-points[p3][0]
#     temp7=points[p4][1]-points[p3][1]
#     temp8=points[p2][0]-points[p1][0]
#     x=(temp1*temp2-temp3*temp4)/(temp5*temp6-temp7*temp8+0.000000000000000000000000000000000000000001)
#     y=x*d1+b
#     crossPoints.append(x)
#     crossPoints.append(y)
#     return crossPoints
# def findnNeighPoints(a):#only find two points in the array points(polyton)
#     neigh1=a-1
#     neigh2=a+1
#     if(neigh1<0):
#         neigh1=len(points)-1#last point
#     if(neigh2>len(points)):
#         neigh2=0#first points;
#     return neigh1,neigh2; 
# def isTetra(a,b,c,d):#0,1,2,3
#     crossP1=findCrossPoint(a,d,b,c)#0,3,1,2
#     crossP2=findCrossPoint(a,b,c,d)  #0,1,2,3
#     temp1=[]
#     temp1.append(points[a])
#     temp1.append(points[d])
#     temp1.append(points[b])
#     temp1.append(points[c])
#     temp2=[]
#     temp2.append(points[a])
#     temp2.append(points[b])
#     temp2.append(points[c])
#     temp2.append(points[d])
#     if(if_isJunction(temp1, crossP1)==False and (if_isJunction(temp2, crossP2)==False)):
#         return True
#     else:
#         return False
# def findMinPoint(firstPoint,lastPoint,array,tetra):
#     #it needs index numbers to find the position in points
#     if(firstPoint==lastPoint):
#         neigh1,neigh2=findnNeighPoints(firstPoint)
#         dist0=calDistance(neigh1,firstPoint)
#         dist1=calDistance(neigh2,lastPoint)
#         if(dist0<dist1):
#             return neigh1,0
#         else:
#             return neigh2,1
#     elif(firstPoint!=lastPoint):
#         temp1,temp2=findnNeighPoints(firstPoint)
#         if(temp1 in array and temp2 in array):
#             return "error",3
#         elif(temp1 not in array):
#             neigh1=temp1
#         else:
#             neigh1=temp2
#         temp3,temp4=findnNeighPoints(lastPoint)
#         #if all not in array:
#         if(temp3 in array and temp4 in array):
#             return "error",3
#         elif(temp3 not in array):
#             neigh2=temp3
#         else:
#             neigh2=temp4
# #---------------------------------------------------------------------#
#     #which one can be a point of tetragon #
#         a=calDistance(firstPoint,neigh1)
#         b=calDistance(lastPoint,neigh2)
#         if(len(tetra)<3):
#             if(a<b):
#                 return neigh1,0
#             else:
#                 return neigh2,1
#         elif(len(tetra)==3):
#             # print("heiehiei:\n",isTetra(neigh1,tetra[0],tetra[1],tetra[2]),isTetra(tetra[0],tetra[1],tetra[2],neigh2),"\n")
#             if(isTetra(neigh1,tetra[0],tetra[1],tetra[2])==True  and isTetra(tetra[0],tetra[1],tetra[2],neigh2)==True):
#                 print("\n1\n")
#                 print("\nlook:",neigh1,neigh2,"\n")
#                 #calculate their areas to decide which one is better;
#                 temp1=[points[neigh1],points[tetra[0]],points[tetra[1]],points[tetra[2]]]
#                 temp2=[points[tetra[0]],points[tetra[1]],points[tetra[2]],points[neigh2]]
#                 temp1=np.array(temp1)
#                 temp2=np.array(temp2)
#                 minRect1=cv2.minAreaRect(temp1)
#                 minRect2=cv2.minAreaRect(temp2)
#                 neigh1Area=minRect1[1][0]*minRect1[1][1]
#                 neigh2Area=minRect2[1][0]*minRect2[1][1]
#                 if(neigh1Area<neigh2Area):
#                     return neigh1,0
#                 else:
#                     return neigh2,1
#             elif(isTetra(neigh1,tetra[0],tetra[1],tetra[2])==True and isTetra(tetra[0],tetra[1],tetra[2],neigh2)==False):
#                 print("\n2\n")
#                 print("\nlook:",neigh1,neigh2,"\n")
#                 return neigh1,0
#             elif(isTetra(neigh1,tetra[0],tetra[1],tetra[2])==False and isTetra(tetra[0],tetra[1],tetra[2],neigh2)==True):
#                 print("\n3\n")
#                 print("\nlook:",neigh1,neigh2,"\n")
#                 return neigh2,1
#             elif(isTetra(neigh1,tetra[0],tetra[1],tetra[2])==False and isTetra(tetra[0],tetra[1],tetra[2],neigh2)==False):
#                 print("\n4\n")
#                 print("\nlook:",neigh1,neigh2,"\n")
#                 if(a<b):
#                     return neigh1,0
#                 else:
#                     return neigh2,1
#             else:
#                 print("\n5\n")
#                 print("\nlook:",neigh1,neigh2,"\n")
#                 return neigh1,0 or neigh2,1
#----------------------------------------------------------------------------------
        # if(a<b):
        #     if(len(tetra)<3):
        #         return neigh1,0
        #     elif(len(tetra)==3 and isTetra(neigh1,tetra[0],tetra[1],tetra[2])):
        #         return neigh1,0
        #     else:
        #         return neigh2,1
        # else:
        #     if(len(tetra)<3):
        #         return neigh2,1
        #     elif(len(tetra)==3 and isTetra(tetra[0],tetra[1],tetra[2],neigh2)):
        #         return neigh2,1
        #     else:
        #         return neigh1,0
img = cv2.imread("newMask.png", 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
maxContours=max(contours,key=len,default='')
# print(maxContours)

points = cv2.approxPolyDP(maxContours,20,True)#change it into a poly   30   20

points=points.squeeze()
cv2.polylines(img, [points], True, (110, 110, 200), 2)
#---
xForPlt=[]
yForPlt=[]
for i in range(len(points)):
    xForPlt.append(points[i][0])
minX=min(xForPlt)
maxX=max(xForPlt)
for i in range(len(points)):
    yForPlt.append(points[i][1])
minY=min(yForPlt)
maxY=max(yForPlt)

#---
# cv2.polylines(img, [points], True, (110, 110, 200), 2)
def reArrangeSenPoints(Points):
    y=[]
    newOrder=points.squeeze().tolist()   
    for i in range(len(Points)):
        y.append(Points[i][1])
    MaxY=max(y)
    while 1:
        if(newOrder[1][1]==MaxY):
            break
        temp=newOrder.pop(0)
        newOrder.append(temp)
    return np.array(newOrder)
    
points=reArrangeSenPoints(points)

print(points)

##-----------------------------------------------##
poly = Polygon(PolyPretreatment(points)) #构造多边形
#运算,绘图脚本
if IsSimplePoly(poly):
    plt.figure(figsize=(8,8))
    tris = []
    tris =  GetDivTri(poly,tris = tris)
    #用mpl画出，原来图形的线框
    plt.subplot(2,2,1)
    plt.plot(*poly.exterior.xy)
    plt.axis("equal")
    #用线框画出剖分
    plt.subplot(2,2,2)
    for tri in tris: #triangulate得到的所有三角形，这是对凸包的一个划分
        plt.plot(*tri.exterior.xy)
    plt.axis("equal")
    poi=np.array(tris)
    # print("points:\n",poi,"\n")
    newTriaPoints=[]
    TriaPointForProcessing=[]
    poisX=[]
    poisY=[]
    for pois in poi:
        # newPoi=np.array(pois)
        # print("points:\n",pois.exterior.coords.xy,"\n")
        poisX.append(pois.exterior.coords.xy[0][0])
        poisX.append(pois.exterior.coords.xy[0][1])
        poisX.append(pois.exterior.coords.xy[0][2])
        poisY.append(pois.exterior.coords.xy[1][0])
        poisY.append(pois.exterior.coords.xy[1][1])
        poisY.append(pois.exterior.coords.xy[1][2])
    minPoisX=min(poisX)
    maxPoisX=max(poisX)
    minPoisY=min(poisY)
    maxPoisY=max(poisY)
    print("haudhioasdhiasd:",(maxPoisX-minPoisX),(maxPoisY-minPoisY))
    for pois in poi:
        # newPoi=np.array(pois)
        # print("points:\n",pois.exterior.coords.xy,"\n")
        temp=[]
        temp.append([(minX+pois.exterior.coords.xy[0][0]/(maxPoisX-minPoisX)*(maxX-minX))/448*500,(minY+pois.exterior.coords.xy[1][0]/(maxPoisY-minPoisY)*(maxY-minY))/448*500])
        temp.append([(minX+pois.exterior.coords.xy[0][1]/(maxPoisX-minPoisX)*(maxX-minX))/448*500,(minY+pois.exterior.coords.xy[1][1]/(maxPoisY-minPoisY)*(maxY-minY))/448*500])
        temp.append([(minX+pois.exterior.coords.xy[0][2]/(maxPoisX-minPoisX)*(maxX-minX))/448*500,(minY+pois.exterior.coords.xy[1][2]/(maxPoisY-minPoisY)*(maxY-minY))/448*500])

        addTria=copy.deepcopy(temp)
        newTriaPoints.append(addTria)
        TriaPointForProcessing.extend(addTria)
    print("newTriaPoints:\n",newTriaPoints,"\n")
    plt.show()
else:
     print("输入的多边形，不是定义要求的简单多边形！")

#================================================
# f=0
# l=len(points)#this value is near f(first point)
# firstPoint=[]
# lastPoint=[]
# all=[]
# tempPoints=points.tolist()#np.array -> list 
# tempPoints.remove(points[0].tolist())
# tetra=[]
# tetra.append(0)
# mappedPoints=[]
# mappedPoints.append(0)

# while(1):
#     while(1):
#         temp,direction=findMinPoint(tetra[0],tetra[len(tetra)-1],mappedPoints,tetra)#only search by first and last one
#         if(temp=="error"):
#             print("Warning")
#         if(direction==0):
#             tetra.insert(0,temp)
#         elif(direction==1):
#             tetra.append(temp)
#         else:
#             print("error")
#         # print(tetra)
#         mappedPoints.append(temp)
#         tempPoints.remove(points[temp].tolist())
#         if(len(tetra)==4):
#             addTetra=copy.deepcopy(tetra)
#             break
#     all.extend(addTetra)
#     del tetra[1:3]#remove middle elements
#     if(len(tempPoints)==1):
#         tetra.append(tetra[1]+1)
#         all.extend(tetra)
#         break
#     elif(len(tempPoints)==0):
#         break
# print(all)
# for i in all:
#     print(points[i])
#=======================================================

#new version1.1(the effect of the last version is not very good)


#------------this is to merge triangles;-----------------
from functools import reduce
import operator
import math
newTetraPoints=[]

for i in newTriaPoints:
    # print("i:\n",i,"\n")
    for j in newTriaPoints:
        if i==j:
            pass
        else:
            a=[x for x in i if x in j]
            if(len(a)==2):
                temp=i+j
                temp1=[]
                for x in temp:
    	            if not x in temp1:
                        temp1.append(x)
                center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), temp1), [len(temp1)] * 2))
                temp1=sorted(temp1, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
                print("center:\n",center)
                addTetra=copy.deepcopy(temp1)
                #this to give them same order.
                if addTetra not in newTetraPoints: 
                    # TetraPointForProcessing.extend(addTetra)
                    newTetraPoints.append(addTetra)
print("Newtetragons:\n",newTetraPoints)