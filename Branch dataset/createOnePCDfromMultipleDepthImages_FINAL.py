import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import pandas as pd
import open3d as o3d
import cv2
import numpy as np
import copy
import json
import os
import vtk
import csv
import os.path

#Asus camera
FX_RGB = 570.3422241210938
FY_RGB = 570.3422241210938
CX_RGB = 319.5
CY_RGB = 239.5

FX_DEPTH = 570.3422241210938
FY_DEPTH = 570.3422241210938
CX_DEPTH = 314.5
CY_DEPTH = 235.5
camera_depth=o3d.camera.PinholeCameraIntrinsic(640,480,FX_DEPTH,FX_DEPTH,CX_DEPTH,CY_DEPTH)

def preprocess_point_cloud(pcd, voxel_size): #point cloud, voxel (volumetric pixel)
    pcd_down = pcd.voxel_down_sample(voxel_size) #postavlja na tu dimenziju voxela (1cm postavljeno)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) #trazi parametre za hibridni knn i radijalnu potragu

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) #funkcija koja radi "fast point feature histogram" za oblak tocaka
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 10 #1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def draw_registration_result(source, target, transformation, tree, imageNumber, window_name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    return [source_temp, target_temp]

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 10 #0.1 #0.01
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
    return result

def save_recontructed_PCD(merged_pcd, treeData,full_path):
    name = treeData + "_merged.ply"
    o3d.io.write_point_cloud(full_path + "\\" + name, merged_pcd)
    print("Saved merged point cloud")


def rotX(theta):
    Rx = [[1, 0 , 0,0], [0 , np.cos(theta), -np.sin(theta),0], [0, np.sin(theta), np.cos(theta),0],[0, 0, 0, 1]]
    return Rx

def rotY(theta):
    Ry = [[np.cos(theta), 0 , np.sin(theta),0], [0 , 1, 0,0], [ -np.sin(theta),0, np.cos(theta),0],[0, 0, 0, 1]]
    return Ry

def rotZ(theta):
    Rz = [[np.cos(theta),-np.sin(theta) , 0,0], [np.sin(theta) , np.cos(theta), 0,0], [0, 0, 1,0], [0, 0, 0, 1]]
    return Rz

def save_which_PCD_was_saved(value, tree, imageNumber):
    fields = ['imageName', 'imageNumber', 'typeOfPCD']
    data = [tree, imageNumber,value]
    if(os.path.exists("popisOblakaKMeansRansac.csv")):
        print("wrote")
        with open("popisOblakaKMeansRansac.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        print("wrote_prvi put")
        with open("popisOblakaKMeansRansac.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )

def save_which_PCD_was_savedA(value, tree, imageNumber):
    fields = ['imageName', 'imageNumber', 'typeOfPCD']
    data = [tree, imageNumber,value]
    if(os.path.exists("popisOblakaKMeansRansac_After.csv")):
        print("wrote  after")
        with open("popisOblakaKMeansRansac_After.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        print("wrote_prvi put u After")
        with open("popisOblakaKMeansRansac_After.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )

def getStoredValueFromBefore(tree, imageTurn):
    treeNumber = tree.split("_")[-1]
    listOfUsedNumbers = []
    fileName = "popisOblakaKMeansRansac" + ".csv"
    dataFrame = pd.read_csv(fileName)
    if(After):
        return True
    try:
        methodUsedForRemovingGrass = dataFrame[(dataFrame.imageName == tree)  & (dataFrame.imageNumber == imageTurn)].typeOfPCD.values[0]
    except:
        return True
    return methodUsedForRemovingGrass

def getKoristeneSlike(tree,After):
    treeNumber = tree.split("_")[-1]
    print(treeNumber)
    listOfUsedNumbers = []
    if not After:
        fileName = "Indexes_Before_Pruning" + ".csv"
        dataFrame = pd.read_csv(fileName)
        strOfUsedNumbers = dataFrame.iloc[int(treeNumber),2]
    else:
        fileName = "Indexes_After_Pruning" + ".csv"
        dataFrame = pd.read_csv(fileName)
        strOfUsedNumbers = dataFrame.iloc[int(treeNumber),2]
    return strOfUsedNumbers


visualisationFlag = False
visualize_XYZ_coordinate=False
After = False
firstTimeRemovingGrass = False
starting_number = 0
imageNumber = 0
row_number = 1
sorta = "V"

csv_data = [["Tree", "BestRMSE_Spajanje 0 i 1", "BestRMSE_Spajanje 01 i 2", "BestRMSE_Spajanje 012 i 3", "Indeksi najboljih iteracija"]]

for i in range(starting_number,70): 
    imageNumber = 0
    if (i== 11 or i == 13):
        continue
    if (starting_number != 0):
        csv_data = []
    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + sorta + "_" + "000" + str(i)
    elif(i>=10 and i <= 99):
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "00" + str(i)
    else:
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "0" + str(i)

    lines = ["Working on tree: " + str(tree) + "\n"]
    #SST 
    full_path = "BRANCH/images/asus/B/E/" + tree + "/angle0"
    lst = os.listdir(full_path + "/color") 
    numberOfImagesOfOneTree = len(lst)
    all_PCD_One_Tree = []

    string_ofUsedImages = getKoristeneSlike(tree,After)
    print("Working on tree: " + str(tree))
    print("Pročitane točke su sljedeće: ", string_ofUsedImages)
    imageTurn = 0
    if(string_ofUsedImages == "-"):
        continue

    for im in range(0, numberOfImagesOfOneTree):
        if(im in np.array(string_ofUsedImages.split(","), dtype = int)):
            
            #print("Working on image: " + tree + " , number: "+ str(im))
            color_bp = o3d.io.read_image(full_path + "/color/" +str(im) + ".png")
            depth_bp = o3d.io.read_image(full_path+"/depth/" + str(im) + ".png")
 
            rgbd_bp = o3d.geometry.RGBDImage.create_from_color_and_depth(color_bp, depth_bp, convert_rgb_to_intensity=False)
            pcd_bp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_bp, camera_depth)
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35)

            if visualize_XYZ_coordinate:
                o3d.visualization.draw_geometries([pcd_bp, origin_frame], window_name = "Before rotating: "+ str(tree))

            pcd_bp.transform(rotZ(-3.14/2))# #Rz -90
            pcd_bp.transform(rotY(3.14))# #RY 180
            if visualize_XYZ_coordinate:
                o3d.visualization.draw_geometries([pcd_bp, origin_frame], window_name = "After rotating: "+ str(tree))
            if visualisationFlag:
                o3d.visualization.draw_geometries([pcd_bp], window_name = "ORIGINAL RGB-D: "+ str(tree))

            #KMeans algoritam
            tocke = np.asarray(pcd_bp.points)        
            km = KMeans(n_clusters=2, init='random', n_init=5, random_state=0)
            km.fit(tocke)
            labels = km.predict(tocke)
            centroids = km.cluster_centers_   
            x = np.array(labels==0)
            y = np.array(labels==1)

            if (firstTimeRemovingGrass):

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(centroids[:,0],centroids[:,1],c="black",label="Centers",alpha=1)
                ax.scatter(tocke[x,0],tocke[x,1],tocke[x,2],c="red") 
                ax.scatter(tocke[y,0],tocke[y,1],tocke[y,2],c="blue")
                plt.show()

            pcb_kmeans = o3d.geometry.PointCloud()
            tocke_colors = np.asarray(pcd_bp.colors)
            #ona klasa koja ima max visinu pripada drvecu a ne travi
            if(max(tocke[x][:,1]) > max(tocke[y][:,1])):
                pcb_kmeans.points = o3d.utility.Vector3dVector(tocke[x])
                pcb_kmeans.colors = o3d.utility.Vector3dVector(tocke_colors[x])
            else:
                pcb_kmeans.points = o3d.utility.Vector3dVector(tocke[y])
                pcb_kmeans.colors = o3d.utility.Vector3dVector(tocke_colors[y])

            if (firstTimeRemovingGrass):
                o3d.visualization.draw_geometries([pcb_kmeans, origin_frame], window_name = "KMeans: "+ str(tree))
    


            tocke = np.asarray(pcd_bp.points)
            tocke_colors = np.asarray(pcd_bp.colors)
            XX = tocke[:,0].reshape(-1, 1)
            YY = tocke[:,1].reshape(-1, 1)
            ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=0.005)
            ransac.fit(XX, YY)
            x = np.linspace(-5,5,len(YY))
            z = ransac.estimator_.intercept_ + ransac.estimator_.coef_[0]*x 
            z1 = ransac.estimator_.intercept_ + ransac.estimator_.coef_[0]*x  + 0.075
            z2 = ransac.estimator_.intercept_ + ransac.estimator_.coef_[0]*x  - 0.075

            if (firstTimeRemovingGrass):
                plt.figure()
                plt.plot(XX,YY, '.')
                plt.plot(x, z, '+b')
                plt.plot(x,z1, '+r')
                plt.plot(x,z2, '+g')
                plt.show()

            new_tocke1 = []
            new_tockecolors = []
            for j in range(0, len(tocke)):
                if (abs(ransac.estimator_.intercept_ + ransac.estimator_.coef_[0]*tocke[j][0]-tocke[j][1]))  > 0.075 : #ako su udaljeni od pravca kroz zemlju za 0.075 ili ako su jako daleko od pravca
                    if(tocke[j][2] > -2 ): #ako su dalje od -2 u dubini
                        new_tocke1.append(tocke[j])
                        new_tockecolors.append(tocke_colors[j])

            #print("Nove tocke1: ", len(new_tocke1))       
            pcd_ransac = o3d.geometry.PointCloud()
            pcd_ransac.points = o3d.utility.Vector3dVector(new_tocke1)
            pcd_ransac.colors = o3d.utility.Vector3dVector(new_tockecolors)
            if (firstTimeRemovingGrass):
                o3d.visualization.draw_geometries([pcd_ransac, origin_frame], window_name = "Ransac: "+ str(tree))
            
            #Čitanje prethodno odabranog micanaj trave iz csv popisa
            value = True
            if not After:
                value = getStoredValueFromBefore(tree, imageTurn)
            imageTurn = imageTurn +1


            if(value == True):
                 inputFlag = True
                 #Ručni odabir micanja trave
                 #inputFlag = True
                 value = input("Odaberi koji oblak zelis sacuvati (o-original, k-kmeans, r-ransac): ")
            else:
                 inputFlag = False

            if(value == "o"):
                all_PCD_One_Tree.append(pcd_bp)
                #print("SAVED original")
            elif(value == "k"):
                all_PCD_One_Tree.append(pcb_kmeans)
                #print("SAVED kmeans")
            elif(value == "r"):
                all_PCD_One_Tree.append(pcd_ransac)
                #print("SAVED ransac")
            if(inputFlag): 
                if(After):
                         #AFTER
                    save_which_PCD_was_savedA(value, tree, imageTurn) #j je imagenumber (slijedni broj slike)
                else: 
                        #BEFORE
                    save_which_PCD_was_saved(value, tree, imageTurn) #j je imagenumber (slijedni broj slike)
           


    #rekonstrukcija 3D modela drveta na temelju snimljenih 4 slika
    #target je fiksni, source se rotira
    target = all_PCD_One_Tree[0]
    best_rmse = 999.9
    best_z_array = []
    csv_row = [str(tree)]
    for k in range(0, 4-1):
        print("Spajanje: " + str(k))
        lines.extend("\t Spajanje: " + str(k) + "\n")
        voxel_size = 0.01 #1cm

        source = all_PCD_One_Tree[k+1]
        source_colors = np.asarray(source.colors) 
        target_colors = np.asarray(target.colors) 

        threshold = 0.2
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

        if visualisationFlag:
            o3d.visualization.draw_geometries([source_down,target_down], "Downsampled preprocessed point cloud")
        z = 0
        best_z = 0
        counter_problem4 = 0
        best_correspondence = 0
        best_source_temp_color = np.asarray(source.colors) 
        best_fake = 0
        #Isprobavanje pronalaska rješenja u 5 koraka
        while(z < 5):
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size)
            #print(result_ransac)

            if visualisationFlag:
                draw_registration_result(source_down, target_down, result_ransac.transformation, tree, imageNumber, "Ransac transformation number: "+ str(k))

            #print("Initial alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, result_ransac.transformation)
            #print(evaluation)
        
            result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
            #print("----Result after ICP, pokusaj: ", z)
            #print(result_icp)
            change = False
            problem3 = False 
            [source_temp, target_temp]= draw_registration_result(source, target, result_icp.transformation, tree, imageNumber, "ICP transformation" + str(k))
            if len(result_icp.correspondence_set) > best_correspondence:
                change = True 
                if((z > 0) & (result_icp.inlier_rmse > best_fake*1.2)):
                    print("Problem 3")
                    problem3 = True
                else:
                    best_z = z
                    best_source_temp = source_temp
                    best_source_temp_color = np.asarray(source.colors)
                    best_correspondence = len(result_icp.correspondence_set)
                    if(k <1 ): best_rmse = result_icp.inlier_rmse #zapamti najbolji rmse za preklop prve 2 slike  (to je referenca)
                    best_fake = result_icp.inlier_rmse
                    print("\tNajbolja iteracija je: ", str(z))
                    #print("- --- --- --- --- --- --- --- -- --- --- -- --- -- --- -- -- -- -- --- -- -- --- --- -- --- -- --- --- -- --- --- -- --- --- -- --- -- --- --- -- -")
            if change:
                if problem3:
                    lines.extend("\t\t Iteracija: " + str(z) + "\n \t\t\t Nastao je problem 3\n")
                else:
                    lines.extend("\t\t Iteracija: " + str(z) + "\n \t\t\t Fitness: " + str(result_icp.fitness) + "\t Inlier_RMSE: " + str(best_fake) + "\t Correspondence set: " + str(best_correspondence) + "\n")
            else:
                lines.extend("\t\t Iteracija: " + str(z) + "\n \t\t\t Nema promjene\n")
            
            z = z+1
            #za spajanje skupa od 3 slike s 4-tom slikom aktivira ovaj slučaj (jer prije svakako nađe najbolje rj),
            #ideja je da se vrti dokle god ne nadje 50% losiji rezultat od najboljeg pronadjenog rezultata za spajanje
            #prve dvije slike jer tu pronalazi najbolji rmse od svih spajanja (plus rmse se povecava sa svakom novom dodanom slikom zato i mora biti raspon od 50% )
            if(z > 4 and best_rmse*1.5 < best_fake):
                counter_problem4 = counter_problem4 + 1
                lines.extend("\t\t\t Nastao je problem 4 (broj pokušaja: " + str(counter_problem4) + ")" + "\n")
                z = z-1
                print(" ! ! ! ! ! ! Ponavljam jer nije nasao i dalje top podudaranje")
            if(counter_problem4 >= 5):
                lines.extend("\t\t\tPreviše ponavljanja zbog problema 4\n")
                print("Previše pokušaja, preskačem")
                z=6

            
        best_source_temp.colors =  o3d.utility.Vector3dVector(best_source_temp_color)
        target_temp.colors = o3d.utility.Vector3dVector(target_colors)
        target = target_temp + best_source_temp
        #o3d.visualization.draw_geometries([target, origin_frame], window_name = "PCD DONE FOR: "+ str(tree))
        best_z_array.append(best_z)
        csv_row.append(best_fake)

    with open("resultsTracker.txt", "a") as file:
        file.writelines(lines)
        file.writelines("\n\tZa drvo " + str(tree) + " najbolja preklapanja su bila u iteracijama: " + str(best_z_array) + "\n\n")
    csv_row.append(best_z_array)
    csv_data.append(csv_row)

    # Path to the CSV file
    csv_file_path = "resultsTracker.csv"

    # Write data to the CSV file
    if starting_number == 0 :
        csv_writer_option = "w"
    else:
        csv_writer_option = "a"

    with open(csv_file_path, csv_writer_option, newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

        
    #save source
    # shouldIsave = input("If the created model is fine, press y")
    # if(shouldIsave == "y"):
    #     save_recontructed_PCD(target, tree, full_path)
    #     #read saved ply 
    #     print("SAVED PCD JANA TUKO")
    #     print(full_path + "/" + tree + "merged.ply")
    #     pcd = o3d.io.read_point_cloud(full_path + "/" + tree + "_merged.ply")
    #     o3d.visualization.draw_geometries([pcd], window_name = "-----OVO sam spremila: ")

