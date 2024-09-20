#Preklop PCD-a dvije slike -> treba prosirit
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

def display_inlier_outlier(cloud, ind, window_name):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    #print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name = window_name)

def preprocess_point_cloud(pcd, voxel_size): #point cloud, voxel (volumetric pixel)
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size) #postavlja na tu dimenziju voxela (1cm postavljeno)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) #trazi parametre za hibridni knn i radijalnu potragu

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) #funkcija koja radi "fast point feature histogram" za oblak tocaka
    return pcd_down, pcd_fpfh

#Global registration
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 10 #1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
def draw_registration_result(source, target, transformation, tree, window_name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    #o3d.visualization.draw_geometries([source_temp, target_temp], window_name = window_name)
    return [source_temp, target_temp]
#Refine registration
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 10 #0.1 #0.01
    #print(":: Point-to-point ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    source.estimate_normals() #needed for TransformationEstimationPointToPlane
    target.estimate_normals() #needed for TransformationEstimationPointToPlane
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
    return result

def save_recontructed_PCD(merged_pcd, treeData,full_path):
    name = treeData + "_mergedBA.ply"
    o3d.io.write_point_cloud(full_path + "/" + name, merged_pcd)
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


visualisationFlag = False
After = True
firstTimeRemovingGrass = True
starting_number = 31
row_number = 1
sorta = "V"


for i in range(starting_number,70): 

    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + sorta + "_" + "000" + str(i)
    elif(i>=10 and i <= 99):
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "00" + str(i)
    else:
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "0" + str(i)

    #SST 
    full_path_Before = "BRANCH/images/asus/B/E/" + tree + "/angle0"
    full_path_After = "BRANCH/images/asus/A/E/" + tree + "/angle0"
    full_path_merged = "BRANCH/images/asus/Merged"

    if not os.path.isfile(full_path_Before + "/" + tree + "_merged.ply"):
        print("No merged file found for tree: " + str(tree))
        continue
    if not os.path.isfile(full_path_After + "/" + tree + "_merged.ply"):
        print("No merged file found for tree: " + str(tree))
        continue

    print("Working on image: " + tree)    
    pcd_before = o3d.io.read_point_cloud(full_path_Before + "/" + tree + "_merged.ply")
    o3d.visualization.draw_geometries([pcd_before], window_name = "ORIGINAL RGB-D: "+ str(tree))

    pcd_after = o3d.io.read_point_cloud(full_path_After + "/" + tree + "_merged.ply")
    o3d.visualization.draw_geometries([pcd_after], window_name = "ORIGINAL RGB-D: "+ str(tree))

    #target je fiksni, source se rotira
    target = pcd_before
    best_rmse = 999.9
    voxel_size = 0.01 #1cm

    source = pcd_after
    source_colors = np.asarray(source.colors) 
    target_colors = np.asarray(target.colors) 


    threshold = 0.2
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    if visualisationFlag:
        o3d.visualization.draw_geometries([source_down,target_down], "Downsampled preprocessed point cloud")
    z = 0
    best_correspondence = 0
    best_source_temp_color = np.asarray(source.colors) 
    #samo ponavljanje da vidim jel moze u 5 pokusaja doci do nekog supac rj
    while(z < 5):
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size)
            #print(result_ransac)

            if visualisationFlag:
                draw_registration_result(source_down, target_down, result_ransac.transformation, tree, "Ransac transformation number: ")

            #print("Initial alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, result_ransac.transformation)
            #print(evaluation)
        
            result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
            print("----Result after ICP, pokusaj: ", z)
            print(result_icp)    
            [source_temp, target_temp]= draw_registration_result(source, target, result_icp.transformation, tree, "ICP transformation" )
            #o3d.visualization.draw_geometries([all_PCD_One_Tree[k]], window_name = "Colored image before pruning")
            #o3d.visualization.draw_geometries([all_PCD_One_Tree[k+1]], window_name = "Colored image after pruning")
            if len(result_icp.correspondence_set) > best_correspondence: 
                if((z > 0) & (result_icp.inlier_rmse > best_rmse*1.2)):
                    print("OVO JE SAD TAJ PROBLEM 3 GDJE IMA VIŠE TOČAKA ALI MANJU PODUDARNOST")
                else:
                    best_source_temp = source_temp
                    best_source_temp_color = np.asarray(source.colors)
                    best_correspondence = len(result_icp.correspondence_set)
                    if( best_rmse > result_icp.inlier_rmse ):#zapamti najbolji rmse za sve 
                        best_rmse = result_icp.inlier_rmse
                    print("so far najbolji je: ", str(z))
                    print("- --- --- --- --- --- --- --- -- --- --- -- --- -- --- -- -- -- -- --- -- -- --- --- -- --- -- --- --- -- --- --- -- --- --- -- --- -- --- --- -- -")
            

            
            z = z+1
            #za spajanje skupa od 3 slike s 4-tom slikom aktivira ovaj slučaj (jer prije svakako nadje najbolje rj),
            #ideja je da se vrti dokle god ne nadje 50% losiji rezultat od najboljeg pronadjenog rezultata za spajanje
            #prve dvije slike jer tu pronalazi najbolji rmse od svih spajanja (plus rmse se povecava sa svakom novom dodanom slikom zato i mora biti raspon od 50% )


            
    best_source_temp.colors =  o3d.utility.Vector3dVector(best_source_temp_color)
    target_temp.colors = o3d.utility.Vector3dVector(target_colors)
    target = target_temp + best_source_temp
    o3d.visualization.draw_geometries([target], window_name = "PCD DONE FOR: "+ str(tree))

    #DOWNSAMPLE = 0.5
    target_temp_vox = target_temp.voxel_down_sample(voxel_size=0.01) #0.01
    best_source_temp_vox = best_source_temp.voxel_down_sample(voxel_size=0.01) #0.01

    best_source_temp_vox.paint_uniform_color([1, 0, 0])
    target_temp_vox.paint_uniform_color([0, 0, 1])

    source_tocke = np.asarray(best_source_temp_vox.points)
    target_tocke = np.asarray(target_temp_vox.points)



    #udaljenost source točke od target točke je zadan neki treshold u iznosu od 10 cm
    #d = √[(x₂ - x₁)² + (y₂ - y₁)² + (z₂ - z₁)²] #racunanje udaljenosti u 3D prostoru

    arrayOfMatchedPoints = np.empty(len(target_tocke), dtype=np.bool_)
    arrayOfMatchedPoints.fill(False) #deblo je True, grane su False
    treshold_value = 0.08 #smanjiti još 
    print("Source: ", len(best_source_temp.points))
    print("Target: ",len(target_temp.points))
    print("Source(downsampled): ", len(source_tocke))
    print("Target(downsampled): ",len(target_tocke))
    #za svaku tocku u sourc-u, ako bilo koja tocka iz targeta pripada tom krugu, onda se stavlja na True
    #inace ostaje False = a to ce onda biti upravo tocke koje su grane orezane.

    distane_threshold = 0.03
    while (distane_threshold != "stop"):
        dists = target_temp.compute_point_cloud_distance(best_source_temp)
        dists = np.asarray(dists)
        ind = np.where(dists > float(distane_threshold))[0]

        best_source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])

        branches_colors = np.array(target_temp.colors)
        branches_colors[ind] = [1.0, 1.0, 0.0] #yellow

        target_temp.colors = o3d.utility.Vector3dVector(branches_colors)
        target_2 = target_temp + best_source_temp
        print("Prikaz grana žutom bojom")
        o3d.visualization.draw_geometries([target_2], window_name = "Stablo: "+ str(tree))
        distane_threshold = input("Unesi željeni threshold:")

    #save source
    shouldIsave = input("If the created model is fine, press y")
    if(shouldIsave == "y"):
        save_recontructed_PCD(target_2, tree, full_path_merged)
        #read saved ply 
        print("SAVED  MERGED PCD")
        pcd = o3d.io.read_point_cloud(full_path_merged + "/" + tree + "_mergedBA.ply")
        o3d.visualization.draw_geometries([pcd], window_name = "Spremljeni model")