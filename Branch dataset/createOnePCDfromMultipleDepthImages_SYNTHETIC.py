import open3d as o3d
import numpy as np
import copy
import os
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
    distance_threshold = voxel_size * 10
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
    if visualisationFlag:
        o3d.visualization.draw_geometries([source_temp, target_temp], window_name = window_name)
    return [source_temp, target_temp]

#Refine registration
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 10
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, 
                                                         result_ransac.transformation, 
                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
    return result

def save_recontructed_PCD(merged_pcd, treeData,full_path):
    name = treeData + "_merged.ply"
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

def save_witch_PCD_was_saved(value, tree, imageNumber):
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


visualisationFlag = False
starting_number =1
imageNumber = 0
row_number = 1
sorta = "V"


for i in range(starting_number,2): #50
    imageNumber = 0

    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + sorta + "_" + "000" + str(i)
    else:
        tree="tree00" + str(row_number)+ "_" + sorta + "_" + "000" + str(i)

    full_path = "/home/andrej/anaconda3/envs/3bot/SyntheticImages/"

    numberOfImagesOfOneTree = 3
    print(numberOfImagesOfOneTree)
    all_PCD_One_Tree = []
    for im in range(0, numberOfImagesOfOneTree):
        print(full_path + str(im) + ".png")
        print(full_path + "ScreenImage" + str(im) + ".png")
        color_bp = o3d.io.read_image(full_path + "ScreenImage" + str(im) + ".png")
        depth_bp = o3d.io.read_image(full_path + str(im) + ".png")
        if visualisationFlag:
            o3d.visualization.draw_geometries([color_bp], window_name = "RGB -origigi: "+ str(im))
            o3d.visualization.draw_geometries([depth_bp], window_name = "dubinska -origigi: "+ str(im))
                
        rgbd_bp = o3d.geometry.RGBDImage.create_from_color_and_depth(color_bp, depth_bp, convert_rgb_to_intensity=False)
        pcd_bp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_bp, camera_depth)
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        #visualization
        if visualisationFlag:
            o3d.visualization.draw_geometries([pcd_bp], window_name = "RGB-D: "+ str(im))

        pcd_bp.transform(rotZ(-3.14))# #Rz -180
        #pcd_bp.transform(rotY(3.14))# #RY 180
        if visualisationFlag:
            o3d.visualization.draw_geometries([pcd_bp], window_name = "ORIGINAL RGB-D (after rotation): "+ str(tree))
        
        value = "o"
        all_PCD_One_Tree.append(pcd_bp)
        save_witch_PCD_was_saved(value, tree, im) #j je imagenumber (slijedni broj slike)

        #all_PCD_One_Tree.append(pcd_bp)


    
    #rekonstrukcija 3D modela drveta na temelju snimljenih n slika
    #target je fiksni, source se rotira
    target = all_PCD_One_Tree[0]

    best_rmse = 999.9
    for k in range(0, numberOfImagesOfOneTree-1):
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
            print("----Result after ICP, pokusaj: ", z)
            print(result_icp)    
            [source_temp, target_temp]= draw_registration_result(source, target, result_icp.transformation, tree, imageNumber, "ICP transformation" + str(k))
            #o3d.visualization.draw_geometries([all_PCD_One_Tree[k]], window_name = "Colored image before pruning")
            #o3d.visualization.draw_geometries([all_PCD_One_Tree[k+1]], window_name = "Colored image after pruning")
            if len(result_icp.correspondence_set) > best_correspondence: 
                best_source_temp = source_temp
                best_correspondence = len(result_icp.correspondence_set)
                if(k <=1 ): best_rmse = result_icp.inlier_rmse #zapamti najbolji rmse za preklop 2 slike 
                best_fake = result_icp.inlier_rmse
                print("so far najbolji je: ", str(z))
                print("- --- --- --- --- --- --- --- -- --- --- -- --- -- --- -- -- -- -- --- -- -- --- --- -- --- -- --- --- -- --- --- -- --- --- -- --- -- --- --- -- -")
            
            z = z+1
            #za spajanje skupa od 3 slike s 4-tom slikom aktivira ovaj slučaj (jer prije svakako nadje najbolje rj),
            #ideja je da se vrti dokle god ne nadje 50% losiji rezultat od najboljeg pronadjenog rezultata za spajanje
            #prve dvije slike jer tu pronalazi najbolji rmse od svih spajanja (plus rmse se povecava sa svakom novom dodanom slikom zato i mora biti raspon od 50% )
            if(z > 5 and best_rmse*1.5 < best_fake):
                z = z-1
                print(" ! ! ! ! ! ! Ponavljam jer nije nasao i dalje top podudaranje")
        
            
        target = target_temp + best_source_temp
        o3d.visualization.draw_geometries([target, origin_frame], window_name = "PCD DONE FOR: "+ str(tree))

    #save pcd
    #target.colors = o3d.utility.Vector3dVector(source_colors)
    #save_recontructed_PCD(target, tree, full_path)
    #read saved ply 
    # pcd = o3d.io.read_point_cloud("tree_1_V_0000merged.ply")
    # o3d.visualization.draw_geometries([pcd], window_name = "Read pcb: ")