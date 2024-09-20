import open3d as o3d
import numpy as np
import copy
import os.path
from scipy.spatial import KDTree

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

def draw_registration_result(source, target, transformation, window_name):
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


#https://medium.com/@sim30217/chamfer-distance-4207955e8612
def chamfer_distance(A, B, filter=False, threshold=0.5):
    #Computes the chamfer distance between two sets of points A and B.
    #If filter is True, filter points that are further than set threshold distance
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    if filter:
        dist_A = dist_A[dist_A <= threshold]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    if filter:
        dist_B = dist_B[dist_B <= threshold]
    return np.mean(dist_A) + np.mean(dist_B)

visualisationFlag = False
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
source = o3d.io.read_point_cloud("tree_synthetic_merged.ply")
target = o3d.io.read_point_cloud("Mesh_equalPoints_4pictures.ply")
numOfSourcePoints = len(source.points)
numOfTargetPoints = len(target.points)
best_rmse = 999.9

voxel_size = 0.01 #1cm

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

    if visualisationFlag:
        draw_registration_result(source_down, target_down, result_ransac.transformation, "Ransac transformation" )

    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
    print("----Result after ICP, pokusaj: ", z)
    print(result_icp)    
    [source_temp, target_temp]= draw_registration_result(source, target, result_icp.transformation, "ICP transformation")
    if len(result_icp.correspondence_set) > best_correspondence: 
        best_source_temp = source_temp
        best_correspondence = len(result_icp.correspondence_set)
        best_rmse = result_icp.inlier_rmse #zapamti najbolji rmse za preklop 2 slike 
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

o3d.visualization.draw_geometries([target_temp, origin_frame], window_name = "PCD DONE")
o3d.visualization.draw_geometries([best_source_temp, origin_frame], window_name = "PCD DONE")     
alligned = target_temp + best_source_temp
o3d.visualization.draw_geometries([alligned, origin_frame], window_name = "PCD DONE")

alligned.colors = o3d.utility.Vector3dVector(source_colors)

best_source_temp.estimate_normals()

#Calculate distance between each point from reconstructed PCD and original mesh
#Which is the same as making a sphere around each point of original mesh, and selecting points which are contained inside
#Number of those points is the intersection
dists = best_source_temp.compute_point_cloud_distance(target_temp)
dists = np.asarray(dists)
threshold_1cm = 0.01 #1 cm threshold
threshold_5cm = 0.05 #5 cm threshold
ind_1cm = np.where(dists > threshold_1cm)[0]
ind_5cm = np.where(dists > threshold_5cm)[0]
#Select by index (remove all that satistfy the condition above)
outliers_1cm = best_source_temp.select_by_index(ind_1cm)
outliers_5cm = best_source_temp.select_by_index(ind_5cm)
correspondencePoints_1cm = best_source_temp.select_by_index(ind_1cm, invert=True)
correspondencePoints_5cm = best_source_temp.select_by_index(ind_5cm, invert=True)
numOfCorrespondencePoints_1cm = numOfSourcePoints - len(outliers_1cm.points)
numOfCorrespondencePoints_5cm = numOfSourcePoints - len(outliers_5cm.points)
numOfUnionPoints_1cm = numOfSourcePoints + numOfTargetPoints - numOfCorrespondencePoints_1cm
numOfUnionPoints_5cm = numOfSourcePoints + numOfTargetPoints - numOfCorrespondencePoints_5cm
print("Number of points in source (reconstruced PCD): " + str(numOfSourcePoints))
print("Number of points in target (synthetic model): " + str(numOfTargetPoints))
print("Number of correspondence points for threshold 1cm (intersection): " + str(numOfCorrespondencePoints_1cm))
print("Number of correspondence points for threshold 5cm (intersection): " + str(numOfCorrespondencePoints_5cm))
print("Number of outliers for threshold 1cm: " + str(len(outliers_1cm.points)))
print("Number of outliers for threshold 5cm: " + str(len(outliers_5cm.points)))
print("Number of points of both point clouds for threshold 1cm (union): " + str(numOfUnionPoints_1cm))
print("Number of points of both point clouds for threshold 5cm (union): " + str(numOfUnionPoints_5cm))
print("Intersection over union for threshold 1cm: " + str(numOfCorrespondencePoints_1cm / (numOfUnionPoints_1cm)))
print("Intersection over union for threshold 5cm: " + str(numOfCorrespondencePoints_5cm / (numOfUnionPoints_5cm)))
chamferDistance_unfiltered = chamfer_distance(best_source_temp.points, target_temp.points)
#Filter points that are further than 50cm away from the tree. Those points are considered points from other trees
#in order to calculate the mean distance of points that are considered part of the observed tree
#This distance shows the average error of reconstruction program
chamferDistance_filtered = chamfer_distance(best_source_temp.points, target_temp.points, True, 0.5)
print("Unfiltered chamfer distance between reconstructed PCD and syntethic model: " + str(chamferDistance_unfiltered))
print("Filtered chamfer distance between reconstructed PCD and syntethic model: " + str(chamferDistance_filtered))

outliers_1cm.paint_uniform_color([1, 0, 0])
correspondencePoints_1cm.paint_uniform_color([0, 1, 0])
target.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([outliers_1cm, correspondencePoints_1cm, target], 
                                  window_name = "Showing PCD (threshold 1cm): outliers - red, correspondencePoints - green, synthetic model - blue")
outliers_5cm.paint_uniform_color([1, 0, 0])
correspondencePoints_5cm.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([outliers_5cm, correspondencePoints_5cm, target], 
                                  window_name = "Showing PCD (threshold 5cm): outliers - red, correspondencePoints - green, synthetic model - blue")