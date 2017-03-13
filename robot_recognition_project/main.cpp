using namespace std;

#include <pcl/apps/render_views_tesselated_sphere.h>
#include <vtkGenericDataObjectReader.h>

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/keypoints/uniform_sampling.h>

#include <pcl/registration/correspondence_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/sample_consensus_prerejective.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>

#include <pcl/recognition/cg/geometric_consistency.h>

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::VFHSignature308 GlobalDescriptorType;
typedef pcl::PointCloud<PointType>::Ptr CloudPtr;

typedef pcl::PointNormal PointNT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::PointCloud<PointNT> NormalPointCloud;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
typedef pcl::SHOT352 LocalDescriptorType;

typedef pcl::ReferenceFrame RFType;
typedef std::tuple<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >, std::vector<pcl::Correspondences>> ClusterType;


string model_filename_ = "stop.pcd";
string scene_filename_ = "robot_with_scene.pcd";

float leaf = 0.02f;

pcl::PCDWriter writer;

//Input simulation from 3D sensor
void
simulate_imput()
{
    // Get all data from the file
    vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    vtkSmartPointer<vtkPolyData> model;
    reader->SetFileName("stopy_cut.vtk");
    reader->Update();

    if(reader->IsFilePolyData())
    {
        std::cout << "output is a polydata" << std::endl;
        model = reader->GetPolyDataOutput();
        std::cout << "output has " << model->GetNumberOfPoints() << " points." << std::endl;
    }

    pcl::apps::RenderViewsTesselatedSphere render_views;
    render_views.addModelFromPolyData(model);
    render_views.setResolution(150);
    render_views.setTesselationLevel(1);
    render_views.setGenOrganized(false);
    render_views.setUseVertices(false); // remove
    render_views.setComputeEntropies(false); // remove
    //render_views.generateViews();

    /*std::vector<CloudPtr> views;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
    render_views.getViews(views);
    render_views.getPoses(poses);

    int i = 0;

    for (i; i < views.size(); i++)
    {
        string iterator = "extracted_pose_" + i;
        writer.write(iterator, *views[i]);
    }*/
}

std::vector<pcl::PointCloud<PointType>::Ptr>
scene_segmentation (pcl::PointCloud<PointType>::Ptr scene)
{
    std::vector<pcl::PointCloud<PointType>::Ptr> clustered_objects_list;
    std::vector<pcl::PointIndices> cluster_indices;

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (scene);

    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (25000);
    ec.setMaxClusterSize (30000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (scene);
    ec.extract (cluster_indices);

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(scene->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        j++;

        //Write clusters to the file
        //std::stringstream ss;
        //ss << "cloud_cluster_" << j << ".pcd";
        //writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false);

        clustered_objects_list.push_back(cloud_cluster);
    }

    return clustered_objects_list;
}

void
filter_data (pcl::PointCloud<PointType>::Ptr object, bool passThrough = false)
{
    if (passThrough)
    {
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud (object);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0, 0.8);
        pass.filter (*object);
    }

    pcl::StatisticalOutlierRemoval<PointType> sor_ro;
    sor_ro.setInputCloud (object);
    sor_ro.setMeanK (7);
    sor_ro.setStddevMulThresh (0.2);
    sor_ro.filter (*object);

    pcl::RadiusOutlierRemoval<PointType> outrem;
    outrem.setInputCloud(object);
    outrem.setRadiusSearch(0.02);
    outrem.setMinNeighborsInRadius (7);
    outrem.filter (*object);

    pcl::VoxelGrid<PointType> vg;
    vg.setInputCloud (object);
    vg.setLeafSize (leaf, leaf, leaf);
    vg.filter (*object);
}

void
compute_normals(pcl::PointCloud<PointType>::Ptr object, pcl::PointCloud<NormalType>::Ptr object_normals)
{
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    pcl::search::KdTree<PointType>::Ptr tree_n(new pcl::search::KdTree<PointType>());
    norm_est.setKSearch (5);
    norm_est.setSearchMethod(tree_n);
    //norm_est.setRadiusSearch(0.02);
    norm_est.setInputCloud (object);
    norm_est.compute (*object_normals);
}

void
compute_global_feature(pcl::PointCloud<PointType>::Ptr object,
                        pcl::PointCloud<PointType>::Ptr scene,
                        pcl::PointCloud<NormalType>::Ptr object_normals,
                        pcl::PointCloud<NormalType>::Ptr scene_normals)
{
    pcl::search::KdTree<PointType>::Ptr object_tree(new pcl::search::KdTree<PointType>);
    pcl::search::KdTree<PointType>::Ptr scene_tree(new pcl::search::KdTree<PointType>);
    pcl::PointCloud<GlobalDescriptorType>::Ptr object_global_descriptor(new pcl::PointCloud<GlobalDescriptorType>());
    pcl::PointCloud<GlobalDescriptorType>::Ptr scene_global_descriptor(new pcl::PointCloud<GlobalDescriptorType>());

    // OUR-CVFH feature estimation
    pcl::OURCVFHEstimation<PointType, NormalType, GlobalDescriptorType> ourcvfh_estimation;
    ourcvfh_estimation.setSearchMethod(object_tree);
    ourcvfh_estimation.setInputCloud(object);
    ourcvfh_estimation.setInputNormals(object_normals);
    ourcvfh_estimation.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees
    ourcvfh_estimation.setCurvatureThreshold(1.0);
    ourcvfh_estimation.setNormalizeBins(false);
    ourcvfh_estimation.setAxisRatio(0.8);
    ourcvfh_estimation.compute(*object_global_descriptor);
    //pcl::io::savePCDFileASCII("object_global_descriptor_ocvfh.pcd", *object_global_descriptor);

    ourcvfh_estimation.setSearchMethod(scene_tree);
    ourcvfh_estimation.setInputCloud(scene);
    ourcvfh_estimation.setInputNormals(scene_normals);
    ourcvfh_estimation.compute(*scene_global_descriptor);
    pcl::io::savePCDFileASCII("scene_global_descriptor_ocvfh.pcd", *scene_global_descriptor);

    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

    pcl::registration::CorrespondenceEstimation<GlobalDescriptorType, GlobalDescriptorType> est;
    est.setInputSource(scene_global_descriptor);
    est.setInputTarget(object_global_descriptor);
    est.determineReciprocalCorrespondences(*model_scene_corrs);

    std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;
}

void
compute_local_feautres(pcl::PointCloud<PointType>::Ptr object,
                       pcl::PointCloud<PointType>::Ptr scene,
                       pcl::PointCloud<NormalType>::Ptr object_normals,
                       pcl::PointCloud<NormalType>::Ptr scene_normals)
{
    // It is possible compute key points and extract their indexes:
    // pcl::UniformSampling<PointType> uniform_sampling; or pcl::ISSKeypoint3D<PointType, PointType> iss_detector;
    // example:
    // iss_detector.setSalientRadius(0.02);
    // iss_detector.setNonMaxRadius(0.03);
    // iss_detector.setInputCloud(object);
    // iss_detector.compute(*object_keypoints);
    // pcl::PointIndicesConstPtr indices = iss_detector.getKeypointsIndices();

    pcl::PointCloud<LocalDescriptorType>::Ptr object_descriptors (new pcl::PointCloud<LocalDescriptorType> ());
    pcl::PointCloud<LocalDescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<LocalDescriptorType> ());
    pcl::PointCloud<PointType>::Ptr object_keypoints (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());

    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::SHOTEstimationOMP<PointType, NormalType, pcl::SHOT352> descr_est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (object);
    uniform_sampling.setRadiusSearch (0.01f); // 3D grid leaf size
    pcl::PointCloud<int> keypoint_indices;
    uniform_sampling.compute(keypoint_indices);
    pcl::copyPointCloud(*object, keypoint_indices.points, *object_keypoints); // create pointcloud from indeces
    std::cout << "Model total points: " << object->size () << "; Selected Keypoints: " << object_keypoints->size () << std::endl;

    uniform_sampling.setInputCloud (scene);
    uniform_sampling.compute(keypoint_indices);
    pcl::copyPointCloud(*scene, keypoint_indices.points, *scene_keypoints); // create pointcloud from indeces
    std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

    descr_est.setInputCloud(scene_keypoints);
    descr_est.setSearchSurface(scene);
    descr_est.setInputNormals(scene_normals);
    descr_est.setSearchMethod(tree);
    descr_est.setRadiusSearch(0.06);
    descr_est.compute(*scene_descriptors);

    descr_est.setInputCloud(object_keypoints);
    descr_est.setSearchSurface(object),
    descr_est.setInputNormals(object_normals);
    descr_est.setSearchMethod(tree2);
    descr_est.compute(*object_descriptors);

    // Look for correspondances:
    pcl::KdTreeFLANN<pcl::SHOT352> match_search;
    match_search.setInputCloud (object_descriptors);

    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        if (match_search.point_representation_->isValid (scene_descriptors->at (i)))
        {
            int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
            if (found_neighs == 1 && neigh_sqr_dists[0] < 0.2)
            {
                pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);

                model_scene_corrs->push_back (corr);
            }
        }
    }

    std::cout << "\tFound " << model_scene_corrs->size () << " correspondences " << std::endl;

    ClusterType cluster_;

    pcl::PointCloud<RFType>::Ptr model_rf_ (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf_ (new pcl::PointCloud<RFType> ());
    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est_;
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer_;

    // Geometric consistency
    //gc_clusterer_.setGCSize (0.03f);
    //gc_clusterer_.setGCThreshold (10.0f);
    gc_clusterer_.setGCSize (0.025f);
    gc_clusterer_.setGCThreshold (15.0f);
    gc_clusterer_.setInputCloud (object_keypoints);
    gc_clusterer_.setSceneCloud (scene_keypoints);
    gc_clusterer_.setModelSceneCorrespondences (model_scene_corrs);

    gc_clusterer_.recognize (std::get < 0 > (cluster_), std::get < 1 > (cluster_));

    std::cout << "\tFound " << std::get < 0 > (cluster_).size () << " model instance/instances " << std::endl;

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    gc_clusterer_.recognize (rototranslations, clustered_corrs);
    std::cout << "Translations instances found: " << rototranslations.size () << std::endl;

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
    }

    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");

    viewer.addPointCloud (scene, "scene_cloud");
    pcl::PointCloud<PointType>::Ptr off_scene(new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*object, *off_scene, Eigen::Vector3f (-2,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> model_keypoints_color_handler (object_keypoints, 0, 0, 255);
    pcl::PointCloud<PointType>::Ptr off_scene_keypoints(new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*object_keypoints, *off_scene_keypoints, Eigen::Vector3f (-2,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    viewer.addPointCloud (off_scene_keypoints, model_keypoints_color_handler, "model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model_keypoints");


    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene, 255, 255, 128);
    viewer.addPointCloud (off_scene, off_scene_model_color_handler, "off_scene_model");
    viewer.addCoordinateSystem(1.0);

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*object, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
        viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

        for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
        {
            std::stringstream ss_line;
            ss_line << "correspondence_line" << i << "_" << j;
            PointType& model_point = off_scene_keypoints->at (clustered_corrs[i][j].index_query);
            PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
        }
    }

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }


}

void
align_model_to_scene(pcl::PointCloud<PointType>::Ptr object,
                     pcl::PointCloud<PointType>::Ptr scene)
{
    FeatureCloudT::Ptr object_features (new FeatureCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);
    NormalPointCloud::Ptr object_aligned (new NormalPointCloud);
    NormalPointCloud::Ptr object_normal(new NormalPointCloud);
    NormalPointCloud::Ptr scene_normal(new NormalPointCloud);

    pcl::copyPointCloud(*object, *object_normal);
    pcl::copyPointCloud(*scene, *scene_normal);

    pcl::NormalEstimationOMP<PointNT,PointNT> nest;
    //nest.setKSearch(5);
    //nest.setRadiusSearch (0.035);
    nest.setRadiusSearch (0.07);
    nest.setInputCloud (scene_normal);
    nest.compute (*scene_normal);
    nest.setInputCloud (object_normal);
    nest.compute (*object_normal);

    FeatureEstimationT fest;
    //fest.setRadiusSearch (0.05);
    fest.setRadiusSearch (0.06);
    fest.setInputCloud (object_normal);
    fest.setInputNormals (object_normal);
    fest.compute (*object_features);
    fest.setInputCloud (scene_normal);
    fest.setInputNormals (scene_normal);
    fest.compute (*scene_features);

    pcl::console::print_highlight ("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
    align.setInputSource (object_normal);
    align.setSourceFeatures (object_features);
    align.setInputTarget (scene_normal);
    align.setTargetFeatures (scene_features);
    //align.setMaximumIterations (50000); // Number of RANSAC iterations
    align.setMaximumIterations (30000); // Number of RANSAC iterations
    align.setNumberOfSamples (7); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness (5); // Number of nearest features to use
    align.setSimilarityThreshold (0.8f); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance (1.3f * leaf); // Inlier threshold
    align.setInlierFraction (0.2f); // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align (*object_aligned);
    }

    if (align.hasConverged ())
    {
        // Print results
        printf ("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation ();
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
        pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());

        // Show alignment
        pcl::visualization::PCLVisualizer visu("Alignment");
        visu.addCoordinateSystem(0.5f);
        visu.addPointCloud (object_normal, ColorHandlerT (object_normal, 255.0, 255.0, 0.0), "object_initial");
        visu.addPointCloud (scene_normal, ColorHandlerT (scene_normal, 0.0, 255.0, 0.0), "scene");
        visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
        visu.spin ();
    }
    else
    {
        pcl::console::print_error ("Alignment failed!\n");
    }
}


int
main (int argc, char *argv[])
{
    pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());

    std::vector<pcl::PointCloud<PointType>::Ptr> clusters;

    if (pcl::io::loadPCDFile (model_filename_, *model) < 0)
    {
        std::cout << "Error loading model cloud!" << std::endl;
        return (-1);
    }
    if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
    {
        std::cout << "Error loading scene cloud!" << std::endl;
        return (-1);
    }

    //simulate_imput();

    // Get the first cluster. In testing program we will take the first one,
    // but in real program we need to perform recognition with all clusters distinguished during this operation
    clusters = scene_segmentation(scene);
    scene = clusters[0];

    cout << "Initial number of points in the model cloud: " << model->size() << std::endl;
    cout << "Initial number of points in the scene cloud: " << scene->size() << std::endl;

    filter_data(model);
    filter_data(scene);

    cout << "Number of points in the model cloud after filtering: " << model->size() << std::endl;
    cout << "Number of points in the scene cloud after filtering: " << scene->size() << std::endl;

    compute_normals(model, model_normals);
    compute_normals(scene, scene_normals);

    compute_local_feautres(model, scene, model_normals, scene_normals);
    compute_global_feature(model, scene, model_normals, scene_normals);

    align_model_to_scene(model, scene);

    /*pcl::visualization::PCLVisualizer viewer ("Robot recognition");
    viewer.addPointCloud (model, "model_cloud");
    viewer.addPointCloud (scene, "scene_cloud");

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }*/

    return 0;
}