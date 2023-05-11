#include <pcl/registration/ia_ransac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <boost/thread/thread.hpp>


using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//点云可视化
void visualize_pcd(PointCloud::Ptr pcd_src,
   PointCloud::Ptr pcd_tgt,
   PointCloud::Ptr pcd_final,
   PointCloud::Ptr pcd_sac)
{
   //int vp_1, vp_2;
   // Create a PCLVisualizer object
   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   //viewer.createViewPort (0.0, 0, 0.5, 1.0, vp_1);
   //viewer.createViewPort (0.5, 0, 1.0, 1.0, vp_2);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h (pcd_src, 0, 255, 0);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h (pcd_tgt, 255, 0, 0);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h (pcd_final, 0, 0, 255);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sac_h (pcd_sac, 255, 255, 255);
   viewer.addPointCloud (pcd_src, src_h, "source cloud");
   viewer.addPointCloud (pcd_tgt, tgt_h, "tgt cloud");
   viewer.addPointCloud (pcd_final, final_h, "final cloud");
   viewer.addPointCloud (pcd_sac, sac_h, "sac cloud");
   //viewer.addCoordinateSystem(1.0);
   while (!viewer.wasStopped())
   {
       viewer.spinOnce(100);
       boost::this_thread::sleep(boost::posix_time::microseconds(100000));
   }
}

//由旋转平移矩阵计算旋转角度
void matrix2angle (Eigen::Matrix4f &result_trans,Eigen::Vector3f &result_angle)
{
  double ax,ay,az;
  if (result_trans(2,0)==1 || result_trans(2,0)==-1)
  {
      az=0;
      double dlta;
      dlta=atan2(result_trans(0,1),result_trans(0,2));
      if (result_trans(2,0)==-1)
      {
          ay=M_PI/2;
          ax=az+dlta;
      }
      else
      {
          ay=-M_PI/2;
          ax=-az+dlta;
      }
  }
  else
  {
      ay=-asin(result_trans(2,0));
      ax=atan2(result_trans(2,1)/cos(ay),result_trans(2,2)/cos(ay));
      az=atan2(result_trans(1,0)/cos(ay),result_trans(0,0)/cos(ay));
  }
  result_angle<<ax,ay,az;
}

int
   main (int argc, char** argv)
{
   //加载点云文件
   PointCloud::Ptr cloud_src_o (new PointCloud);//原点云，待配准
   pcl::io::loadPCDFile ("/home/cq/code/bag/aloam_pointcloud/velodyne/1683364417.043181419.pcd",*cloud_src_o);  
   PointCloud::Ptr cloud_tgt_o (new PointCloud);//目标点云
   pcl::io::loadPCDFile ("/home/cq/code/bag/aloam_pointcloud/data_origin_segmented&subsampled_0.1.pcd",*cloud_tgt_o);
   /*
   //调整初始位姿
   Eigen::Matrix4f transform_origin;
   transform_origin << -0.775220334530, -0.631300151348,  0.022215599194, 13.632355690002,
                        0.584285974503, -0.729967415333, -0.354622989893, -6.820482730865,
                        0.240090221167, -0.261930674314,  0.934745430946, -2.261974096298,
                        0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000;
   pcl::transformPointCloud(*cloud_src_o, *cloud_src_o, transform_origin);
   */

   clock_t start=clock();
   //去除NAN点
   std::vector<int> indices_src; //保存去除的点的索引
   pcl::removeNaNFromPointCloud(*cloud_src_o,*cloud_src_o, indices_src);
   std::cout<<"remove *cloud_src_o nan"<<endl;
   
   std::vector<int> indices_tgt;
   pcl::removeNaNFromPointCloud(*cloud_tgt_o,*cloud_tgt_o, indices_tgt);
   std::cout<<"remove *cloud_tgt_o nan"<<endl;

   //对点云进行滤波处理
   //针对scan   
/*   //下采样滤波
   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
   voxel_grid.setLeafSize(0.1,0.1,0.1);
   voxel_grid.setInputCloud(cloud_src_o);
   PointCloud::Ptr cloud_src (new PointCloud);
   voxel_grid.filter(*cloud_src);
   std::cout<<"down size *cloud_src_o from "<<cloud_src_o->size()<<"to"<<cloud_src->size()<<endl;
*/
   //statisticalOutlierRemoval统计滤波   
   pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_1;  //创建滤波器对象   
   sor_1.setInputCloud (cloud_src_o);//设置待滤波的点云   
   sor_1.setMeanK (10);//设置用于平均距离估计的 KD-tree最近邻搜索点的个数   
   sor_1.setStddevMulThresh (1.0);//设置判断是否为离群点的阀值;高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的倍数1、2、3 
   //存储滤波后的点云
   PointCloud::Ptr cloud_src (new PointCloud);
   sor_1.filter (*cloud_src);
   sor_1.setNegative (true);//这个参数不设置默认为false，true为取被过滤的点
   std::cout<<"down size *cloud_src_o from "<<cloud_src_o->size()<<"to"<<cloud_src->size()<<endl;

   //针对点云map
   //体素滤波
   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
   voxel_grid_2.setLeafSize(0.1,0.1,0.1);
   voxel_grid_2.setInputCloud(cloud_tgt_o);
   PointCloud::Ptr cloud_tgt (new PointCloud);
   voxel_grid_2.filter(*cloud_tgt);
   std::cout<<"down size *cloud_tgt_o.pcd from "<<cloud_tgt_o->size()<<"to"<<cloud_tgt->size()<<endl;
/*   //statisticalOutlierRemoval统计滤波
   
   pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_2;  //创建滤波器对象   
   sor_2.setInputCloud (cloud_tgt_o);  //设置待滤波的点云   
   sor_2.setMeanK (80);  //设置用于平均距离估计的 KD-tree最近邻搜索点的个数   
   sor_2.setStddevMulThresh (1.0);  //设置判断是否为离群点的阀值;高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的倍数1、2、3 
   //存储滤波后的点云
   PointCloud::Ptr cloud_tgt (new PointCloud);
   sor_2.filter (*cloud_tgt);
   sor_2.setNegative (true);  //这个参数不设置默认为false，true为取被过滤的点
   std::cout<<"down size *cloud_tgt_o from "<<cloud_tgt_o->size()<<"to"<<cloud_tgt->size()<<endl;
*/


   //计算表面法线
   pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
   ne_src.setInputCloud(cloud_src);
   pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
   ne_src.setSearchMethod(tree_src);
   pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
   ne_src.setRadiusSearch(1);
   ne_src.compute(*cloud_src_normals);

   pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_tgt;
   ne_tgt.setInputCloud(cloud_tgt);
   pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZ>());
   ne_tgt.setSearchMethod(tree_tgt);
   pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
   //ne_tgt.setKSearch(20);
   ne_tgt.setRadiusSearch(0.30);
   ne_tgt.compute(*cloud_tgt_normals);

   //计算FPFH
   pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_src;
   fpfh_src.setInputCloud(cloud_src);
   fpfh_src.setInputNormals(cloud_src_normals);
   pcl::search::KdTree<PointT>::Ptr tree_src_fpfh (new pcl::search::KdTree<PointT>);
   fpfh_src.setSearchMethod(tree_src_fpfh);
   pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
   fpfh_src.setRadiusSearch(0.10);
   fpfh_src.compute(*fpfhs_src);
   std::cout<<"compute *cloud_src fpfh"<<endl;

   pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_tgt;
   fpfh_tgt.setInputCloud(cloud_tgt);
   fpfh_tgt.setInputNormals(cloud_tgt_normals);
   pcl::search::KdTree<PointT>::Ptr tree_tgt_fpfh (new pcl::search::KdTree<PointT>);
   fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
   pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
   fpfh_tgt.setRadiusSearch(0.30);
   fpfh_tgt.compute(*fpfhs_tgt);
   std::cout<<"compute *cloud_tgt fpfh"<<endl;

   //SAC配准
   pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
   scia.setInputSource(cloud_src);
   scia.setInputTarget(cloud_tgt);
   scia.setSourceFeatures(fpfhs_src);
   scia.setTargetFeatures(fpfhs_tgt);
   //scia.setMinSampleDistance(1);
   //scia.setNumberOfSamples(2);
   //scia.setCorrespondenceRandomness(20);
   PointCloud::Ptr sac_result (new PointCloud);
   scia.align(*sac_result);
   std::cout  <<"sac has converged:"<<scia.hasConverged()<<"  score: "<<scia.getFitnessScore()<<endl;
   Eigen::Matrix4f sac_trans;
   sac_trans=scia.getFinalTransformation();
   std::cout<<sac_trans<<endl;
   //pcl::io::savePCDFileASCII("bunny_transformed_sac.pcd",*sac_result);
   clock_t sac_time=clock();

   //icp配准
   PointCloud::Ptr icp_result (new PointCloud);
   pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
   icp.setInputSource(cloud_src); 
   icp.setInputTarget(cloud_tgt_o);
   //设置对应点对之间的最大距离（此值对配准结果影响较大）:10cm
   //icp.setMaxCorrespondenceDistance (0.10);
   // 最大迭代次数
   icp.setMaximumIterations (500);
   // 两次变化矩阵之间的差值
   icp.setTransformationEpsilon (1e-10);
   // 均方误差
   icp.setEuclideanFitnessEpsilon (1e-6);
   icp.align(*icp_result,sac_trans);

   clock_t end=clock();
   cout<<"total time: "<<(double)(end-start)/(double)CLOCKS_PER_SEC<<" s"<<endl;
   //我把计算法线和点特征直方图的时间也算在SAC里面了
   cout<<"sac time: "<<(double)(sac_time-start)/(double)CLOCKS_PER_SEC<<" s"<<endl;
   cout<<"icp time: "<<(double)(end-sac_time)/(double)CLOCKS_PER_SEC<<" s"<<endl;

   std::cout << "ICP has converged:" << icp.hasConverged()
       << " score: " << icp.getFitnessScore() << std::endl;
   Eigen::Matrix4f icp_trans;
   icp_trans=icp.getFinalTransformation();
   //cout<<"ransformationProbability"<<icp.getTransformationProbability()<<endl;
   std::cout<<icp_trans<<endl;
   //使用创建的变换对未过滤的输入点云进行变换
   pcl::transformPointCloud(*cloud_src_o, *icp_result, icp_trans);

   //计算误差
   Eigen::Vector3f ANGLE_origin;
   ANGLE_origin<<0,0,M_PI/5;
   double error_x,error_y,error_z;
   Eigen::Vector3f ANGLE_result;
   matrix2angle(icp_trans,ANGLE_result);
   error_x=fabs(ANGLE_result(0))-fabs(ANGLE_origin(0));
   error_y=fabs(ANGLE_result(1))-fabs(ANGLE_origin(1));
   error_z=fabs(ANGLE_result(2))-fabs(ANGLE_origin(2));
   cout<<"original angle in x y z:\n"<<ANGLE_origin<<endl;
   cout<<"error in aixs_x: "<<error_x<<"  error in aixs_y: "<<error_y<<"  error in aixs_z: "<<error_z<<endl;


   //可视化
   visualize_pcd(cloud_src_o,cloud_tgt_o,icp_result,sac_result);
   return (0);
}
