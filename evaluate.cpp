#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include<cstdlib>
#include<string>
#include<vector>
#include<dirent.h>
#include <time.h>

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"

#include "compute_regions.h"

using namespace std;
using namespace cv;

#define SAVE_IMAGE_INFO    0 // Saves in a txt file each image filename and its score
#define SAVE_REGION_INFO       1 // Saves a txt filo for each image with its region BB and their score



int main()
{  
  
  /*
  string img_path("/home/rgomez/caffe-master/data/ICDAR/ICDAR_myTest/img_801.jpg");
  cv::Mat img2 = cv::imread(img_path, -1);
  const int w = 100;
  const int h = 200;
  cv::resize(img2, img2, cvSize(w, h));
   */
   
   
   
  cout << "\nStarting ...";
  string directory("/home/rgomez/datasets/COCO-Text/");
  DIR *dpdf;
  struct dirent *epdf;
  string filename;
  Mat img;
  Mat heatmap;
  int count = 0;
  int msec = 0;
  
  
  //Build evaluation file  
  ofstream imevfile;
  //string evaluation_filename(directory + "images_evaluation/images_scores.txt");
  //imevfile.open (evaluation_filename.c_str());
  
  cout << "\nOpening dir...";
  dpdf = opendir((directory + "val-former-legible/").c_str());
  //clock_t start = clock(), diff;
  

  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
	filename = epdf->d_name;
	
	if (filename.length() < 4){
	  continue;}  
	 if (filename.compare(filename.length()-4,filename.length(),".jpg")){
	  continue;}  
	
	//clock_t start_i = clock(), diff;
	
	count ++;
	img = imread(directory + "val-former-legible/" + filename);
	cout << "\nImage read: " << filename;
	
	
	ComputeRegions segmentator;
	vector<HCluster> regions_info;
	segmentator(img, regions_info, filename, directory);

	
  }
  //diff = clock() - start;
  printf("\nNum images: %d", count);
  //msec = diff * 1000 / CLOCKS_PER_SEC;
  //printf("\nTime taken %d seconds %d milliseconds", msec/1000, msec%1000);
  //printf("\nTime taken per image has been %d seconds %d milliseconds", (msec/count)/1000, (msec/count)%1000); 
  cout << "\nDone";	
  imevfile.close();
  }}






