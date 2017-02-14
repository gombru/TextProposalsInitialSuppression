#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include<string>

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"

#include "region.h"
#include "agglomerative_clustering.h"
//#include "utils.h"

using namespace std;
using namespace cv;


class ComputeRegions
{
public:
	
	ComputeRegions();
	void operator()(Mat img, vector<HCluster> &regions_info, string filename, string directory);
    
};

