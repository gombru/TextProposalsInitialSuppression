#include "compute_regions.h"
#include "utils.h"
#include<string>
/* Diversivication Configurations :                                     */
/* These are boolean values, indicating whenever to use a particular    */
/*                                   diversification strategy or not    */

#define PYRAMIDS     1 // Use spatial pyramids
#define CUE_D        1 // Use Diameter grouping cue
#define CUE_FGI      1 // Use ForeGround Intensity grouping cue
#define CUE_BGI      1 // Use BackGround Intensity grouping cue
#define CUE_G        1 // Use Gradient magnitude grouping cue
#define CUE_S        1 // Use Stroke width grouping cue
#define CUE_FCN      1 // Use FCN score as grouping cue
#define CHANNEL_I    0 // Use Intensity color channel
#define CHANNEL_R    1 // Use Red color channel
#define CHANNEL_G    1 // Use Green color channel
#define CHANNEL_B    1 // Use Blue color channel

#define INITIAL_SUPPRESSION    1 // Do Initial Suppression
#define POST_SUPPRESSION    0 // Do Post Suppression


ComputeRegions::ComputeRegions()
{

}


void ComputeRegions::operator()(Mat img, vector<HCluster> &regions_info, string filename, string directory)
{
    // Params    
    float SUPPRESSION_THRESHOLD	= 0.05; //Suppress initial regions below this energy
    float x_coord_mult              = 0.25; // a value of 1 means rotation invariant

    // Pipeline configuration
    bool conf_channels[4]={CHANNEL_R,CHANNEL_G,CHANNEL_B,CHANNEL_I};
    bool conf_cues[6]={CUE_D,CUE_FGI,CUE_BGI,CUE_G,CUE_S,CUE_FCN};

    /* initialize random seed: */
    srand (time(NULL));

    Mat src, grey, lab_img, gradient_magnitude, heatmap; //,img

    img.copyTo(src);
    
    //Read heatmap
    heatmap = imread(directory + "val-former-legible-heatmaps/" + filename.substr(0, filename.length() - 4) + ".png", CV_LOAD_IMAGE_GRAYSCALE);

    //Open output file
    ofstream myfile;
    string evaluation_filename(directory + "regions_evaluation/initial_suppression_05/" + filename.substr(0, filename.length() - 3) + "csv");	
    myfile.open (evaluation_filename.c_str());
    
    int delta = 13;
    int img_area = img.cols*img.rows;
    Ptr<MSER> cv_mser = MSER::create(delta,(int)(0.00002*img_area),(int)(0.11*img_area),55,0.);

    cvtColor(img, grey, CV_BGR2GRAY);
    cvtColor(img, lab_img, CV_BGR2Lab);
    gradient_magnitude = Mat_<double>(img.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    vector<Mat> channels;
    split(img, channels);
    channels.push_back(grey);
    int num_channels = channels.size();

    if (PYRAMIDS)
    {
      for (int c=0; c<num_channels; c++)
      {
        Mat pyr;
        resize(channels[c],pyr,Size(channels[c].cols/2,channels[c].rows/2));
        //resize(pyr,pyr,Size(channels[c].cols,channels[c].rows));
        channels.push_back(pyr);
      }
      /*for (int c=0; c<num_channels; c++)
      {
        Mat pyr;
        resize(channels[c],pyr,Size(channels[c].cols/4,channels[c].rows/4));
        //resize(pyr,pyr,Size(channels[c].cols,channels[c].rows));
        channels.push_back(pyr);
      }*/
    }

    for (int c=0; c<channels.size(); c++)
    {

        if (!conf_channels[c%4]) continue;

        if (channels[c].size() != grey.size()) // update sizes for smaller pyramid lvls
        {
          resize(grey,grey,Size(channels[c].cols,channels[c].rows));
          resize(lab_img,lab_img,Size(channels[c].cols,channels[c].rows));
          resize(gradient_magnitude,gradient_magnitude,Size(channels[c].cols,channels[c].rows));
        }

        // TODO you want to try single pass MSER?
        //channels[c] = 255 - channels[c];
        //cv_mser->setPass2Only(true);

        /* Initial over-segmentation using MSER algorithm */
        vector<vector<Point> > contours;
        vector<Rect>  mser_bboxes;
        //t = (double)getTickCount();
        cv_mser->detectRegions(channels[c], contours, mser_bboxes);
        //cout << " OpenCV MSER found " << contours.size() << " regions in " << ((double)getTickCount() - t)*1000/getTickFrequency() << " ms." << endl;


        /* Extract simple features for each region */
        vector<Region> regions;
        Mat mask = Mat::zeros(grey.size(), CV_8UC1);
        int max_stroke = 0;
        for (int i=contours.size()-1; i>=0; i--)
        {
	  
	  
	    //INITIAL SUPPRESSION
	    if (INITIAL_SUPPRESSION )
	    {
	      //Check FCN heatmap energy. Suppress region if its below a threshold
	      Mat regionHeatMap;
	      float energy;
	      
	      if (c<=num_channels) 
	      {
		  regionHeatMap = heatmap(mser_bboxes[i]);
	      }
	      
	      else   //If the channel is from the pyramid pick the correct heatmap region  
	      {
		  Rect rect;
		  rect.x = mser_bboxes[i].x * 2;
		  rect.y = mser_bboxes[i].y * 2;
		  rect.width = mser_bboxes[i].width * 2;
		  rect.height = mser_bboxes[i].height * 2;
		  regionHeatMap = heatmap(rect);		
	      }
	      
	      energy = sum(regionHeatMap)[0] / (regionHeatMap.total() * 255);
	      /*
	      double min, max;
	      minMaxIdx(regionHeatMap, &min, &max);
	      max = max / 255;
	      */
	      if(energy < SUPPRESSION_THRESHOLD) continue;
	     }
	    
            Region region;
            region.pixels_.push_back(Point(0,0)); //cannot swap an empty vector
            region.pixels_.swap(contours[i]);
            region.bbox_ = mser_bboxes[i];
            region.extract_features(lab_img, grey, gradient_magnitude, heatmap, mask, conf_cues);
            max_stroke = max(max_stroke, region.stroke_mean_);
            regions.push_back(region);
        }

        unsigned int N = regions.size();
        if (N<3) continue;
        int dim = 3;
        t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));

        /* Single Linkage Clustering for each individual cue */
        for (int cue=0; cue<6; cue++)
        {

          if (!conf_cues[cue]) continue;

          int count = 0;
          for (int i=0; i<regions.size(); i++)
          {
            data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/channels[c].cols*x_coord_mult;
            data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/channels[c].rows;
            switch(cue)
            {
              case 0:
                data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(channels[c].rows,channels[c].cols);
                break;
              case 1:
                data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
                break;
              case 2:
                data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;
                break;
              case 3:
                data[count+2] = (t_float)regions.at(i).gradient_mean_/255;
                break;
              case 4:
                data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;
                break;
              case 5:
                data[count+2] = (t_float)regions.at(i).fcn_score_mean_/255;
                break;
            }
            count = count+dim;
          }

          HierarchicalClustering h_clustering(regions);
          vector<HCluster> dendogram;
          h_clustering(data, N, dim, (unsigned char)0, (unsigned char)3, dendogram, x_coord_mult, channels[c].size());

	
	//Save regions data to a file  	
	for (int k=0; k<dendogram.size(); k++)
          {
	    
	    int ml = 1;
	    if (c>=num_channels) ml=2;// update sizes for smaller pyramid lvls
	    if (c>=2*num_channels) ml=4;// update sizes for smaller pyramid lvls

	    /*
	    //Save also mean energy 
	    Mat regionHeatMap;
	    float energy;
	    if (c<=num_channels) 
	    {
	      	regionHeatMap = heatmap(dendogram[k].rect);

	    }
	    
	    else   //If the channel is from the pyramid pick the correct heatmap region  
	    {
		Rect rect;
		rect.x = dendogram[k].rect.x * 2;
		rect.y = dendogram[k].rect.y * 2;
		rect.width = dendogram[k].rect.width * 2;
		rect.height = dendogram[k].rect.height * 2;
		regionHeatMap = heatmap(rect);		
	    }
	    
	    energy = sum(regionHeatMap)[0] / (regionHeatMap.total() * 255);
	    
	    
	    //POSTERIORI SUPPRESSION
	    if(POST_SUPPRESSION)
	    {
	      double min, max;
	      minMaxIdx(regionHeatMap, &min, &max);
	      if(max/255 < SUPPRESSION_THRESHOLD)
		continue;
	    }
	    */
	    
	      
	    //Longability computation
	    Mat regionHeatMap;
	    float longality;
	    int pyramid_factor = 1;
	    float l_x;
	    float l_y;
	    int l_start;
	    int l_end;
	    float l_TH = 0.10;
	    bool assigned = false;
	    
	    if (c<=num_channels) 
	    {
	      	regionHeatMap = heatmap(dendogram[k].rect);
	    }
	    
	    else   //If the channel is from the pyramid pick the correct heatmap region  
	    {
		Rect rect;
		rect.x = dendogram[k].rect.x * 2;
		rect.y = dendogram[k].rect.y * 2;
		rect.width = dendogram[k].rect.width * 2;
		rect.height = dendogram[k].rect.height * 2;
		regionHeatMap = heatmap(rect);	
		pyramid_factor = 2;

	    }
	    
	    
	    l_y = dendogram[k].rect.height * pyramid_factor;	  
	    l_start = 0;
	    l_end = 0;
	    
	    // Cycle col step
	    for (int col = 0; col <= dendogram[k].rect.width * pyramid_factor; col += 1)
	    {
	      if (assigned){break;}
	      
	      // Cycle row step
	      for (int row = 0; row <= dendogram[k].rect.height * pyramid_factor; row += 1)
	      {
		if(((float)regionHeatMap.at<char>(row,col) / 255) > l_TH)
		{
		  l_start = col;
		  assigned = true;
		  break;
		}		
	      }
	    }
	    
	    assigned = false;
	    
	    // Cycle col step
	    for (int col = dendogram[k].rect.width * pyramid_factor; col >= 0; col -= 1)
	    {
	      if (assigned){break;}
	      
	      // Cycle row step
	      for (int row = 0; row <= dendogram[k].rect.height * pyramid_factor; row += 1)
	      {
		if(((float)regionHeatMap.at<char>(row,col) / 255) > l_TH)
		{
		  l_end = col;
		  assigned = true;
		  break;
		}		
	      }
	    }
	    
	    l_x = l_end - l_start; 
	    	    
	    longality = l_x / (l_x + l_y);
	    

	      
	    int aux = 0;
	    myfile << dendogram[k].rect.x*ml << "," << dendogram[k].rect.y*ml << ","
	    << dendogram[k].rect.width*ml << "," << dendogram[k].rect.height*ml << ","
	    << (float)dendogram[k].probability << "," << (float)longality << "," << l_start << "," <<  l_end << ","<< aux  << endl;	
	    
	   }
          
    
        }
        free(data);

    }

    
    myfile.close();
}
