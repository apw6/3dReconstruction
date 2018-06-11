#define OPTICAL_FLOW 0 // display first frame with tracked points and 
                       // their position in subsequent images
#define ORIGINAL_TRACKED 0 // display the first frame with tracked points
#define KLT_TRACK 1 // display first frame with tracked points and show
                    // each new frame with new location of found points
#define SHOW_SIFT 0 // display each new frame with SIFT keypoints shown

#define TRACK_FROM_ORIGINAL 0 // Track from original frame to current frame. 
                              // Otherwise track from the previous frame 

#define CERES_FOUND 1 //need this to make sfm module happy

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/sfm.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

const int windowsSize = 10;
const float framesToProcess = 10;
const int minTrackedPoints = 10;

const string videoName = "test3.mp4";
const string trackFileName = "tracks.txt";
const string pmvs2TxtDir = "root/txt/";
const string pmvs2ImgDir = "root/visualize/";
const string pmvs2ModelDir = "root/models/";

void writePMVS2(const  vector<Mat> &, const  vector<Mat> &);
void writeTracks(const vector<vector<Point2f>> &);

int main() {
    //setup video cap
    auto cap = cv::VideoCapture(videoName);
    auto frameNum = cap.get(CV_CAP_PROP_POS_FRAMES);
    auto totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    
    //setup frame storage
    vector<Mat> frameStore;
    Mat prevFrame, currentFrame, prevFrameGrey, currentFrameGrey;
    
    //get first frame
    cap >> currentFrame;
    frameStore.push_back(currentFrame.clone());
    auto originalFrame = prevFrame.clone();
    
    //setup corner point tracking and get feature for first frame
    vector<Point2f> trackedCorners;
    vector < vector<Point2f>> trackedPoints;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);
    cvtColor(currentFrame, currentFrameGrey, CV_BGR2GRAY);
    goodFeaturesToTrack(currentFrameGrey, trackedCorners, 0, .1, 10, noArray(), 3, false);
    cornerSubPix(currentFrameGrey, trackedCorners, subPixWinSize, Size(-1,-1), termcrit);
    trackedPoints.push_back(trackedCorners);
    
    #if OPTICAL_FLOW || ORIGINAL_TRACKED || KLT_TRACK
    //Display of optial flow from first image
    Mat opticalFlowEx = currentFrame.clone();
    for(auto itr = trackedCorners.begin(); itr < trackedCorners.end(); itr++){
        circle(opticalFlowEx, *itr, 1, CV_RGB(0,200,0), 5);
    }
    imshow("Tracked Points", opticalFlowEx);
    waitKey(50000);
    #endif
    
    #if SHOW_SIFT
    //Sift
    auto siftDetector = SIFT::create();
    vector<KeyPoint> siftFeatures;
    #endif
    
    if(trackedCorners.size() < minTrackedPoints) {
       printf("Too few features detected");
       exit(EXIT_SUCCESS);
    }
    
    while (frameNum < framesToProcess) {
        
        #if TRACK_FROM_ORIGINAL
        prevFrame = frameStore.front();
        #else
        prevFrame = currentFrame.clone();
        #endif
        
        
        cap >> currentFrame;
        frameStore.push_back(currentFrame.clone());
        
        frameNum = cap.get(CV_CAP_PROP_POS_FRAMES);
        
        cvtColor(prevFrame, prevFrameGrey, CV_BGR2GRAY);
        cvtColor(currentFrame, currentFrameGrey, CV_BGR2GRAY);

        vector<uchar> status;
        vector<float> err;
        
        #if TRACK_FROM_ORIGINAL
        const vector<Point2f> & prevPoints = trackedPoints.front();
        #else
        const vector<Point2f> & prevPoints = trackedPoints.back();
        #endif
        
        trackedCorners = trackedPoints.back();  //use last known locations as best guess
        calcOpticalFlowPyrLK(prevFrameGrey, currentFrameGrey, prevPoints, trackedCorners, status, err);

        Mat currentFrameKLT = currentFrame.clone();
        for (int i = 0; i < status.size(); i++) {
            if (status[i] == 0) {
                (trackedCorners)[i] = Point2f(-1, -1);
            }
            else {
                #if OPTICAL_FLOW
                circle(opticalFlowEx,  (trackedCorners)[i], 1, CV_RGB(0,0,200), 1);
                #endif
                #if KLT_TRACK
                circle(currentFrameKLT, (trackedCorners)[i], 1, CV_RGB(200,0,200), 5);
                #endif
            }
        }
        trackedPoints.push_back(trackedCorners);
        
        #if SHOW_SIFT
        //show sift features to compare
        Mat currentFrameSIFT = currentFrame.clone();
        siftDetector->detect(currentFrameSIFT, siftFeatures);
        drawKeypoints(currentFrameSIFT, siftFeatures, currentFrameSIFT);
        imshow("New Frame SIFT", currentFrameSIFT);
        #endif
        
        #if KLT_TRACK
        imshow("New Frame Tracked", currentFrameKLT);
        #endif
        
        #if KLT_TRACK || SHOW_SIFT
        waitKey(500);
        #endif
    }
    
    #if OPTICAL_FLOW
    imshow("Tracked Points", opticalFlowEx);
    #endif
    
    Mat cameraInstrinsics;
    cameraInstrinsics = Mat(Matx33d( 1230, 0, 960,
                                 0, 1230, 540,
                                 0, 0,  1));
    
    vector<Mat> points3D;
    vector<Mat> projections;
    vector<Mat> points2D;
    
    for (auto itr = trackedPoints.begin(); itr != trackedPoints.end(); itr++){
        Mat cur2DPoints(2, itr->size(), CV_64FC1);
        for (int i = 0; i < itr->size(); i++){
            cur2DPoints.at<double>(0,i) = (*itr)[i].x;
            cur2DPoints.at<double>(1,i) = (*itr)[i].y;
        }
        points2D.push_back(Mat(cur2DPoints).clone());
    }

    
    cout << "Begin reconstruction" << endl;
    sfm::reconstruct(points2D, projections, points3D, cameraInstrinsics, true);
    cout << "Reconstruction complete" << endl;
    
    writePMVS2(projections, frameStore);
    
    writeTracks(trackedPoints);
    
    cout << "Done" << endl;
}


/* writePMVS2 - Create the directory structure and files for pmvs2
   precondition: projection matrices are valid 3x4 and frames store contains valid images 
   postconditions: The directory structure for pmvs2 is created. Root dir contains option file
   and other dirs. Txt dir contains text documents for the projection mats. Visualize dir contains
   the frames written as jpg. Model dir is left empty for use by pmvs2.
*/	
void writePMVS2(const  vector<Mat> & projections, const  vector<Mat> & frameStore){
    //create dirs
    system(("mkdir -p " + pmvs2ImgDir).c_str());
    system(("mkdir -p " + pmvs2TxtDir).c_str());
    system(("mkdir -p " + pmvs2ModelDir).c_str());
    
    char strBuff[256];
    
    //write projection mats and images
    for (int idx = 0; idx < projections.size(); idx++){
        
        //projection
        Mat projectionMat = projections[idx];
        ofstream outFile;
        sprintf(strBuff, (pmvs2TxtDir + "%04d.txt").c_str(), idx);
        outFile.open(strBuff);
        outFile << "CONTOUR" << endl;
        for (int row = 0; row < projectionMat.rows; row++){
            for (int col = 0; col < projectionMat.cols; col++){
                outFile << projectionMat.at<double>(row, col) << " ";
            }
            outFile << endl;
        }
        outFile.close();
        
        //images
        sprintf(strBuff, (pmvs2ImgDir + "%04d.jpg").c_str(), idx);
        imwrite(strBuff, frameStore[idx]);
    }
    
    //write option file for pmvs2
    ofstream optionfile("root/options.txt");
    optionfile << "timages  -1 " << 0 << " " << projections.size() << endl;;
    optionfile << "oimages 0" << endl;
    optionfile << "level 1" << endl;
}


/* writeTracks - Writes to a file the track for each point through the images
   precondition: trackedPoints contains vectors of the same size containing points
   postconditions: a file is writen that contains the track in the format
                   im1x1 im1y1 im2x1 im2y1 ... \n
                   im1x2 im1y2 im2x2 im2y2 ... \n
                   ...
*/	
void writeTracks(const vector<vector<Point2f>> & trackedPoints){
    const int pointsPerTrack = trackedPoints[0].size();
    
    ofstream tracksFile(trackFileName);
    for (int i = 0; i < pointsPerTrack; i++) {
        for (int k = 0; k < trackedPoints.size(); k++) {
            Point2d point = trackedPoints[k][i];
            tracksFile << point.x << " " << point.y << " ";
        }
        tracksFile << "\n";
    }
    tracksFile.close();
}

