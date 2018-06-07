#define CERES_FOUND 1
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/sfm.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

int main() {
    //setup video cap and get first frame
    auto cap = cv::VideoCapture("test3.mp4");
    vector<Mat> frameStore;
    Mat frame1, frame2, frame1Grey, frame2Grey;
    cap >> frame1;
    frameStore.push_back(frame1.clone());
    frame2 = frame1;
    auto originalFrame = frame1.clone();
    auto frameNum = cap.get(CV_CAP_PROP_POS_FRAMES);
    auto totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    
    //setup corner point tracking and get feature for first frame
    vector<Point2f> trackedCorners, newPointLocations;
    vector < vector<Point2f>> trackedPoints;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);
    cvtColor(frame1, frame1Grey, CV_BGR2GRAY);
    goodFeaturesToTrack(frame1Grey, trackedCorners, 0, .1, 10, noArray(), 3, false);
    //cornerSubPix(frame1Grey, trackedCorners, subPixWinSize, Size(-1,-1), termcrit);
    trackedPoints.push_back(trackedCorners);
    
    //Display of optial flow from first image
    Mat opticalFlowEx = frame1.clone();
    for(auto itr = trackedCorners.begin(); itr < trackedCorners.end(); itr++){
        circle(opticalFlowEx, *itr, 1, CV_RGB(0,200,0), 5);
    }
//    imshow("Tracked Points", opticalFlowEx);
//    waitKey(50000);
//    
    //Sift
    auto siftDetector = SIFT::create();
    vector<KeyPoint> siftFeatures;
    
    if(trackedCorners.size() < 10) {
       printf("Too few features detected");
       exit(EXIT_SUCCESS);
    }
    
    while (frameNum < 30) {
        frame1 = frame2.clone();
        frame1 = originalFrame;
        cap >> frame2;
        frameStore.push_back(frame2.clone());
        frameNum = cap.get(CV_CAP_PROP_POS_FRAMES);
        
//        //sharpen frame
//        Mat blurred;
//        GaussianBlur(frame2, blurred, Size(0,0), 3);
//        addWeighted(frame2, 2, blurred, -1, 0, frame2);
        
        cvtColor(frame1, frame1Grey, CV_BGR2GRAY);
        cvtColor(frame2, frame2Grey, CV_BGR2GRAY);

        vector<uchar> status;
        vector<float> err;
        
        newPointLocations = trackedPoints.back();
        calcOpticalFlowPyrLK(frame1Grey, frame2Grey, trackedPoints.front(), newPointLocations, status, err);

        Mat frame2KLT = frame2.clone();
        for (int i = 0; i < status.size(); i++) {
            if (status[i] == 0) {
                (newPointLocations)[i] = Point2f(-1, -1);
            }
            else {
                circle(opticalFlowEx,  (newPointLocations)[i], 1, CV_RGB(0,0,200), 1);
                circle(frame2KLT, (newPointLocations)[i], 1, CV_RGB(200,0,200), 5);
            }
        }
        trackedPoints.push_back(newPointLocations);
        
//        //setup best guess for each point for next frame
//        newPointLocations = trackedPoints.back();
//        for (int i = 0; i < newPointLocations.size(); i++){
//            if (status[i] == 0) {
//                //find most recent frame where we saw point
//                for(int k = trackedPoints.size() - 2; k >= 0; k--){
//                    Point2f point = trackedPoints[k][i];
//                    if (point.x != -1){
//                        newPointLocations[i] = point;
//                        break;
//                    }
//                }
//            }
//        }
        
//        Mat frame2SIFT = frame2.clone();
//        siftDetector->detect(frame2SIFT, siftFeatures);
//        drawKeypoints(frame2SIFT, siftFeatures, frame2SIFT);
//        imshow("New Frame SIFT", frame2SIFT);
        
//        imshow("New Frame Tracked", frame2KLT);
//        
//        waitKey(500);
    }
    
//    imshow("Tracked Points", opticalFlowEx);
    
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
    
//    //debuging input types
//    printf("%d\n",_InputArray(projections).kind()>>16);
//    exit(0);
    
    sfm::reconstruct(points2D, projections, points3D, cameraInstrinsics, true);
    
    //write out projection mats and images for pmvs2
    system("mkdir -p root/visualize/");
    system("mkdir -p root/txt/");
    system("mkdir -p root/models/");
    
    char strBuff[256];
    int idx = 0;
    
    for (auto itr = projections.begin(); itr != projections.end(); itr++, idx++){
        
        //projection
        ofstream outFile;
        sprintf(strBuff, "root/txt/%04d.txt", idx);
        outFile.open(strBuff);
        outFile << "CONTOUR" << endl;
        for (int row = 0; row < itr->rows; row++){
            for (int col = 0; col < itr->cols; col++){
                outFile << itr->at<double>(row, col) << " ";
            }
            outFile << endl;
        }
        outFile.close();
        
        //images
        sprintf(strBuff, "root/visualize/%04d.jpg", idx);
        imwrite(strBuff, frameStore[idx]);
    }
    
    //write option file for pmvs2
    ofstream optionfile("root/options.txt");
    optionfile << "timages  -1 " << 0 << " " << projections.size() << endl;;
    optionfile << "oimages 0" << endl;
    optionfile << "level 1" << endl;


    //write out tracks
    const int windowsSize = 10;

    for (int window = 0; window < trackedPoints.size() / windowsSize; window++) {
        
        //Setup initial points
        int frameIdx = window * windowsSize;
        vector<int> highQualityPointsIdx;
        for (int i = 0; i < trackedCorners.size(); i++) {
            if (trackedPoints[frameIdx][i].x != -1) {
                highQualityPointsIdx.push_back(i);
            }
        }
        
        //remove all points that aren't tracked throughout the window
        frameIdx++;
        for (; frameIdx < (window + 1) * windowsSize; frameIdx++) {
            for (auto itr = highQualityPointsIdx.begin(); itr != highQualityPointsIdx.end();) {
                if (trackedPoints[frameIdx][*itr].x == -1) {
                    itr = highQualityPointsIdx.erase(itr);
                }
                else {
                    itr++;
                }
            }
        }

        //pick some frames in the window and get set of "high quality" points from those frames
        vector<vector<Point2f>> highQualityPoints;
        for (frameIdx = window * windowsSize; frameIdx < (window + 1) * windowsSize;frameIdx+=(windowsSize/2-1) ) {
            vector<Point2f > points;
            for (auto itr = highQualityPointsIdx.begin(); itr != highQualityPointsIdx.end(); itr++) {
                points.push_back(trackedPoints[frameIdx][*itr]);
            }
            highQualityPoints.push_back(points);
        }
        
    }
    
    ofstream tracksFile("tracks.txt");
    //Print out the tracks
    for (int i = 0; i < trackedCorners.size(); i++) {
        for (int k = 0; k < trackedPoints.size(); k++) {
            Point2d point = trackedPoints[k][i];
            tracksFile << point.x << " " << point.y << " ";
        }
        tracksFile << "\n";
    }
    tracksFile.close();
    cout << "Done" << endl;
}
