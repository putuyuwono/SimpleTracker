#include <stdlib.h>
#include <iostream>
#include "opencv2\highgui.hpp"
#include "opencv2\bgsegm.hpp"
#include "opencv\cvblob.h"

using namespace std;
using namespace cv;
using namespace cvb;

static bool paused = false;
static bool terminated = false;

#define YELLOW Scalar(0, 255, 255)
#define RED Scalar(0, 0, 255)
#define BLUE Scalar(255, 0, 0)
#define GREEN Scalar(0, 255, 0)

#define ESCKEY 27
#define SPCKEY 32

void KeyPressHandler()
{
	int key = waitKey(20);
	if (key < 0) return;
	switch (key)
	{
	case ESCKEY: //Esc key
		terminated = !terminated;
		cout << "Terminated" << endl;
		break;
	case SPCKEY: //Space key
		paused = !paused;
		cout << "Paused: " << (paused ? "true" : "false") << endl;
		break;
	default:
		cout << "Unsupported Key: " << key << endl;
		break;
	}
}

void DetectTarget(CvBlobs &blobs, Mat fg_mask, Mat ori_frame, Mat &blob_frame)
{
	blobs = CvBlobs();

	IplImage foreground = fg_mask;

	IplImage *labelImg = cvCreateImage(cvGetSize(&foreground), IPL_DEPTH_LABEL, 1);
	uint result = cvLabel(&foreground, labelImg, blobs);
	cvFilterByArea(blobs, 100, 100000);

	ori_frame.copyTo(blob_frame);
	IplImage original = ori_frame;	
	IplImage blobframe = blob_frame;
	cvRenderBlobs(labelImg, blobs, &original, &blobframe);
}

void TrackTarget(CvBlobs blobs, CvTracks &tracks, Mat ori_frame, Mat &track_frame)
{
	cvUpdateTracks(blobs, tracks, 30, 1000, 15);
	
	ori_frame.copyTo(track_frame);
	IplImage original = ori_frame;
	IplImage trackframe = track_frame;
	cvRenderTracks(tracks, &original, &trackframe);
}

void main(){
	string filename = "E:\\My Video\\Sample\\Dome\\640.avi";
	VideoCapture capture(filename);
	
	Ptr<BackgroundSubtractorMOG2> fg_extractor;
	fg_extractor = createBackgroundSubtractorMOG2();

	CvBlobs blobs;
	CvTracks tracks;

	Mat ori_frame, fg_mask, fg_frame, pfg_mask, blob_frame, track_frame;
	while (!terminated)
	{
		if (!paused)
		{
			if (capture.read(ori_frame))
			{
				fg_extractor->apply(ori_frame, fg_mask);
				GaussianBlur(fg_mask, pfg_mask, Size(13, 13), 3);
				threshold(pfg_mask, pfg_mask, 150, 150, THRESH_TOZERO);
				
				ori_frame.copyTo(fg_frame, pfg_mask);
				
				DetectTarget(blobs, pfg_mask, ori_frame, blob_frame);
				TrackTarget(blobs, tracks, ori_frame, track_frame);

				//imshow("ORIGINAL IMAGE", ori_frame);
				//imshow("RAW FOREGROUND", fg_mask);
				//imshow("PROCESSSED FOREGROUND", pfg_mask);
				//imshow("FG Frame", fg_frame);
				//imshow("DETECTION", blob_frame);
				imshow("TRACKING", track_frame);

				ori_frame.release();
				fg_mask.release();
				pfg_mask.release();
				fg_frame.release();
				//blob_frame.release();
				//track_frame.release();
			}
		}		

		KeyPressHandler();
	}
}