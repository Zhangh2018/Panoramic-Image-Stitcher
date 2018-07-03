#pragma once

#include <opencv2/opencv.hpp>



#define SIFT_GRID_SIZE 16
#define SIFT_HALF_GRID_SIZE ((int) SIFT_GRID_SIZE >> 1)
#define SIFT_QUARTER_GRID_SIZE ((int)SIFT_GRID_SIZE >> 2)
#define SIFT_EIGTH_GRID_SIZE ((int)SIFT_GRID_SIZE >> 3)
#define SIFT_DESCRIPTOR_SIZE SIFT_GRID_SIZE * SIFT_BINS
#define SIFT_BINS 8
#define SIFT_ANGLE_PER_BIN ((int)360 / SIFT_BINS)
#define IMAGE_PAD_SIZE 2 * SIFT_HALF_GRID_SIZE

#define ROTATION_INVARIANCE_BINS 36
#define ROTATION_INVARIANCE_ANGLE_PER_BIN ((int)360 / ROTATION_INVARIANCE_BINS)


#define CORNER_RESPONSE_THRESHOLD 0.6f
#define GAUSS_SIGMA 0.5f
#define GAUSS_KERNEL_SIZE 5
#define SUPRESS_KERNEL_SIZE 21
#define DESCRIPTOR_CLAMP_VALUE 0.2f
#define RATIO_SSD_THRESHOLD 0.8f
#define DEG_2_RAD CV_PI / 180



#define uint32 unsigned int
#define NDEBUG
#include <assert.h>



struct GradientData {
	cv::Mat dx;
	cv::Mat dy;
	cv::Mat dxdy;
	cv::Mat angle;
	cv::Mat mag;

};

struct KeypointDescriptor {
	cv::KeyPoint keypoint;
	cv::Mat descriptor;
	uint32 descIndex;
		

	KeypointDescriptor(float x, float y, float mag = 0, float angle = 0) {
		keypoint.pt.x = x;
		keypoint.pt.y = y;
		keypoint.angle = angle;
		keypoint.size = mag;
	}
};

struct DesriptorMatch {

	float bestSsd;
	float secondBestSsd;
	float ratio;
	union {
		uint32 bestSsdIndex;
		uint32 matchIndex;
	};
	uint32 secondBestSsdIndex;
	uint32 srcIndex;
	bool useMatch;

	KeypointDescriptor* srcDesc = nullptr;
	KeypointDescriptor* dstDesc = nullptr;

};

struct InlierSet {
	std::vector<cv::Point2f> src;
	std::vector<cv::Point2f> dst;
	size_t count;
};


void getGradient(cv::Mat& src, cv::Mat& dst, int dx, int dy) {
		using namespace cv;
		src.convertTo(src, CV_32F);
		dst.convertTo(dst, CV_32F);
		// Convert our images to float 32 bit precision
		Sobel(src, dst, CV_32F, dx, dy, 3);
	}

float inline cornerResponse2x2(cv::Mat& src) {
		float a = src.at<float>(0, 0);
		float b = src.at<float>(0, 1);
		float c = src.at<float>(1, 0);
		float d = src.at<float>(1, 1);

		return ((a * d) - (b * c)) / (a + d);

		//return (cv::determinant(src) / trace(src))[0];
	}

void localNonMaxSupress(cv::Mat& src, int ksize, std::vector<KeypointDescriptor>& keypoints) {
		using namespace cv;

		for (int i = 0; i < src.rows - ksize; i += ksize) {
			for (int j = 0; j < src.cols - ksize; j += ksize) {
				int iMax = -1;
				int jMax = -1;
				float max = -1;
				for (int ki = 0; ki < ksize; ++ki) {
					for (int kj = 0; kj < ksize; ++kj) {
						float test = src.at<float>(i + ki, j + kj);
						src.at<float>(i + ki, j + kj) = 0;
						if (test > max) {
							iMax = ki;
							jMax = kj;
							max = test;
						}
					}
				}
				if (max > 0) {
					int x = i + iMax;
					int y = j + jMax;
					src.at<float>(x, y) = max;
					// Note: for some reason the points in opencv are flipped
					// (x,y) image coordinates in image are mapped to (y,x) in opencv's point
					keypoints.push_back(KeypointDescriptor(y, x));
				}
			}
		}

	}

void harrisCornerDetect(cv::Mat& src, std::vector<KeypointDescriptor>& keypoints, GradientData& gData, float cornerResponseThreshold, int suppressKSize) {
		using namespace  cv;
		// Convert our image to grayscale
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

		// Set up matrices to compute edge directions for the harris matrix
		gData.dx = Mat::zeros(src.rows, src.cols, CV_32F);
		gData.dy = Mat::zeros(src.rows, src.cols, CV_32F);
		gData.dxdy = Mat::zeros(src.rows, src.cols, CV_32F);

		Mat ix2 = Mat::zeros(src.rows, src.cols, CV_32F);
		Mat iy2 = Mat::zeros(src.rows, src.cols, CV_32F);

		Mat ix2Sum = Mat::zeros(src.rows, src.cols, CV_32F);
		Mat iy2Sum = Mat::zeros(src.rows, src.cols, CV_32F);
		Mat ixiySum = Mat::zeros(src.rows, src.cols, CV_32F);



		Mat sumFilter = Mat::ones(3, 3, CV_32F);


		// Compute derivatives Ix^2, Iy^2, IxIy
		getGradient(src, gData.dx, 1, 0);
		getGradient(src, gData.dy, 0, 1);
		getGradient(src, gData.dxdy, 1, 1);

		pow(gData.dx, 2, ix2);
		pow(gData.dy, 2, iy2);



		// Smooth them with a gaussian
		Size gaussSize = Size(GAUSS_KERNEL_SIZE, GAUSS_KERNEL_SIZE);

		GaussianBlur(ix2, ix2, gaussSize, GAUSS_SIGMA, GAUSS_SIGMA);
		GaussianBlur(iy2, iy2, gaussSize, GAUSS_SIGMA, GAUSS_SIGMA);
		GaussianBlur(gData.dxdy, gData.dxdy, gaussSize, GAUSS_SIGMA, GAUSS_SIGMA);

		// Compute the harris matrix
		Mat harrisMatrix = Mat::zeros(2, 2, CV_32F);

		// Get the sum values for each pixel
		filter2D(ix2, ix2Sum, CV_32F, sumFilter);
		filter2D(iy2, iy2Sum, CV_32F, sumFilter);
		filter2D(gData.dxdy, ixiySum, CV_32F, sumFilter);

		// Calculate the angles and magnitudes from the gradients
		cartToPolar(gData.dx, gData.dy, gData.mag, gData.angle, true);

		Mat crPoints = Mat::zeros(src.rows, src.cols, CV_32F);
		// for each pixel in the image calculate the harris matrix
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				harrisMatrix.at<float>(0, 0) = ix2Sum.at<float>(i, j);
				harrisMatrix.at<float>(0, 1) = ixiySum.at<float>(i, j);
				harrisMatrix.at<float>(1, 0) = ixiySum.at<float>(i, j);
				harrisMatrix.at<float>(1, 1) = iy2Sum.at<float>(i, j);
				// Calculate the corner response function
				float cr = cornerResponse2x2(harrisMatrix);
				// Threshold the response
				if (cr > cornerResponseThreshold) {
					crPoints.at<float>(i, j) = cr;
				}
			}
		}
		// Apply non maxmimum supression
		localNonMaxSupress(crPoints, suppressKSize, keypoints);

	}

void getDominateOrientation(KeypointDescriptor& keydesc, cv::Mat& angles, cv::Mat& mags, cv::Mat& src) {
		using namespace cv;

		Mat rotationInvarianceBins = Mat::zeros(1, ROTATION_INVARIANCE_BINS, CV_32F);
		Mat grid4x4CellAngle = angles(Rect(keydesc.keypoint.pt.x + IMAGE_PAD_SIZE, keydesc.keypoint.pt.y + IMAGE_PAD_SIZE, SIFT_HALF_GRID_SIZE, SIFT_HALF_GRID_SIZE));
		Mat grid4x4CellMag = mags(Rect(keydesc.keypoint.pt.x + IMAGE_PAD_SIZE, keydesc.keypoint.pt.y + IMAGE_PAD_SIZE, SIFT_HALF_GRID_SIZE, SIFT_HALF_GRID_SIZE));


		for (int i = 0; i < grid4x4CellAngle.rows; ++i) {
			for (int j = 0; j < grid4x4CellAngle.cols; ++j) {
				float angle = grid4x4CellAngle.at<float>(i, j);
				float mag = grid4x4CellMag.at<float>(i, j);
				// where cvFloor(angle/ ROTATION_INVARIANCE_ANGLE_PER_BIN) converts the angle to a bin index
				int angleToBinIndex = cvFloor(angle / ROTATION_INVARIANCE_ANGLE_PER_BIN) % ROTATION_INVARIANCE_ANGLE_PER_BIN;
				rotationInvarianceBins.at<float>(0, angleToBinIndex) += mag;
			}

		}

		double min, max;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(rotationInvarianceBins, &min, &max, &minLoc, &maxLoc);
		keydesc.keypoint.angle = maxLoc.x * ROTATION_INVARIANCE_ANGLE_PER_BIN;
		keydesc.keypoint.size = max;
	}

void contrastInvariance(KeypointDescriptor& keydesc) {

		// This is taken from the sift paper to reduce the chances of illumination change
		// Normalize the descriptor so the length = 1
		cv::normalize(keydesc.descriptor, keydesc.descriptor);

		// Clamp values > 0.2 to 0.2
		for (int i = 0; i < keydesc.descriptor.cols; ++i) {


			float val = keydesc.descriptor.at<float>(0, i);
			if (val > DESCRIPTOR_CLAMP_VALUE) {
				keydesc.descriptor.at<float>(0, i) = DESCRIPTOR_CLAMP_VALUE;
			}
		}
		// Renormalize
		cv::normalize(keydesc.descriptor, keydesc.descriptor);

	}

void getSiftDescriptors(cv::Mat& src, std::vector<KeypointDescriptor>& keydesc) {
		using namespace cv;
		Mat srcCopy = src.clone();
		srcCopy.convertTo(srcCopy, CV_32F, 1.0 / 255.0);



		GradientData gradients;
		harrisCornerDetect(srcCopy, keydesc, gradients, CORNER_RESPONSE_THRESHOLD, SUPRESS_KERNEL_SIZE);

		Mat anglePad;
		Mat magPad;
		// TODO remove when done testing
		Mat srcPad;


		// TODO: instead of making copies, see if you can just resuse the gradients.angle/mag mats
		copyMakeBorder(gradients.angle, anglePad, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, BORDER_CONSTANT, Scalar(0, 0, 0));
		copyMakeBorder(gradients.mag, magPad, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, BORDER_CONSTANT, Scalar(0, 0, 0));
		// TODO: remove when done testing
		copyMakeBorder(src, srcPad, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, BORDER_CONSTANT, Scalar(0, 0, 0));


		// TODO: implement scale invariance


	

		for (int i = 0; i < keydesc.size(); ++i) {

			KeypointDescriptor& p = keydesc[i];
			p.descIndex = i;

			getDominateOrientation(p, anglePad, magPad, srcPad);



			// 128 dimensional vector of orientations
			p.descriptor = Mat::zeros(1, SIFT_DESCRIPTOR_SIZE, CV_32F);
			// Since we have a 128 vector, we need to offset the indexes for 0-7
			// example [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, ..]
			// Each range of 0-7 represents a histogram for each 4x4 grid.
			int histogramIndexOffset = 0;




			int s = 2 * SIFT_GRID_SIZE;
			//int t = 
			Point2f pt((s / 2) - 0.5f, (s / 2) - 0.5f);
			Mat r = getRotationMatrix2D(pt, p.keypoint.angle, 1.0);

			Mat subAngles = anglePad(Rect(p.keypoint.pt.x, p.keypoint.pt.y, s, s));
			Mat angleCopy = subAngles.clone();

			Mat subMags = magPad(Rect(p.keypoint.pt.x, p.keypoint.pt.y, s, s));
			Mat magCopy = subMags.clone();

			Mat subSrc = srcPad(Rect(p.keypoint.pt.x, p.keypoint.pt.y, s, s));
			Mat subSrcCopy = subSrc.clone();
			Size st(s, s);

			warpAffine(subMags, subMags, r, st);
			warpAffine(subAngles, subAngles, r, st);
			warpAffine(subSrc, subSrc, r, st);


			// Create the 4x4 sub windows for this point
			for (int xOffset = -SIFT_HALF_GRID_SIZE; xOffset < SIFT_HALF_GRID_SIZE; xOffset += SIFT_QUARTER_GRID_SIZE) {
				for (int yOffset = -SIFT_HALF_GRID_SIZE; yOffset < SIFT_HALF_GRID_SIZE; yOffset += SIFT_QUARTER_GRID_SIZE) {

					// In his inner loop we are creating a new 4x4 grid cell
					// So now we need to increment the index offset
					++histogramIndexOffset;
					Mat grid4x4CellAngle = anglePad(Rect(p.keypoint.pt.x + IMAGE_PAD_SIZE + xOffset, p.keypoint.pt.y + IMAGE_PAD_SIZE + yOffset, SIFT_QUARTER_GRID_SIZE, SIFT_QUARTER_GRID_SIZE));
					Mat grid4x4CellMag = magPad(Rect(p.keypoint.pt.x + IMAGE_PAD_SIZE + xOffset, p.keypoint.pt.y + IMAGE_PAD_SIZE + yOffset, SIFT_QUARTER_GRID_SIZE, SIFT_QUARTER_GRID_SIZE));



					for (int xr = 0; xr < grid4x4CellAngle.rows; ++xr) {
						for (int yr = 0; yr < grid4x4CellAngle.cols; ++yr) {


							float angle = grid4x4CellAngle.at<float>(xr, yr);
							angle = fmod(angle, 360.0f);
							if (angle < 0) angle += 360;
							float mag = grid4x4CellMag.at<float>(xr, yr);


							// where cvFloor(angle/ SIFT_ANGLE_PER_BIN) converts the angle to a bin index
							// TODO: maybe we have to add the angle from the keypoint here keypoint.angle + angle


							int angleToBinIndex = histogramIndexOffset * cvFloor(angle / SIFT_ANGLE_PER_BIN);
							p.descriptor.at<float>(0, angleToBinIndex) += mag;


						}
					}

				}
			}



			angleCopy.copyTo(anglePad(Rect(p.keypoint.pt.x, p.keypoint.pt.y, s, s)));
			magCopy.copyTo(magPad(Rect(p.keypoint.pt.x, p.keypoint.pt.y, s, s)));

			contrastInvariance(p);

		}


	}

inline float squaredSumDiff(KeypointDescriptor& feature1, KeypointDescriptor& feature2) {
		float ssd = 0;
		// This allows the compiler to optimize the loop
		for (int i = 0; i < SIFT_DESCRIPTOR_SIZE; ++i) {
			float diff = feature1.descriptor.at<float>(0, i) - feature2.descriptor.at<float>(0, i);
			ssd += diff * diff;
			//ssd += cv::abs(feature1.descriptor.at<float>(0, i) - feature2.descriptor.at<float>(0, i));
		}
		return ssd;
	}

DesriptorMatch GetMatches(KeypointDescriptor& srcDescriptor, std::vector<KeypointDescriptor>& dstDescriptors) {
	DesriptorMatch match;
	match.srcDesc = &srcDescriptor;
	match.bestSsd = FLT_MAX;
	match.bestSsdIndex = -1;
	match.secondBestSsd = FLT_MAX;
	match.secondBestSsdIndex = -1;
	for (int i = 0; i < dstDescriptors.size(); ++i) {
		float ssdTest = squaredSumDiff(srcDescriptor, dstDescriptors[i]);
		if (ssdTest <= match.bestSsd) {
			match.secondBestSsd = match.bestSsd;
			match.secondBestSsdIndex = match.bestSsdIndex;
			match.bestSsd = ssdTest;
			match.bestSsdIndex = i;
		}
	}

	if (match.secondBestSsdIndex == -1) match.secondBestSsdIndex = match.bestSsdIndex;

	// Ratio test
	match.ratio = match.bestSsd / match.secondBestSsd;

	// discard match if greater than our threshold

	match.useMatch = match.ratio < RATIO_SSD_THRESHOLD;
	return match;
}

void extractMatches(std::vector<KeypointDescriptor>& srcDescriptors, std::vector<KeypointDescriptor>& dstDescriptors, std::vector<DesriptorMatch>& matches) {

		for (int i = 0; i < srcDescriptors.size(); ++i) {
			int pt = i;
			DesriptorMatch match = GetMatches(srcDescriptors[pt], dstDescriptors);
			
			if (match.useMatch) {
				match.srcIndex = srcDescriptors[pt].descIndex;
				// While this statement is useless and redundant, it what a match index is, and makes things more clear to understand
				match.matchIndex = match.bestSsdIndex;
				match.dstDesc = &dstDescriptors[match.matchIndex];
				matches.push_back(match);

			}
		}
	}

// Faster to copy a pt, then pass by ref
// Faster to return a copy of pt then pass by ref
cv::Point2f project(cv::Point2f pt, cv::Mat& homography) {
	// Build a 3x1 vec <x1,y1,1>
	cv::Mat vec = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1);

	// Multiply/Project our homography onto our vec
	cv::Mat result = homography * vec;
	// Get w
	float w = result.at<double>(2, 0);
	
	// Point2f(x/w, y/w)
	return cv::Point2f(result.at<double>(0, 0) / w, result.at<double>(1, 0) / w);
}


// TODO: figure out how to speed this up, either simd or better block processing
float inline ptDist(cv::Point2f pt1, cv::Point2f pt2) {
	return cv::sqrt(((pt2.x - pt1.x) * (pt2.x - pt1.x)) + ((pt2.y - pt1.y) * (pt2.y - pt1.y)));
}


uint32 computeInlierCount(cv::Mat& homorgraphy, std::vector<DesriptorMatch>& matches, float inlierThreshold) {
	// Random reserve size. This should probably be tested multiple times to see what gives the fastest results

	uint32 count = 0;
	for (int i = 0; i < matches.size(); ++i) {
		cv::Point2f srcPt = matches[i].srcDesc->keypoint.pt;
		cv::Point2f projectedPt = project(srcPt, homorgraphy);
		cv::Point2f dstPt = matches[i].dstDesc->keypoint.pt;


		float distance = ptDist(projectedPt, dstPt);

		if (distance < inlierThreshold) {
			++count;
		}
	}
	return count;
}


InlierSet computeInlierSet(cv::Mat& homorgraphy, std::vector<DesriptorMatch>& matches, float inlierThreshold) {
	// Random reserve size. This should probably be tested multiple times to see what gives the fastest results
	
	std::vector<cv::Point2f> inlierSrc;
	std::vector<cv::Point2f> inlierDst;


	
	for (int i = 0; i < matches.size(); ++i) {
		cv::Point2f srcPt = matches[i].srcDesc->keypoint.pt;
		cv::Point2f projectedPt = project(srcPt, homorgraphy);
		cv::Point2f dstPt = matches[i].dstDesc->keypoint.pt;

		
		float distance = ptDist(projectedPt, dstPt);

		if (distance < inlierThreshold) {
			inlierSrc.push_back(srcPt);
			inlierDst.push_back(dstPt);
		}
	}

	assert(inlierSrc.size() == inlierDst.size());
	assert(inlierSrc.size() < matches.size());

	return{
		inlierSrc,
		inlierDst,
		inlierSrc.size() // inlierSrc and inlierDst are the same size
	};
}


void RANSAC_(
	std::vector<DesriptorMatch>& matches,
	uint32 numIterations,
	float inlierThreshold,
	cv::Mat& hom,
	cv::Mat& homInv,
	cv::Mat& img1,
	cv::Mat& img2) {
	

	std::vector<cv::Point2f> src(4);
	std::vector<cv::Point2f> dst(4);

	
	cv::Mat bestHomography;
	size_t bestInlierCount = 0;
	

	cv::RNG rng = cv::RNG();

	

	for (uint32 i = 0; i < numIterations; ++i) {
		size_t matchSize = matches.size();
		auto matchCount = static_cast<int>(matchSize);
		// Randomly select 4 pairs
		auto m1 = matches[rng.uniform(0, matchCount)];;
		auto m2 = matches[rng.uniform(0, matchCount)];;
		auto m3 = matches[rng.uniform(0, matchCount)];;
		auto m4 = matches[rng.uniform(0, matchCount)];;
		
		// Split the src and dst points
		src[0] = m1.srcDesc->keypoint.pt;
		src[1] = m2.srcDesc->keypoint.pt;
		src[2] = m3.srcDesc->keypoint.pt;
		src[3] = m4.srcDesc->keypoint.pt;

		dst[0] = m1.dstDesc->keypoint.pt;
		dst[1] = m2.dstDesc->keypoint.pt;
		dst[2] = m3.dstDesc->keypoint.pt;
		dst[3] = m4.dstDesc->keypoint.pt;
		
		
		// Compute the homography using the four selected matches
		// Note: RANSAC is not enabled, third input is 0
		cv::Mat h = cv::findHomography(src, dst, 0);

		uint32 inlierCount = computeInlierCount(h, matches, inlierThreshold);

		if (inlierCount > bestInlierCount) {
			bestInlierCount = inlierCount;
			bestHomography = h;
		}
	}


	InlierSet inlierSet = computeInlierSet(bestHomography, matches, inlierThreshold);
	hom = cv::findHomography(inlierSet.src, inlierSet.dst, 0);
	homInv = hom.inv();
	

	cv::Mat inlierImg;
	hconcat(img1, img2, inlierImg);

	cv::Point2f concatVector = cv::Point2f(img1.cols, 0);
	for (int i = 0 ; i < inlierSet.count; ++i) {
		auto p1 = inlierSet.src[i];
		auto p2 = inlierSet.dst[i] + concatVector;
		
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		circle(inlierImg, p1, 2, color, 2);
		circle(inlierImg, p2, 2, color, 2);
		line(inlierImg, p1, p2, color, 0.5);
	}

	cv::imshow("inlier_img", inlierImg);
	cv::imwrite("3.png", inlierImg);
	
}

bool inline inImageRange(cv::Point2f p, cv::Mat& img) {
	return p.x > 0 && p.x < img.cols && p.y > 0 && p.y < img.rows;
}


void stich(cv::Mat& img1, cv::Mat& img2, cv::Mat& hom, cv::Mat& homInv, cv::Mat& outputStich) {



	std::vector<cv::Point2f> corners;
	// Top left
	corners.push_back(project(cv::Point2f(0, 0), homInv));
	// Top right
	corners.push_back(project(cv::Point2f(img2.cols, 0), homInv));
	// Bottom Left
	corners.push_back(project(cv::Point2f(0, img2.rows), homInv));
	// Bottom right
	corners.push_back(project(cv::Point2f(img2.cols, img2.rows), homInv));	
	
	int colMin, colMax, rowMin, rowMax;
	

	colMin = 0;
	rowMin = 0;
	colMax = img1.cols;
	rowMax = img1.rows;

	
	// Find min/max of stiched image
	for (int i = 0; i < corners.size(); ++i) {
		auto pt = corners[i];

		int row = pt.y;
		int col = pt.x;

		if (colMin > col) { colMin = col; }
		if (colMax < col) { colMax = col; }
		if (rowMin > row) { rowMin = row; }
		if (rowMax < row) { rowMax = row; }
			
	}

	int stichCols = colMax - colMin;
	int stichRows = rowMax - rowMin;
	outputStich = cv::Mat::zeros(stichRows, stichCols, img1.type());
	img1.copyTo(outputStich(cv::Rect(0 - colMin, 0 - rowMin, img1.cols, img1.rows)));
	

	cv::Size size(1, 1);
	cv::Mat patch(size, img2.type());


	

	for (int row = 0; row < outputStich.rows; ++row) {
		for (int col = 0; col < outputStich.cols; ++col) {

			
			// Forward warp image
			auto p = project(cv::Point2f(col + colMin, row + rowMin), hom);
			if (inImageRange(p, img2)) {
				// Bilinear Interpolate
				cv::getRectSubPix(img2, size, cv::Point2f(p.x, p.y), patch);
				uchar px[3] = { outputStich.at<cv::Vec3b>(row, col).val[0], outputStich.at<cv::Vec3b>(row, col).val[1], outputStich.at<cv::Vec3b>(row, col).val[2] };
				if (px[0] == 0 && px[1] == 0 && px[2] == 0) {
					// Set rgb values of image
					outputStich.at<cv::Vec3b>(row, col).val[0] = patch.at<uchar>(0, 0);
					outputStich.at<cv::Vec3b>(row, col).val[1] = patch.at<uchar>(0, 1);
					outputStich.at<cv::Vec3b>(row, col).val[2] = patch.at<uchar>(0, 2);
				} else {
					// Blend them using a simple average
					outputStich.at<cv::Vec3b>(row, col).val[0] = (patch.at<uchar>(0, 0) + px[0]) * 0.5f;
					outputStich.at<cv::Vec3b>(row, col).val[1] = (patch.at<uchar>(0, 1) + px[1]) * 0.5f;
					outputStich.at<cv::Vec3b>(row, col).val[2] = (patch.at<uchar>(0, 2) + px[2]) * 0.5f;
				}

			}
		}
	}

}


void drawKeypointDescs(cv::Mat&dst, std::vector<KeypointDescriptor>& interestPoints) {
	using namespace cv;
	Scalar color = Scalar(1, 1, 0);
	for (int i = 0; i < interestPoints.size(); ++i) {
		circle(dst, Point(interestPoints[i].keypoint.pt.x, interestPoints[i].keypoint.pt.y), interestPoints[i].keypoint.size * 0.15, color, 2, LINE_4);

	
	}
}




int main(int argc, char** argv)
{
	using namespace cv;


	// NOTE: you can swap these files around to check the stiching for different image pairs
	
	//std::string files[] = {"imgs/yosemite/Yosemite1.jpg", "imgs/yosemite/Yosemite2.jpg"};
	//std::string files[] = { "imgs/graf/img1.ppm", "imgs/graf/img2.ppm" };
	std::string files[] = { "imgs/project_images/Rainier1.png", "imgs/project_images/Rainier2.png" };
	//std::string files[] = { "imgs/project_images/MelakwaLake3.png", "imgs/project_images/MelakwaLake4.png" };

	Mat input1 = imread(files[0], CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread(files[1], CV_LOAD_IMAGE_COLOR);

	
	Mat output;

	std::vector<KeypointDescriptor> keypoints1;
	std::vector<KeypointDescriptor> keypoints2;
	getSiftDescriptors(input1, keypoints1);
	getSiftDescriptors(input2, keypoints2);

	std::vector<DesriptorMatch> matches;
	extractMatches(keypoints1, keypoints2, matches);

	
	cv::Mat hom;
	cv::Mat homInv;
	cv::Mat stichedImage;

	cv::Mat img1 = input1.clone();
	cv::Mat img2 = input2.clone();


	uint32 iterations = 1000;
	float inlierThreshold = 2;

	RANSAC_(matches, iterations, inlierThreshold, hom, homInv, img1, img2);
	stich(input1, input2, hom, homInv, stichedImage);
	imshow("Stich", stichedImage);
	imwrite("4.png", stichedImage);

	

	waitKey();
    return 0;
}


