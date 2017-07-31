
#include <iostream>


#include "Globals.hpp"
#include "ImageUtils.hpp"
#include "FilterUtils.hpp"

#include "itkImageRegionIteratorWithIndex.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

#include "boost/tuple/tuple.hpp"
#include "boost/lexical_cast.hpp"

#include <vector>

using namespace std;
//////////////////////////////////////////
// PARAMETERS

const short INITIAL_THRESHOLD = 70;
const unsigned int WINDOW_SPAN_X = 7;
const unsigned int WINDOW_SPAN_Y = 7;
const unsigned int WINDOW_SPAN_Z = 1;


//////////////////////////////////////////
// CODE
#include <iostream>
#include <fstream>

using namespace std;

namespace SystemUtils {


    long getValueFromProcFile(const string & procFile, const string & pattern) {

        ifstream statusFile;
        statusFile.open (procFile.c_str());

        while(!statusFile.eof()) // To get you all the lines.
        {
            string token;
            statusFile >> token;

            if (token.compare(pattern) == 0) {
                long valueKb;
                statusFile >> valueKb;
                return valueKb;
            }
        }

        return -1;
    }



    long maxMemoryUsageInKb() {
        return getValueFromProcFile("/proc/self/status","VmHWM:");
    }

    long freeMemoryInKb() {
        return getValueFromProcFile("/proc/meminfo","MemFree:") +
            getValueFromProcFile("/proc/meminfo","Cached:");
    }



} //namespace






void addOffset(vector<ImageOffset> &offsets, int x, int y, int z) {
    ImageOffset offset;
    offset[0] = x;
    offset[1] = y;
    offset[2] = z;
    offsets.push_back(offset);
}


ShortImagePtr computeBoundaryVoxels(ShortImagePtr B) {

    ShortImagePtr boundaryVoxels = FilterUtils<ShortImage>::createEmptyFrom(B);
    itk::ImageRegionIteratorWithIndex<ShortImage>it(B, B->GetLargestPossibleRegion());
    ImageSize imageSize = B->GetLargestPossibleRegion().GetSize();

    //prepare the neighborhood
    vector<ImageOffset> offsets;
    addOffset(offsets, -1, -1,  0);
    addOffset(offsets, -1,  0,  0);
    addOffset(offsets, -1, +1,  0);
    addOffset(offsets,  0, -1,  0);
    addOffset(offsets,  0, +1,  0);
    addOffset(offsets, +1, -1,  0);
    addOffset(offsets, +1,  0,  0);
    addOffset(offsets, +1, +1,  0);
    addOffset(offsets,  0,  0, -1);
    addOffset(offsets,  0,  0, +1);

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

        ImageIndex cIndex = it.GetIndex();
        bool isCenterBone = (it.Get() == 1);

        // non-bone => not a boundary pixel
        if (!isCenterBone) {
            boundaryVoxels->SetPixel(cIndex, 0);
            continue;
        }

        // explore the 10-neighborhood of the central pixel
        for (unsigned i=0; i < offsets.size(); ++i) {

            ImageIndex nIndex = cIndex + offsets[i];
            if(!ImageUtils<ShortImage>::isInside(B,nIndex))
                continue;

            bool isNeighBone = (B->GetPixel(nIndex) == 1);

            // non-bone neigh => a boundary pixel detected
            if (!isNeighBone) {
                boundaryVoxels->SetPixel(cIndex, 1);
                goto endOfImageLoop;
            }
        }

        // no non-bone voxel detected => not a boundary pixel
        boundaryVoxels->SetPixel(cIndex, 0);

endOfImageLoop:
        ;
    } // iterating through the image


    return boundaryVoxels;
}



float dNorm(float mu, float sigma2, float x) {
    return exp(-pow(x - mu,2)/(2*sigma2)) / sqrt(2*3.14159265358797*sigma2);
}




boost::tuple<ShortImagePtr, unsigned> identifyMisclassifiedVoxels(
    ShortImagePtr inputImg, ShortImagePtr B, ShortImagePtr Eb
){

    ShortImagePtr errorImg = FilterUtils<ShortImage>::createEmptyFrom(B);

    typedef itk::NeighborhoodIterator<ShortImage> NeighborhoodIteratorType;
    typedef itk::ImageRegionIteratorWithIndex<ShortImage> ImageRegionIterator;

    NeighborhoodIteratorType::RadiusType radius;
    radius[0] = WINDOW_SPAN_X;
    radius[1] = WINDOW_SPAN_Y;
    radius[2] = WINDOW_SPAN_Z;
    int size = (2*radius[0]+1) * (2*radius[1]+1)* (2*radius[2]+1);

    ImageRegionIterator itBorderPixels(Eb, Eb->GetLargestPossibleRegion());
    ImageRegionIterator itError(errorImg, errorImg->GetLargestPossibleRegion());
    NeighborhoodIteratorType itIntensity(radius, inputImg, inputImg->GetLargestPossibleRegion());
    NeighborhoodIteratorType itBone(radius, B, B->GetLargestPossibleRegion());

    ImageSize imageSize = Eb->GetLargestPossibleRegion().GetSize();
    unsigned totalPixels = imageSize[0] * imageSize[1] * imageSize[2];

    // iterate over the input image
    unsigned reclasiffied = 0;

//    unsigned counter = 0;
    for (
        itBorderPixels.GoToBegin(), itIntensity.GoToBegin(), itError.GoToBegin(), itBone.GoToBegin();
        !itBorderPixels.IsAtEnd();
        ++itBorderPixels, ++itIntensity, ++itError, ++itBone
    ) {
//        counter ++;
//        if (counter % 1000 == 0)
//            fprintf(stdout, "\r%2.3f%%   ", 100 * counter / (1.0 * totalPixels) );

        // skip non-border pixels
        if (itBorderPixels.Get() == 0 )
            continue;

        assert(itBone.GetCenterPixel() == 1);

        // compute means and variances for bone and non-bone classes
        float meanBone = 0;
        float meanNonBone = 0;
        float varBone = 0;
        float varNonBone = 0;

        int Nb = 0, Nnb = 0;
        for (int idx = 0; idx < size; ++idx) {
            bool isBone = (itBone.GetPixel(idx) == 1);
            short i = itIntensity.GetPixel(idx);
            if (isBone) {
                meanBone += i;
                varBone += i*i;
                Nb++;
            } else {
                meanNonBone += i;
                varNonBone += i*i;
                Nnb++;
            }
        }
        assert(Nb + Nnb == size);
        assert(Nb > 0);
        assert(Nnb > 0);

        meanBone /= Nb;
        meanNonBone /= Nnb;
        varBone = (Nb / (Nb - 1.0)) * ((varBone / Nb) - meanBone*meanBone);
        varNonBone = (Nnb / (Nnb - 1.0)) * ((varNonBone / Nnb) - meanNonBone*meanNonBone);

        short i = itIntensity.GetCenterPixel();
        float pBone    = dNorm(meanBone, varBone, i) * ( Nb / (1.0 *size));
        float pNonBone = dNorm(meanNonBone, varNonBone, i) * ( Nnb / (1.0 *size));

//        log("(%7.1f,%7.1f)  (%7.1f,%7.1f)  %5.10f %5.10f")
//            % meanBone % varBone % meanNonBone % varNonBone % pBone % pNonBone;

        if (pNonBone > pBone) {
            itError.Set(1);
            reclasiffied++;
        }
    }


    return boost::make_tuple(errorImg, reclasiffied);

}






int main(int argc, char * argv [])
{

    boost::timer t;


	if (argc != 3) {
	    cerr << "Usage: " << argv[0] << " input-CT-image output-binary-image\n";
	    return EXIT_FAILURE;
	}


    ShortImagePtr inputImg = ImageUtils<ShortImage>::readImage(argv[1]);

    logSetStage("Initialization");

    log("Initial thresholding [T=%d]") % INITIAL_THRESHOLD;
    ShortImagePtr B =
        FilterUtils<ShortImage>::binaryThresholding(
            FilterUtils<ShortImage>::relabelComponents(
                FilterUtils<ShortImage>::connectedComponents(
                FilterUtils<ShortImage>::binaryThresholding(
                    inputImg, -5000, INITIAL_THRESHOLD)
                )),
            1,1,0,1
        );

//    string filename = string(argv[2]) + "0.nii";
//    log("Saving results of this step to %s") % filename;
//    ImageUtils<ShortImage>::writeImage(filename, B);


    unsigned iteration = 1;
    bool converged = false;

    while (!converged) {
        logSetStage("Iteration#" + boost::lexical_cast<string>(iteration));

        log("Computing boundary voxels");
        ShortImagePtr Eb = computeBoundaryVoxels(B);

        log("Identifying mis-classified boundary voxels");
        ShortImagePtr errorImg;
        unsigned totalErrorVoxels;
        boost::tie(errorImg, totalErrorVoxels) =
            identifyMisclassifiedVoxels(inputImg, B, Eb);


        log("%d voxels in the error class, updating the segmentation") % totalErrorVoxels;
        B = FilterUtils<ShortImage>::negatedMask(B, errorImg);


        converged = (totalErrorVoxels < 500);
        iteration++;
    }

    string filename = argv[2];// + boost::lexical_cast<string>(iteration) + ".nii";
    log("Saving results of this step to %s") % filename;
    ImageUtils<ShortImage>::writeImage(filename, B);

    cout << boost::format("%1%,%2%\n") % t.elapsed() % SystemUtils::maxMemoryUsageInKb();

    log("VERTIG :)");

	return EXIT_SUCCESS;


}
