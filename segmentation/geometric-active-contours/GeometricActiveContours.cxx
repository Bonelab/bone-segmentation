
#include <iostream>

#include "itkShapeDetectionLevelSetImageFilter.h"

#include "ImageUtils.hpp"
#include "FilterUtils.hpp"
#include "Globals.hpp"
#include "ChamferDistanceTransform.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkGeodesicActiveContourLevelSetImageFilter.h"
#include "itkSigmoidImageFilter.h"

using namespace std;

template<class ImageType>
typename ImageType::Pointer
joinDistances(
    typename ImageType::Pointer img1, typename ImageType::Pointer img2
){

    itk::ImageRegionIteratorWithIndex<ImageType> it(
        img1, img1->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (it.Get()==0)
            it.Set(-img2->GetPixel(it.GetIndex()));
    }

    return img1;
}

template<class ImageType>
typename ImageType::Pointer chamferDistance(typename ImageType::Pointer image) {

    typedef ChamferDistanceTransform<ImageType, ImageType> CDT;
    CDT cdt;
    return cdt.compute(image, CDT::MANHATTEN, false);
}


template<class ImageType>
typename ImageType::Pointer getInitialLevelSet(typename ImageType::Pointer inputCT) {

    IntImagePtr fat =
        FilterUtils<IntImage>::binaryThresholding(
            FilterUtils<IntImage>::relabelComponents(
                FilterUtils<IntImage>::connectedComponents(
                FilterUtils<ImageType,IntImage>::binaryThresholding(
                    inputCT, -5000, -50)
                )),
            1,1,0,1
        );

    typename ImageType::Pointer distancesInside =
        chamferDistance<ImageType>(FilterUtils<IntImage,ImageType>::cast(fat));

    typename ImageType::Pointer distancesOutside =
        chamferDistance<ImageType>(FilterUtils<IntImage,ImageType>::cast(
            FilterUtils<IntImage>::binaryThresholding(fat, 1,1,0,1)));

    return joinDistances<ImageType>(distancesInside, distancesOutside);


}



int main(int argc, char * argv [])
{
    boost::timer t;

	if (argc != 3) {
	    cerr << "Usage: " << argv[0] << " input-CT-image output-image \n";
	    return EXIT_FAILURE;
	}

    string inputImage = argv[1];
    string outputImage = argv[2];
    float gradientSigma = 1.0;

    float propagationScaling = -5.0;
    float curvatureScaling = 1.0;
    float advectionScaling = 0;

    float maximumRMSError = 0.001;
    unsigned numberOfIterations = 400;

    // read the input image
    FloatImagePtr inputCT = ImageUtils<FloatImage>::readImage(inputImage);

    // initialize the level set
#if 1
    IntImagePtr levelSet = getInitialLevelSet<IntImage>(
        FilterUtils<FloatImage, IntImage>::cast(inputCT));
//    ImageUtils<IntImage>::writeImage(outputImage+"-level-set.nii", levelSet);
#else
    IntImagePtr levelSet = ImageUtils<IntImage>::readImage(outputImage+"-level-set.nii");
#endif

//    printf("Sigma=%.2f, adv=%.2f, prop=%.2f, curv=%.2f, rms=%.2f, it=%d\n",
//        gradientSigma, advectionScaling, propagationScaling, curvatureScaling,
//        maximumRMSError, numberOfIterations);

    // compute the feature image as sigmoid of gaussian
    typedef itk::GradientMagnitudeRecursiveGaussianImageFilter<
        FloatImage, FloatImage > GradientFilterType;
    GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
    gradientFilter->SetSigma(gradientSigma);
    gradientFilter->SetInput(inputCT);
    gradientFilter->Update();
    FloatImagePtr gradientImg = gradientFilter->GetOutput();
//    ImageUtils<FloatImage>::writeImage(outputImage+"-gradient.nii", gradientImg);


    // transform gradient into a feature image
    itk::ImageRegionIterator<FloatImage> it(gradientImg, gradientImg->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float g = it.Get();

        float gNew = exp( -g / 35);

        it.Set(gNew);
    }

    FloatImagePtr featureImage = gradientImg;
 //   ImageUtils<FloatImage>::writeImage(outputImage+"-feature.nii", featureImage);

    // segment
    typedef itk::GeodesicActiveContourLevelSetImageFilter<FloatImage,FloatImage> GeodesicAC;
    GeodesicAC::Pointer geodesicAC = GeodesicAC::New();

    geodesicAC->SetInput(FilterUtils<IntImage, FloatImage>::cast(levelSet));
    geodesicAC->SetFeatureImage(featureImage);
    geodesicAC->SetPropagationScaling( propagationScaling );
    geodesicAC->SetCurvatureScaling( curvatureScaling );
    geodesicAC->SetAdvectionScaling( advectionScaling );
    geodesicAC->SetMaximumRMSError( maximumRMSError );
    geodesicAC->SetNumberOfIterations( numberOfIterations );

    geodesicAC->Update();
    FloatImagePtr result = geodesicAC->GetOutput();

    ImageUtils<UCharImage>::writeImage(
        outputImage,
        FilterUtils<FloatImage,UCharImage>::binaryThresholding(result, 0, 10000, 0,1)
    );

//    cout << "Elapsed iterations: " << geodesicAC->GetElapsedIterations() << endl;
//    cout << "RMS change: " << geodesicAC->GetRMSChange() << endl;
    cout << boost::format("%1%,%2%\n") % t.elapsed() % AVAILABLE_MEMORY_IN_MB;


	return EXIT_SUCCESS;
}
