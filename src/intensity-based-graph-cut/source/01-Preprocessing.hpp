/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/





#include "Globals.hpp"
#include "ImageUtils.hpp"
#include "FilterUtils.hpp"
#include "SheetnessMeasure.hpp"
#include "ChamferDistanceTransform.hpp"
#include "boost/tuple/tuple.hpp"


namespace Preprocessing {

using namespace boost;


// compute multiscale sheetness measure
// if roi is specified, than compute the measure only for pixels within ROI
// (i.e. pixels where roi(pixel) >= 1)
FloatImagePtr multiscaleSheetness(
    FloatImagePtr img, vector<float> scales, UCharImagePtr roi = 0
) {

    assert(scales.size() >= 1);

    FloatImagePtr multiscaleSheetness;

    for (unsigned i = 0; i < scales.size(); ++i) {

        log("Computing single-scale sheetness, sigma=%4.2f") % scales[i];

        MemoryEfficientObjectnessFilter *sheetnessFilter =
            new MemoryEfficientObjectnessFilter();
        sheetnessFilter->SetImage(img);
        sheetnessFilter->SetAlpha(0.5);
        sheetnessFilter->SetBeta(0.5);
        sheetnessFilter->SetSigma(scales[i]);
        sheetnessFilter->SetObjectDimension(2);
        sheetnessFilter->SetBrightObject(true);
        sheetnessFilter->ScaleObjectnessMeasureOff();
        sheetnessFilter->Update();
        sheetnessFilter->SetROIImage(roi);

        FloatImagePtr singleScaleSheetness = sheetnessFilter->GetOutput();

        if (i==0) {
            multiscaleSheetness = singleScaleSheetness;
            continue;
        }

        // update the multiscale sheetness
        // take the value which is larger in absolute value
        itk::ImageRegionIterator<FloatImage>
            itMulti(singleScaleSheetness,singleScaleSheetness->GetLargestPossibleRegion());
        itk::ImageRegionIterator<FloatImage>
            itSingle(multiscaleSheetness,multiscaleSheetness->GetLargestPossibleRegion());
        for (
                itMulti.GoToBegin(),itSingle.GoToBegin();
                !itMulti.IsAtEnd();
                ++itMulti, ++itSingle
            ) {
                float multiVal = itMulti.Get();
                float singleVal = itSingle.Get();

                // higher absolute value is better
                if (abs(singleVal) > abs(multiVal)) {
                    itMulti.Set(singleVal);
                }
            }
    } // iteration trough scales

    return multiscaleSheetness;

}



FloatImagePtr chamferDistance(UCharImagePtr image) {
    typedef ChamferDistanceTransform<UCharImage, FloatImage> CDT;
    CDT cdt;
    return cdt.compute(image, CDT::MANHATTEN);
}



/*
Input: Normalized CT image, scales for the sheetness measure
Output: (ROI, MultiScaleSheetness, SoftTissueEstimation)
*/
tuple<UCharImagePtr,  UCharImagePtr>
compute(
    ShortImagePtr inputCT,
    float sigmaSmallScale,
    vector<float> sigmasLargeScale
) {

    UCharImagePtr roi;
    UCharImagePtr softTissueEstimation;

    {

        log("Estimating soft-tissue voxels");
        softTissueEstimation =  FilterUtils<UIntImage,UCharImage>::binaryThresholding(
                FilterUtils<UIntImage>::relabelComponents(
                    FilterUtils<UCharImage, UIntImage>::connectedComponents(
                    FilterUtils<ShortImage,UCharImage>::binaryThresholding(
                        inputCT, -5000, -50)
                    )),
                1,1
            );


        log("Estimating bone voxels");
        UCharImagePtr boneEstimation =
            FilterUtils<ShortImage,UCharImage>::createEmptyFrom(inputCT);
        itk::ImageRegionIteratorWithIndex<ShortImage>
            it(inputCT,inputCT->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            short hu = it.Get();

            bool bone = (hu > 200);

            boneEstimation->SetPixel(it.GetIndex(), bone ? 1 : 0);
        }

        log("Computing ROI from bone estimation using Chamfer Distance");
        roi = FilterUtils<FloatImage,UCharImage>::binaryThresholding(
            chamferDistance(boneEstimation),0, 30);
    }

    return make_tuple(roi, softTissueEstimation);
}





} // namespace Preprocessing
