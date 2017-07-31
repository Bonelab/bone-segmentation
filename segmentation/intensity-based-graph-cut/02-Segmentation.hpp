/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/






#pragma once

#include "GraphCut.hpp"
#include "ImageUtils.hpp"
#include "Globals.hpp"


namespace Segmentation {



typedef GraphCutSegmentation<Dimension> GCSegm;
typedef GCSegm::LabelID Label;


const Label BONE   = 1;
const Label TISSUE = 0;
const int COST_AMPLIFIER = 1000;




//==============================================================================
//    Cost Functions
//==============================================================================


class SheetnessBasedDataCost : public GCSegm::DataCostFunction {
private:

    ShortImagePtr intensity;
    UCharImagePtr softTissueEstimation;

public:

    // constructor
    SheetnessBasedDataCost(
        ShortImagePtr p_intensity,
        UCharImagePtr p_softTissueEstimation
    )
    : intensity(p_intensity)
    , softTissueEstimation(p_softTissueEstimation)
    { /* empty body */}

    virtual int compute(ImageIndex idx, Label label) {


        short hu = intensity->GetPixel(idx);
        unsigned char t = softTissueEstimation->GetPixel(idx);

        assert( t==0 || t==1);

        float totalCost;
        switch (label) {
            case BONE  :
                totalCost = (t==1) ? 1 : 0;
                break;

            case TISSUE:
                  totalCost = ( hu > 400) ? 1 : 0;
                break;

            default:
                assert(false);
        }

        assert(totalCost >= 0 || totalCost < 1.001);

        return COST_AMPLIFIER * totalCost;


    }

};



class SheetnessBasedSmoothCost : public GCSegm::SmoothnessCostFunction {
private:

    ShortImagePtr intensity;

public:

    SheetnessBasedSmoothCost(
        ShortImagePtr p_intensity
    )
    : intensity(p_intensity)
    { /* empty body */}


    virtual int compute(
        ImageIndex idx1, ImageIndex idx2
    ) {

        float hu1 = intensity->GetPixel(idx1);
        float hu2 = intensity->GetPixel(idx2);
        float dHU   = abs(hu1 - hu2);

        float cost = exp ( - dHU / 100);

        float alpha = 1.0;
        return COST_AMPLIFIER * alpha *  cost  + 1;

    }
};




//==============================================================================
//    Segmentation
//==============================================================================




UCharImagePtr compute(
    ShortImagePtr intensity,
    UCharImagePtr roi,
    UCharImagePtr softTissueEstimation
) {


    // data cost functions
    SheetnessBasedDataCost dataCostFunction(
        intensity, softTissueEstimation);
    SheetnessBasedSmoothCost smoothCostFunction(
        intensity);

    // graph-cut segmentation
    GCSegm gcSegm;
    UIntImagePtr gcOutput = gcSegm.optimize(
        FilterUtils<UCharImage,UIntImage>::cast(roi),
        &dataCostFunction, &smoothCostFunction
    );

    // finitto :)
    return FilterUtils<UIntImage,UCharImage>::cast(gcOutput);

}






}
