
#pragma once

#include "GraphCut.hpp"
#include "ImageUtils.hpp"
#include "Globals.hpp"

namespace AnnotatedGC {
    typedef GraphCutSegmentation<Dimension> GCSegm;
    typedef GCSegm::LabelID Label;

    const Label FOREGROUND   = 1;
    const Label BACKGROUND   = 0;
    const int COST_AMPLIFIER = 1000;

    class AnnotatedSheetnessDataCost : public GCSegm::DataCostFunction {
    private:
        ShortImagePtr intensity;
        FloatImagePtr sheetness;
        UCharImagePtr backgroundEstimation;
        UCharImagePtr foregroundEstimation;
    public:
        // constructor
        AnnotatedSheetnessDataCost(
            ShortImagePtr p_intensity,
            FloatImagePtr p_sheetnessMeasure,
            UCharImagePtr p_backgroundEstimation,
            UCharImagePtr p_foregroundEstimation
        )
        : intensity(p_intensity)
        , sheetness(p_sheetnessMeasure)
        , backgroundEstimation(p_backgroundEstimation)
        , foregroundEstimation(p_foregroundEstimation)
        { /* empty body */}

        virtual int compute(ImageIndex idx, Label label) {
            short hu = intensity->GetPixel(idx);
            float s = sheetness->GetPixel(idx);
            unsigned char back = backgroundEstimation->GetPixel(idx);
            unsigned char fore = foregroundEstimation->GetPixel(idx);

            assert( back==0 || back==1);
            assert( fore==0 || fore==1);
            assert( s > -1.001 && s < 1.001);

            float totalCost;
            switch (label) {
                case FOREGROUND:
                    totalCost = (back == 1) ? 1 : 0;
                    break;

                case BACKGROUND:
                    totalCost = (fore == 1) ? 1 : 0;
                    break;

                default:
                    assert(false);
            }

            assert(totalCost >= 0 || totalCost < 1.001);
            return COST_AMPLIFIER * totalCost;
        }

    };



    class AnnotatedSheetnessSmoothCost : public GCSegm::SmoothnessCostFunction {
    private:
        ShortImagePtr intensity;
        FloatImagePtr sheetness;
    public:
        AnnotatedSheetnessSmoothCost(
            ShortImagePtr p_intensity,
            FloatImagePtr p_sheetnessMeasure
        )
        : intensity(p_intensity)
        , sheetness(p_sheetnessMeasure)
        { /* empty body */}


        virtual int compute(
            ImageIndex idx1, ImageIndex idx2
        ) {
    //        float hu1 = intensity->GetPixel(idx1);
    //        float hu2 = intensity->GetPixel(idx2);
    //        float dHU   = abs(hu1 - hu2);

            float s1 = sheetness->GetPixel(idx1);
            float s2 = sheetness->GetPixel(idx2);
            float dSheet = abs(s1 - s2);

            float sheetnessCost = (s1 < s2) ? 1.0 : exp ( - 5 * dSheet);

            float alpha = 5.0;
            return COST_AMPLIFIER * alpha *  sheetnessCost  + 1;
        }
    };

    UCharImagePtr compute(
        ShortImagePtr intensity,
        FloatImagePtr sheetnessMeasure,
        UCharImagePtr backgroundEstimation,
        UCharImagePtr foregroundEstimation,
        UCharImagePtr roi
    ) {
        // data cost functions
        AnnotatedSheetnessDataCost dataCostFunction(intensity, sheetnessMeasure, backgroundEstimation, foregroundEstimation);
        AnnotatedSheetnessSmoothCost smoothCostFunction(intensity,sheetnessMeasure);

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
