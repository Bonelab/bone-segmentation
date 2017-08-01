#include "Globals.hpp"
#include "ImageUtils.hpp"
#include "FilterUtils.hpp"
#include <string>
#include <vector>
#include "boost/lexical_cast.hpp"
#include "SheetnessMeasure.hpp"
#include "annotated-gc.hxx"

#define DEBUG (true)

/**
Class for unified and platform-independent
access to filenames used in the program.
*/
struct FilenameDb {

    std::string m_tempDir;
    std::string m_inputFilename;
    std::string m_annotatedFilename;
    std::string m_outputFilename;

    FilenameDb(std::string input, std::string annotated, std::string output, std::string tempDir) {
        m_inputFilename = input;
        m_annotatedFilename = annotated;
        m_outputFilename = output;
        m_tempDir = tempDir;
    }

    std::string input()          { return m_inputFilename; }
    std::string annotated()      { return m_annotatedFilename; }
    std::string output()         { return m_outputFilename; }
    std::string sheetness()      { return m_tempDir + "/sheetness.nii"; }
    std::string unsharp()        { return m_tempDir + "/unsharp.nii"; }
    std::string roi()        { return m_tempDir + "/annotated-roi.nii"; }
    std::string flat()        { return m_tempDir + "/annotated-flat.nii"; }

    std::string annotatedOutputPart(unsigned part) {
        std::string filename =
            "/annotated-output-part-" + std::to_string(part) + ".nii";
        return m_tempDir + filename;
    }

    std::string backgroundOutputPart(unsigned part) {
        std::string filename =
            "/annotated-background-part-" + std::to_string(part) + ".nii";
        return m_tempDir + filename;
    }

    std::string foregroundOutputPart(unsigned part) {
        std::string filename =
            "/annotated-foreground-part-" + std::to_string(part) + ".nii";
        return m_tempDir + filename;
    }
};

FloatImagePtr multiscaleSheetness(
    FloatImagePtr img, std::vector<float> scales, UCharImagePtr roi = 0
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
            itSingle(singleScaleSheetness,singleScaleSheetness->GetLargestPossibleRegion());
        itk::ImageRegionIterator<FloatImage>
            itMulti(multiscaleSheetness,multiscaleSheetness->GetLargestPossibleRegion());
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

int main(int argc, char * argv [])
{
    if (sizeof(void*) == 8) {
        std::cerr << "Warning: This executable is compiled for 64bit architectures.\n";
        std::cerr << "Compilition for 32bit would significatly descrease memory requirements.\n";
        std::cerr << "\n";
    }

	if (argc != 5) {
	    std::cerr << "Usage: " << argv[0] << " input-CT-image input-annotated temp-folder output-image\n";
	    return EXIT_FAILURE;
	}

    FilenameDb filenames(argv[1], argv[2], argv[4], argv[3]);

    // Get number of images in annotated image
    logSetStage("Preprocessing");
    FloatImagePtr sheetness;
    ShortImagePtr input;
    { // Scope to prevent storing in memory forever
        log("Reading in %s") % filenames.input();
        input = ImageUtils<ShortImage>::readImage(filenames.input());

        log("Unsharp masking");
        FloatImagePtr inputCTUnsharpMasked =
            FilterUtils<FloatImage>::add(
                FilterUtils<ShortImage,FloatImage>::cast(input),
                FilterUtils<FloatImage>::linearTransform(
                    FilterUtils<FloatImage>::substract(
                        FilterUtils<ShortImage,FloatImage>::cast(input),
                        FilterUtils<ShortImage,FloatImage>::gaussian(input, 1.0)), // larger appears to be better
                    10.0, 0.0)
            );
        if (DEBUG) {
            log("Writing unsharp image to %s") % filenames.unsharp();
            ImageUtils<FloatImage>::writeImage(filenames.unsharp(), inputCTUnsharpMasked);
        }

        std::vector<float> sigma;
        sigma.push_back(0.6);
        assert(sigma.size() == 1);
        log("Computing sheetness measure with sigma = %.2f") % sigma.at(0);
        sheetness = multiscaleSheetness(inputCTUnsharpMasked, sigma);

        log("Writing sheetness image to %s") % filenames.sheetness();
        ImageUtils<FloatImage>::writeImage(filenames.sheetness(), sheetness);
    }

    log("Reading in annotated image %s") % filenames.annotated();
    // We really only need a uchar image, so store that to reduce memory footprint
    UCharImagePtr annotatedImage = FilterUtils<ShortImage, UCharImage>::cast(
        ImageUtils<ShortImage>::readImage(filenames.annotated())
    );
    UCharImage::PixelType nLabels = ImageUtils<UCharImage>::maximumValueInImage(annotatedImage);
    log("Found %d labels") % nLabels;

    log("Preparing for segmentation");
    UCharImagePtr temp = FilterUtils<UCharImage, UCharImage>::binaryThresholding(annotatedImage, 1, nLabels);
    UCharImagePtr roi = FilterUtils<UCharImage, UCharImage>::binaryThresholding(annotatedImage,  0, nLabels);

    if (DEBUG) {
        log("Writing roi image to %s") % filenames.roi();
        ImageUtils<UCharImage>::writeImage(filenames.roi(), roi);

        log("Writing thresheld binary image to %s") % filenames.flat();
        ImageUtils<UCharImage>::writeImage(filenames.flat(), temp);
    }

    // for (UCharImage::PixelType i = 0; i < nLabels; i++){
    //     logSetStage("Segmentation #" + std::to_string(i));

    //     UCharImage::PixelType currentLabel = i + 1;
    //     log("Selecting label " + std::to_string(currentLabel));
    //     UCharImagePtr foreground = FilterUtils<UCharImage, UCharImage>::binaryThresholding(annotatedImage, currentLabel, currentLabel);
    //     UCharImagePtr background = FilterUtils<UCharImage, UCharImage>::substract(temp, foreground);

    //     if (DEBUG) {
    //         log("Writing background image to %s") % filenames.backgroundOutputPart(currentLabel);
    //         ImageUtils<UCharImage>::writeImage(filenames.backgroundOutputPart(currentLabel), background);

    //         log("Writing foreground image to %s") % filenames.foregroundOutputPart(currentLabel);
    //         ImageUtils<UCharImage>::writeImage(filenames.foregroundOutputPart(currentLabel), foreground);
    //     }

    //     log("Computing graph cut");
    //     UCharImagePtr currentSegmentationImage = AnnotatedGC::compute(
    //         input, sheetness, background, foreground, roi
    //     );

    //     log("Writing label " + std::to_string(currentLabel)  + " to %s") % filenames.annotatedOutputPart(currentLabel);
    //     ImageUtils<UCharImage>::writeImage(filenames.annotatedOutputPart(currentLabel), currentSegmentationImage);
    // }

    logSetStage("Segmentation");

        UCharImage::PixelType currentLabel = i + 1;
        log("Selecting label " + std::to_string(currentLabel));
        UCharImagePtr foreground = FilterUtils<UCharImage, UCharImage>::binaryThresholding(annotatedImage, currentLabel, currentLabel);
        UCharImagePtr background = FilterUtils<UCharImage, UCharImage>::substract(temp, foreground);

        if (DEBUG) {
            log("Writing background image to %s") % filenames.backgroundOutputPart(currentLabel);
            ImageUtils<UCharImage>::writeImage(filenames.backgroundOutputPart(currentLabel), background);

            log("Writing foreground image to %s") % filenames.foregroundOutputPart(currentLabel);
            ImageUtils<UCharImage>::writeImage(filenames.foregroundOutputPart(currentLabel), foreground);
        }

    log("Computing graph cut");
    UCharImagePtr currentSegmentationImage = AnnotatedGC::compute(
        input, sheetness, background, foreground, roi
    );

        log("Writing label " + std::to_string(currentLabel)  + " to %s") % filenames.annotatedOutputPart(currentLabel);
        ImageUtils<UCharImage>::writeImage(filenames.annotatedOutputPart(currentLabel), currentSegmentationImage);
    }

    logSetStage("Postprocessing");


	return EXIT_SUCCESS;
}