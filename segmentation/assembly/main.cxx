#include "Globals.hpp"
#include "ImageUtils.hpp"
#include "FilterUtils.hpp"
#include <string>
#include <vector>

int main(int argc, char * argv [])
{
    if (sizeof(void*) == 8) {
        std::cerr << "Warning: This executable is compiled for 64bit architectures.\n";
        std::cerr << "Compilition for 32bit would significatly descrease memory requirements.\n";
        std::cerr << "\n";
    }

	if (argc < 2) {
	    std::cerr << "Usage: " << argv[0] << " outputImage inputImage1 [inputImage2] [...]\n";
	    return EXIT_FAILURE;
	}

    logSetStage("Assembling");
    std::string outputFileName = argv[1];

    log("Processing %s") % argv[2];
    UCharImagePtr ithImage = FilterUtils<ShortImage, UCharImage>::cast(ImageUtils<ShortImage>::readImage(argv[2]));
    UCharImagePtr outputImage = FilterUtils<UCharImage, UCharImage>::binaryThresholding(
        ithImage,
        1, ImageUtils<UCharImage>::maximumValueInImage(ithImage)
    );

    for (int i = 3; i < argc; i++){
        log("Processing %s") % argv[i];

        // Read the input and make all pixels value 1
        ithImage = FilterUtils<ShortImage, UCharImage>::cast(ImageUtils<ShortImage>::readImage(argv[i]));
        UCharImagePtr procInput = FilterUtils<UCharImage, UCharImage>::binaryThresholding(
            ithImage,
            1, ImageUtils<UCharImage>::maximumValueInImage(ithImage)
        );

        // Ensure there is no overlap between images
        UCharImage::PixelType max = ImageUtils<UCharImage>::maximumValueInImage(
            FilterUtils<UCharImage, UCharImage>::add(
                procInput,
                FilterUtils<UCharImage, UCharImage>::binaryThresholding(
                    outputImage,
                    1, ImageUtils<UCharImage>::maximumValueInImage(outputImage)
                )
            )
        );
        assert(max ==  1 && "Images overlap"); // Cheap trick to get error message in assert

        // Add images properly
        UCharImagePtr tempOutputImage = FilterUtils<UCharImage, UCharImage>::add(
            outputImage,
            FilterUtils<UCharImage, UCharImage>::binaryThresholding(
                    ithImage,
                    1, ImageUtils<UCharImage>::maximumValueInImage(ithImage),
                    i-1, 0 //label = i-1
            )
        );

        std::swap(outputImage, tempOutputImage);
    }

    log("Writing to %s") % outputFileName;
    ImageUtils<UCharImage>::writeImage(outputFileName, outputImage);

	return EXIT_SUCCESS;
}