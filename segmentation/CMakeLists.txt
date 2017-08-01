# Setup libraries and helper functions
include_directories(include)
include_directories(include/utils)
include_directories(include/filters)

# Boost
include_directories(include/boost_1_44_0)

# MaxFlow
include_directories(include/maxflow-v3.01)
add_library (MaxFlow include/maxflow-v3.01/maxflow.cpp include/maxflow-v3.01/graph.cpp)

# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
#set(ITK_LIBS ${ITK_LIBRARIES})

# Add each algorithm as an option
option(ALGO_BUILD_ALL "Build and install all segmentation algorithms" ON)
if (ALGO_BUILD_ALL)
    option (ALGO_GEOMETRIC_AC "Build and install geometric active contours." ON)
    if (ALGO_GEOMETRIC_AC)
        add_subdirectory (geometric-active-contours)
    endif(ALGO_GEOMETRIC_AC)

    option (ALGO_INTENSITY_BASED_GC "Build and install intensity based graph cut." ON)
    if (ALGO_INTENSITY_BASED_GC)
        add_subdirectory (intensity-based-graph-cut)
    endif(ALGO_INTENSITY_BASED_GC)

    option (ALGO_KRCAH_GC "Build and install Krcah et al. graph cut." ON)
    if (ALGO_KRCAH_GC)
        add_subdirectory (krcah-graph-cut)
    endif(ALGO_KRCAH_GC)

    option (ALGO_ZHANG_ET_AL "Build and install Zhang et al." ON)
    if (ALGO_ZHANG_ET_AL)
        add_subdirectory (zhang)
    endif(ALGO_ZHANG_ET_AL)

    option (ALGO_ANNOTATED "Build and install annotated slices algorithm" ON)
    if (ALGO_ANNOTATED)
        add_subdirectory (annotated-slices)
    endif(ALGO_ANNOTATED)

    option (ALGO_ASSEMBLY "Build and install assembly" ON)
    if (ALGO_ASSEMBLY)
        add_subdirectory (assembly)
    endif(ALGO_ASSEMBLY)

    # New segmentation algorithms go here
endif (ALGO_BUILD_ALL)