project(INTENSITY_BASED_GRAPH_CUT)

# Sources and headers
set (SRCS Main.cxx)

# Build, link, install
add_executable(IntensityBasedGraphCut ${SRCS})
target_link_libraries(IntensityBasedGraphCut MaxFlow ${ITK_LIBRARIES})
install (TARGETS IntensityBasedGraphCut RUNTIME DESTINATION bin)