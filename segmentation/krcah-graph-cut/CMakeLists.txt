project(KRCAH_GRAPH_CUT)

# Sources and headers
set (SRCS Main.cxx)

# Build, link, install
add_executable(KcrahSegmentation ${SRCS})
target_link_libraries(KcrahSegmentation MaxFlow ${ITK_LIBRARIES})
install (TARGETS KcrahSegmentation RUNTIME DESTINATION bin)