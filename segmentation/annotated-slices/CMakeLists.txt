project(ANNOTATED_SLICES)

# Sources and headers
set (SRCS main.cxx)

# Build, link, install
add_executable(AnnotatedSlices ${SRCS})
target_link_libraries(AnnotatedSlices MaxFlow ${ITK_LIBRARIES})
install (TARGETS AnnotatedSlices RUNTIME DESTINATION bin)