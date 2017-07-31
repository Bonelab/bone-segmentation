project(ZHANG)

# Sources and headers
set (SRCS Zhang.cxx)

# Build, link, install
add_executable(Zhang ${SRCS})
target_link_libraries(Zhang ${ITK_LIBRARIES})
install (TARGETS Zhang RUNTIME DESTINATION bin)