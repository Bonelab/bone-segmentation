cmake_minimum_required(VERSION 2.6)
project(GEOMETRIC_AC)

# Sources and headers
set (SRCS GeometricActiveContours.cxx)

# Build, link, install
add_executable(GeometricAC ${SRCS} ${HDRS})
target_link_libraries(GeometricAC ${ITK_LIBRARIES} ${Boost_LIBRARIES})
install (TARGETS GeometricAC RUNTIME DESTINATION bin)
