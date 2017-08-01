project(ASSEMBLY)

# Sources and headers
set (SRCS main.cxx)

# Build, link, install
add_executable(Assembly ${SRCS})
target_link_libraries(Assembly ${ITK_LIBRARIES})
install (TARGETS Assembly RUNTIME DESTINATION bin)