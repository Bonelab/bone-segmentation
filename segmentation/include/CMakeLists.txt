set(MAXFLOW_VERSION 3.01)
set(MAXFLOW_DIR maxflow-v${MAXFLOW_VERSION})

set(MAXFLOW_SRCS
    ${MAXFLOW_DIR}/graph.cpp
    ${MAXFLOW_DIR}/maxflow.cpp
    ${MAXFLOW_DIR}/instances.inc
)
set(MAXFLOW_HDRS
    ${MAXFLOW_DIR}/block.h
    ${MAXFLOW_DIR}/graph.h
)

add_library (MaxFlow ${MAXFLOW_SRCS} ${MAXFLOW_HDRS})
target_include_directories (MaxFlow PUBLIC include)
install(TARGETS MaxFlow
    EXPORT MaxFlowTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/maxflow
)
install(FILES ${MAXFLOW_HDRS} DESTINATION include)


# CMake config files for importing this project
# =============================================

# Set some locations for the install
set(INSTALL_LIB_DIR lib)
set(INSTALL_BIN_DIR bin)
set(INSTALL_INCLUDE_DIR include)
if(WIN32 AND NOT CYGWIN)
  set(INSTALL_CMAKE_DIR CMake)
else()
  set(INSTALL_CMAKE_DIR lib/CMake/MaxFlow)
endif()

set (MaxFlow_VERSION_MAJOR 3)
set (Tutorial_VERSION_MINOR 1)

export(PACKAGE MaxFlow)