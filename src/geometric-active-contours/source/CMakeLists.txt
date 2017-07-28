project(GEOMETRIC_AC)

cmake_minimum_required(VERSION 2.6)

set(EXTERNAL_LIBS_SRC_DIR ${GEOMETRIC_AC_SOURCE_DIR}/external-libs-src)


# Project dependency: ITK
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
	INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
	MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)
set(ITK_LIBS ITKCommon ITKIO)


# Compilation and linking of the whole project
include_directories(${GEOMETRIC_AC_SOURCE_DIR} utils 	${EXTERNAL_LIBS_SRC_DIR}/boost_1_44_0 )
add_executable(GeometricAC GeometricActiveContours.cxx)

target_link_libraries(GeometricAC  ${ITK_LIBS})

