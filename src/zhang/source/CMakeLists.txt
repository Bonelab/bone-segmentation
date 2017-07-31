project(Zhang)

cmake_minimum_required(VERSION 2.6)

# Project dependency: ITK
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
	INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
	MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)
set(ITK_LIBS ITKCommon ITKIO)


# include common header files
include_directories(utils external-libs-src/boost_1_44_0)

add_executable(Zhang Zhang.cxx)
target_link_libraries (Zhang ${ITK_LIBS})

