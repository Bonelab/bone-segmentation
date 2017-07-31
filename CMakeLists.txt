# Basic project setup
cmake_minimum_required(VERSION 3.0.2)
project(BONE_SEGMENTATION)

add_subdirectory (segmentation)

# Install the data directory
option(INSTALL_DATA "Install the samples-volumes" ON)
if (INSTALL_DATA)
    install (DIRECTORY sample-volumes/ DESTINATION sample-volumes)
endif (INSTALL_DATA)