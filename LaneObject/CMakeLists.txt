cmake_minimum_required(VERSION 2.8)
project( advanced-lane-line )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV 3.2 REQUIRED )
message("OpenCV_LIBS: " ${OpenCV_LIBS} )


file(GLOB Cam_files
    "src/camera_matrix.cpp"
)

file(GLOB Main_files
    "include/lane_lines.hpp"
    "src/lane_lines.cpp"
    "src/polyfit.cpp" 
    "src/main.cpp"
)

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "describe --abbrev=4 --dirty --always --tags"
    OUTPUT_VARIABLE GIT_INFO )

add_definitions( -DGIT_COMMIT="${GIT_INFO}" )

# CAMERA MATRIX
add_executable( compute_camera_matrix ${Cam_files} )	# bind all together
target_compile_options( compute_camera_matrix PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( compute_camera_matrix ${OpenCV_LIBS})		# link libraries

# MAIN
add_executable( main ${Main_files} )	# bind all together
target_compile_options( main PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( main ${OpenCV_LIBS})		# link libraries