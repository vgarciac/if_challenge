cmake_minimum_required(VERSION 2.8)
project( advanced-lane-line )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV 3.2 REQUIRED )
message("OpenCV_LIBS: " ${OpenCV_LIBS} )

file(GLOB SVM_files
    "include/clasifier.hpp"
    "src/clasifier.cpp"
    "src/train_clasifier.cpp"
)

file(GLOB Cam_files
    "src/calibration.cpp"
)

file(GLOB Cars_Files
    "include/clasifier.hpp"
    "src/clasifier.cpp"
    "include/car_detector.hpp"
    "src/car_detector.cpp"
    "src/cars.cpp"
)

file(GLOB Lane_files
    "include/lane_lines.hpp"
    "src/lane_lines.cpp"
    "src/polyfit.cpp"
    "src/lane.cpp"
)

file(GLOB Pilepine_files
    "include/clasifier.hpp"
    "src/clasifier.cpp"
    "include/lane_lines.hpp"
    "include/car_detector.hpp"
    "src/lane_lines.cpp"
    "src/car_detector.cpp"
    "src/polyfit.cpp"
    "src/main.cpp"
)

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "describe --abbrev=4 --dirty --always --tags"
    OUTPUT_VARIABLE GIT_INFO )

add_definitions( -DGIT_COMMIT="${GIT_INFO}" )

# SVM Clasifier
add_executable( clasifier ${SVM_files} )	# bind all together
target_compile_options( clasifier PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( clasifier ${OpenCV_LIBS})		# link libraries

# CAMERA MATRIX
add_executable( compute_camera_matrix ${Cam_files} )	# bind all together
target_compile_options( compute_camera_matrix PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( compute_camera_matrix ${OpenCV_LIBS})		# link libraries

# LANE
add_executable( lane ${Lane_files} )	# bind all together
target_compile_options( lane PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( lane ${OpenCV_LIBS})		# link libraries

# CARS
add_executable( cars ${Cars_Files} )	# bind all together
target_compile_options( cars PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( cars ${OpenCV_LIBS})		# link libraries

# PIPELINE
add_executable( main ${Pilepine_files} )	# bind all together
target_compile_options( main PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( main ${OpenCV_LIBS})		# link libraries

