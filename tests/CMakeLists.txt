# tests/CMakeLists.txt

# Set the directory where test source files are located
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cv181x")

# Collect all test source files with names starting with "tiu_"
file(GLOB TEST_SOURCES "${TEST_DIR}/tiu_*.c")

# Define a function to add tests
function(add_cvikernel_test test_source)
    # Extract the test name by removing the path and file extension
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Define an executable target name, such as tiu_add_exe
    set(exe_name ${test_name}_exe)
    
    # Add the executable file target
    add_executable(${exe_name} ${test_source})
    
    # Link the necessary libraries
    target_link_libraries(${exe_name} cvikernel m)
    
    # Set the runtime path if on a UNIX system
    if(UNIX)
        set_target_properties(${exe_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
    
    # Register the test with CTest
    add_test(NAME ${test_name} COMMAND ${exe_name})
endfunction()

# Iterate through all test source files and add each as a test
foreach(test_src ${TEST_SOURCES})
    add_cvikernel_test(${test_src})
endforeach()
