# tests/CMakeLists.txt

# Collect all test source files,Test file names starting with "tiu_"
file(GLOB TEST_SOURCES "tiu_*.c")

# Define a function to add tests
function(add_cvikernel_test test_source)
    # Extract test name (remove path and extension)
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Define an executable target name, such as test_tiu_add
    set(exe_name ${test_name}_exe)
    
    # Add executable file
    add_executable(${exe_name} ${test_source})
    
    # Library
    target_link_libraries(${exe_name} cvikernel m)
    
    # Set the runtime path (if on UNIX system)
    if(UNIX)
        set_target_properties(${exe_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
    
    # Register Test
    add_test(NAME ${test_name} COMMAND ${exe_name})
endfunction()

# Iterate through all test source files and add tests
foreach(test_src ${TEST_SOURCES})
    add_cvikernel_test(${test_src})
endforeach()
