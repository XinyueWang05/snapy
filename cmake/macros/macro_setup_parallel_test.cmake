# A small macro used for setting up the build of a test.
#
# Usage: setup_parallel_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

function(snapy_test_gloo_loopback test_name)
  if(APPLE)
    set(_iface lo0)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_iface lo)
  else()
    return()
  endif()
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
               "GLOO_SOCKET_IFNAME=set:${_iface}")
endfunction()

macro(setup_parallel_test namel cores)
  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(${namel}.${buildl}
                        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${KINTERA_INCLUDE_DIR}
            ${SNAP_INCLUDE_DIR}
            ${HARP_INCLUDE_DIR}
            ${NETCDF_INCLUDES}
            ${TORCH_INCLUDE_DIR}
            ${TORCH_API_INCLUDE_DIR})

  target_link_libraries(${namel}.${buildl}
    PRIVATE snapy::snap
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>
            gtest_main)

  add_test(NAME ${namel}.${buildl}
           COMMAND torchrun
                   --local-addr=127.0.0.1
                   --no-python
                   --nproc-per-node=${cores}
                   $<TARGET_FILE:${namel}.${buildl}>)
  snapy_test_gloo_loopback(${namel}.${buildl})
endmacro()
