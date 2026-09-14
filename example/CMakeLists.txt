add_executable(build_lssg build_lssg.cc)
add_executable(search_lssg search_lssg.cc)
add_executable(gen_label_gt gen_label_gt.cc)

foreach(target build_lssg search_lssg gen_label_gt)
  target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
