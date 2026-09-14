pybind11_add_module(_pylssg lssg_bindings.cc)
target_link_libraries(_pylssg PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(_pylssg PRIVATE ${PROJECT_SOURCE_DIR})

set(LSSG_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/pylssg")
file(MAKE_DIRECTORY "${LSSG_PYTHON_PACKAGE_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pylssg/__init__.py"
    "${LSSG_PYTHON_PACKAGE_DIR}/__init__.py"
    COPYONLY
)
set_target_properties(_pylssg PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${LSSG_PYTHON_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${LSSG_PYTHON_PACKAGE_DIR}"
)
