# axflowConfig.cmake
# Allows downstream projects to do: find_package(axflow REQUIRED)

@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/axflowTargets.cmake")

check_required_components(axflow)