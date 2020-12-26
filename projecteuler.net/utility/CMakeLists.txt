find_package(range-v3 REQUIRED)

add_library(utility INTERFACE)

target_sources(utility INTERFACE "include/generators.hpp")

target_include_directories(utility INTERFACE "include")
target_link_libraries(utility INTERFACE range-v3::range-v3)
