find_package(Catch2 REQUIRED)

add_executable(tests "main.cpp")

target_link_libraries(tests PRIVATE Catch2::Catch2 utility)
target_compile_definitions(tests PRIVATE CATCH_CONFIG_CONSOLE_WIDTH=200)
