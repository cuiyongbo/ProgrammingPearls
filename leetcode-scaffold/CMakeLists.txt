cmake_minimum_required(VERSION 3.0.0)
project(scaffold VERSION 0.1.0)

# message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
# message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
# message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

include(CTest)
enable_testing()


if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()


file(GLOB_RECURSE PUB_UTIL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp")
# message(STATUS ${PUB_UTIL_FILES})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

SET(CMAKE_CXX_FLAGS " -Wall")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    message(STATUS "Configuring ${PROJECT_NAME} in debug mode")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Configuring ${PROJECT_NAME} in release mode")
else()
    message(STATUS "Unrecognized build type - will use cmake defaults")
endif()

# message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
# message("List of compile features: ${CMAKE_CXX_COMPILE_FEATURES}")

add_executable(graph_connected_components graph_connected_components.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_connected_components PUBLIC cxx_std_17)
target_link_libraries(graph_connected_components PRIVATE spdlog::spdlog_header_only)

add_executable(graph_redundant_connection graph_redundant_connection.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_redundant_connection PUBLIC cxx_auto_type) # let cmake deduce the c++ standard to use
target_link_libraries(graph_redundant_connection PRIVATE spdlog::spdlog_header_only)

add_executable(graph_town_judge graph_town_judge.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_town_judge PUBLIC cxx_auto_type)
target_link_libraries(graph_town_judge PRIVATE spdlog::spdlog_header_only)

add_executable(graph_network_delay_time graph_network_delay_time.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_network_delay_time PUBLIC cxx_auto_type)
target_link_libraries(graph_network_delay_time PRIVATE spdlog::spdlog_header_only)

add_executable(graph_shortest_path_visiting_all_nodes graph_shortest_path_visiting_all_nodes.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_shortest_path_visiting_all_nodes PUBLIC cxx_auto_type)
target_link_libraries(graph_shortest_path_visiting_all_nodes PRIVATE spdlog::spdlog_header_only)

add_executable(graph_minimum_genetic_mutation graph_minimum_genetic_mutation.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_minimum_genetic_mutation PUBLIC cxx_auto_type)
target_link_libraries(graph_minimum_genetic_mutation PRIVATE spdlog::spdlog_header_only)

add_executable(graph_reconstruct_itinerary graph_reconstruct_itinerary.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_reconstruct_itinerary PUBLIC cxx_auto_type)
target_link_libraries(graph_reconstruct_itinerary PRIVATE spdlog::spdlog_header_only)

add_executable(graph_critical_connections graph_critical_connections.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_critical_connections PUBLIC cxx_auto_type)
target_link_libraries(graph_critical_connections PRIVATE spdlog::spdlog_header_only)

add_executable(graph_scc_alg graph_scc_alg.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_scc_alg PUBLIC cxx_auto_type)
target_link_libraries(graph_scc_alg PRIVATE spdlog::spdlog_header_only)

add_executable(graph_unique_path graph_unique_path.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_unique_path PUBLIC cxx_auto_type)
target_link_libraries(graph_unique_path PRIVATE spdlog::spdlog_header_only)

add_executable(graph_region_cut_by_slash graph_region_cut_by_slash.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_region_cut_by_slash PUBLIC cxx_auto_type)
target_link_libraries(graph_region_cut_by_slash PRIVATE spdlog::spdlog_header_only)

add_executable(graph_traversal graph_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_traversal PUBLIC cxx_auto_type)
target_link_libraries(graph_traversal PRIVATE spdlog::spdlog_header_only)

add_executable(graph_clone graph_clone.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_clone PUBLIC cxx_auto_type)
target_link_libraries(graph_clone PRIVATE spdlog::spdlog_header_only)

add_executable(graph_course_schedule graph_course_schedule.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_course_schedule PUBLIC cxx_auto_type)
target_link_libraries(graph_course_schedule PRIVATE spdlog::spdlog_header_only)

add_executable(graph_coloring graph_coloring.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_coloring PUBLIC cxx_auto_type)
target_link_libraries(graph_coloring PRIVATE spdlog::spdlog_header_only)

# BINARY SEARCH

add_executable(bs_search_insert_pos bs_search_insert_pos.cpp ${PUB_UTIL_FILES})
target_compile_features(bs_search_insert_pos PUBLIC cxx_auto_type)
target_link_libraries(bs_search_insert_pos PRIVATE spdlog::spdlog_header_only)

add_executable(bs_search_in_rotated_sorted_array bs_search_in_rotated_sorted_array.cpp ${PUB_UTIL_FILES})
target_compile_features(bs_search_in_rotated_sorted_array PUBLIC cxx_auto_type)
target_link_libraries(bs_search_in_rotated_sorted_array PRIVATE spdlog::spdlog_header_only)

add_executable(bs_matrix_search bs_matrix_search.cpp ${PUB_UTIL_FILES})
target_compile_features(bs_matrix_search PUBLIC cxx_auto_type)
target_link_libraries(bs_matrix_search PRIVATE spdlog::spdlog_header_only)

add_executable(bs_sqrt bs_sqrt.cpp ${PUB_UTIL_FILES})
target_compile_features(bs_sqrt PUBLIC cxx_auto_type)
target_link_libraries(bs_sqrt PRIVATE spdlog::spdlog_header_only)

add_executable(bs_kth_smallest_pair_dis bs_kth_smallest_pair_dis.cpp ${PUB_UTIL_FILES})
target_compile_features(bs_kth_smallest_pair_dis PUBLIC cxx_auto_type)
target_link_libraries(bs_kth_smallest_pair_dis PRIVATE spdlog::spdlog_header_only)

# BACKTRACE

add_executable(backtrace_combination backtrace_combination.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_combination PUBLIC cxx_auto_type)
target_link_libraries(backtrace_combination spdlog::spdlog_header_only)

add_executable(backtrace_permutation backtrace_permutation.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_permutation PUBLIC cxx_auto_type)
target_link_libraries(backtrace_permutation spdlog::spdlog_header_only)

add_executable(backtrace_generate_parenthesis backtrace_generate_parenthesis.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_generate_parenthesis PUBLIC cxx_auto_type)
target_link_libraries(backtrace_generate_parenthesis spdlog::spdlog_header_only)

add_executable(backtrace_sudoku_solver backtrace_sudoku_solver.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_sudoku_solver PUBLIC cxx_auto_type)
target_link_libraries(backtrace_sudoku_solver spdlog::spdlog_header_only)

add_executable(backtrace_word_search backtrace_word_search.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_word_search PUBLIC cxx_auto_type)
target_link_libraries(backtrace_word_search spdlog::spdlog_header_only)

add_executable(backtrace_word_ladder backtrace_word_ladder.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_word_ladder PUBLIC cxx_auto_type)
target_link_libraries(backtrace_word_ladder spdlog::spdlog_header_only)

add_executable(backtrace_01_matrix backtrace_01_matrix.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_01_matrix PUBLIC cxx_auto_type)
target_link_libraries(backtrace_01_matrix spdlog::spdlog_header_only)

add_executable(backtrace_partition_array backtrace_partition_array.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_partition_array PUBLIC cxx_auto_type)
target_link_libraries(backtrace_partition_array spdlog::spdlog_header_only)

# ADVANCE TOPICS

add_executable(trie_tree_scaffold trie_tree_scaffold.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_scaffold PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_scaffold)

add_executable(trie_tree_replace_word trie_tree_replace_word.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_replace_word PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_replace_word)

add_executable(trie_tree_magic_word trie_tree_magic_word.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_magic_word PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_magic_word)

add_executable(trie_tree_map_sum_pairs trie_tree_map_sum_pairs.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_map_sum_pairs PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_map_sum_pairs)

add_executable(trie_tree_prefix_and_suffix_search trie_tree_prefix_and_suffix_search.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_prefix_and_suffix_search PUBLIC cxx_std_20)
target_link_libraries(trie_tree_prefix_and_suffix_search)

add_executable(trie_tree_add_and_search_word trie_tree_add_and_search_word.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_add_and_search_word PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_add_and_search_word)

add_executable(advanced_topics_range_sum_query advanced_topics_range_sum_query.cpp ${PUB_UTIL_FILES})
target_compile_features(advanced_topics_range_sum_query PUBLIC cxx_auto_type)
target_link_libraries(advanced_topics_range_sum_query)

add_executable(lru_cache lru_cache.cpp ${PUB_UTIL_FILES})
target_compile_features(lru_cache PUBLIC cxx_std_20)
target_link_libraries(lru_cache PRIVATE spdlog::spdlog_header_only)

add_executable(lfu_cache lfu_cache.cpp ${PUB_UTIL_FILES})
target_compile_features(lfu_cache PUBLIC cxx_auto_type)
target_link_libraries(lfu_cache PRIVATE spdlog::spdlog_header_only)

# TWO POINTERS

add_executable(two_pointers_assign_cookies two_pointers_assign_cookies.cpp ${PUB_UTIL_FILES})
target_compile_features(two_pointers_assign_cookies PUBLIC cxx_auto_type)
target_link_libraries(two_pointers_assign_cookies)

add_executable(two_pointers_trapping_water two_pointers_trapping_water.cpp ${PUB_UTIL_FILES})
target_compile_features(two_pointers_trapping_water PUBLIC cxx_auto_type)
target_link_libraries(two_pointers_trapping_water)

add_executable(two_pointers_squares_of_sorted_array two_pointers_squares_of_sorted_array.cpp ${PUB_UTIL_FILES})
target_compile_features(two_pointers_squares_of_sorted_array PUBLIC cxx_auto_type)
target_link_libraries(two_pointers_squares_of_sorted_array)

add_executable(two_pointers_reverse_letters two_pointers_reverse_letters.cpp ${PUB_UTIL_FILES})
target_compile_features(two_pointers_reverse_letters PUBLIC cxx_auto_type)
target_link_libraries(two_pointers_reverse_letters)

add_executable(two_pointers_two_sum two_pointers_two_sum.cpp ${PUB_UTIL_FILES})
target_compile_features(two_pointers_two_sum PUBLIC cxx_auto_type)
target_link_libraries(two_pointers_two_sum)

add_executable(two_pointers_median_finder two_pointers_median_finder.cpp ${PUB_UTIL_FILES})
target_compile_features(two_pointers_median_finder PUBLIC cxx_auto_type)
target_link_libraries(two_pointers_median_finder)

# DP

add_executable(dp_longest_common_subsequence dp_longest_common_subsequence.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_longest_common_subsequence PUBLIC cxx_auto_type)
target_link_libraries(dp_longest_common_subsequence PRIVATE spdlog::spdlog_header_only)

add_executable(dp_range_query dp_range_query.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_range_query PUBLIC cxx_auto_type)
target_link_libraries(dp_range_query spdlog::spdlog_header_only)

add_executable(dp_climb_stairs dp_climb_stairs.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_climb_stairs PUBLIC cxx_auto_type)
target_link_libraries(dp_climb_stairs spdlog::spdlog_header_only)

add_executable(dp_max_subarray dp_max_subarray.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_max_subarray PUBLIC cxx_auto_type)
target_link_libraries(dp_max_subarray spdlog::spdlog_header_only)

add_executable(dp_unique_paths dp_unique_paths.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_unique_paths PUBLIC cxx_auto_type)
target_link_libraries(dp_unique_paths spdlog::spdlog_header_only)

add_executable(dp_max_rectangle dp_max_rectangle.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_max_rectangle PUBLIC cxx_auto_type)
target_link_libraries(dp_max_rectangle spdlog::spdlog_header_only)

add_executable(dp_house_robber dp_house_robber.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_house_robber PUBLIC cxx_auto_type)
target_link_libraries(dp_house_robber spdlog::spdlog_header_only)

add_executable(dp_perfect_squares dp_perfect_squares.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_perfect_squares PUBLIC cxx_auto_type)
target_link_libraries(dp_perfect_squares spdlog::spdlog_header_only)

add_executable(dp_word_break dp_word_break.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_word_break PUBLIC cxx_auto_type)
target_link_libraries(dp_word_break spdlog::spdlog_header_only)

add_executable(dp_longest_increasing_subsequence dp_longest_increasing_subsequence.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_longest_increasing_subsequence PUBLIC cxx_auto_type)
target_link_libraries(dp_longest_increasing_subsequence spdlog::spdlog_header_only)

add_executable(dp_unique_bst dp_unique_bst.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_unique_bst PUBLIC cxx_auto_type)
target_link_libraries(dp_unique_bst spdlog::spdlog_header_only)

add_executable(dp_min_shelf_height dp_min_shelf_height.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_min_shelf_height PUBLIC cxx_auto_type)
target_link_libraries(dp_min_shelf_height spdlog::spdlog_header_only)

add_executable(dp_palindrome_partition dp_palindrome_partition.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_palindrome_partition PUBLIC cxx_auto_type)
target_link_libraries(dp_palindrome_partition spdlog::spdlog_header_only)

add_executable(dp_edit_distance dp_edit_distance.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_edit_distance PUBLIC cxx_auto_type)
target_link_libraries(dp_edit_distance spdlog::spdlog_header_only)

add_executable(dp_cut_rod dp_cut_rod.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_cut_rod PUBLIC cxx_auto_type)
target_link_libraries(dp_cut_rod spdlog::spdlog_header_only)

add_executable(dp_misc_exercises dp_misc_exercises.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_misc_exercises PUBLIC cxx_auto_type)
target_link_libraries(dp_misc_exercises spdlog::spdlog_header_only)

# BINARY TREE 

add_executable(binary_tree_dfs_traversal binary_tree_dfs_traversal.cpp ${PUB_UTIL_FILES})
#target_compile_features(binary_tree_dfs_traversal PUBLIC cxx_auto_type)
target_compile_features(binary_tree_dfs_traversal PUBLIC cxx_std_17)
target_link_libraries(binary_tree_dfs_traversal spdlog::spdlog_header_only)

add_executable(binary_tree_level_order_traversal binary_tree_level_order_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_level_order_traversal PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_level_order_traversal spdlog::spdlog_header_only)

add_executable(binary_tree_depth binary_tree_depth.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_depth PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_depth spdlog::spdlog_header_only)

add_executable(binary_tree_lca binary_tree_lca.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_lca PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_lca spdlog::spdlog_header_only)

add_executable(binary_tree_is_subtree binary_tree_is_subtree.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_is_subtree PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_is_subtree spdlog::spdlog_header_only)

add_executable(binary_tree_prune_tree binary_tree_prune_tree.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_prune_tree PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_prune_tree spdlog::spdlog_header_only)

add_executable(binary_tree_path_sum binary_tree_path_sum.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_path_sum PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_path_sum spdlog::spdlog_header_only)

add_executable(binary_tree_serialization_and_deserializaiton binary_tree_serialization_and_deserializaiton.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_serialization_and_deserializaiton PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_serialization_and_deserializaiton spdlog::spdlog_header_only)

add_executable(binary_tree_misc_exercises binary_tree_misc_exercises.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_misc_exercises PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_misc_exercises spdlog::spdlog_header_only)

# LIST

add_executable(list_traversal list_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(list_traversal PUBLIC cxx_auto_type)
target_link_libraries(list_traversal PRIVATE spdlog::spdlog_header_only)

add_executable(list_cycle list_cycle.cpp ${PUB_UTIL_FILES})
target_compile_features(list_cycle PUBLIC cxx_auto_type)
target_link_libraries(list_cycle PRIVATE spdlog::spdlog_header_only)

add_executable(list_sort list_sort.cpp ${PUB_UTIL_FILES})
target_compile_features(list_sort PUBLIC cxx_auto_type)
target_link_libraries(list_sort PRIVATE spdlog::spdlog_header_only)

add_executable(singly_linked_list_tester singly_linked_list_tester.cpp ${PUB_UTIL_FILES})
target_compile_features(singly_linked_list_tester PUBLIC cxx_auto_type)
target_link_libraries(singly_linked_list_tester)

add_executable(divide_and_conquer_exercises divide_and_conquer_exercises.cpp ${PUB_UTIL_FILES})
target_compile_features(divide_and_conquer_exercises PUBLIC cxx_auto_type)
target_link_libraries(divide_and_conquer_exercises)

add_executable(quick_sort_tester quick_sort_tester.cpp ${PUB_UTIL_FILES})
target_compile_features(quick_sort_tester PUBLIC cxx_auto_type)
target_link_libraries(quick_sort_tester)

add_executable(bst_traversal bst_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(bst_traversal PUBLIC cxx_auto_type)
target_link_libraries(bst_traversal PRIVATE spdlog::spdlog_header_only)

add_executable(bst_search bst_search.cpp ${PUB_UTIL_FILES})
target_compile_features(bst_search PUBLIC cxx_auto_type)
target_link_libraries(bst_search PRIVATE spdlog::spdlog_header_only)

add_executable(array_sort array_sort.cpp ${PUB_UTIL_FILES})
target_compile_features(array_sort PUBLIC cxx_auto_type)
target_link_libraries(array_sort PRIVATE spdlog::spdlog_header_only)

add_executable(binary_search_tester binary_search_tester.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_tester PUBLIC cxx_auto_type)
target_link_libraries(binary_search_tester)

add_executable(misc_exercises misc_exercises.cpp ${PUB_UTIL_FILES})
target_compile_features(misc_exercises PUBLIC cxx_auto_type)
target_link_libraries(misc_exercises)

add_executable(monotonic_stack_exercises monotonic_stack_exercises.cpp ${PUB_UTIL_FILES})
target_compile_features(monotonic_stack_exercises PUBLIC cxx_auto_type)
target_link_libraries(monotonic_stack_exercises)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
