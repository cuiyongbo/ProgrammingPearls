cmake_minimum_required(VERSION 3.0.0)
project(scaffold VERSION 0.1.0)

# message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
# message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
# message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

include(CTest)
enable_testing()

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

add_executable(graph_evaluate_division graph_evaluate_division.cpp ${PUB_UTIL_FILES})
# target_compile_features(graph_evaluate_division PUBLIC cxx_std_17)
target_compile_features(graph_evaluate_division PUBLIC cxx_auto_type)
target_link_libraries(graph_evaluate_division)

add_executable(graph_redundant_connection graph_redundant_connection.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_redundant_connection PUBLIC cxx_auto_type) # let cmake deduce the c++ standard to use
target_link_libraries(graph_redundant_connection)

add_executable(graph_town_judge graph_town_judge.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_town_judge PUBLIC cxx_auto_type)
target_link_libraries(graph_town_judge)

add_executable(graph_network_delay_time graph_network_delay_time.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_network_delay_time PUBLIC cxx_auto_type)
target_link_libraries(graph_network_delay_time)

add_executable(graph_shortest_path_visiting_all_nodes graph_shortest_path_visiting_all_nodes.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_shortest_path_visiting_all_nodes PUBLIC cxx_auto_type)
target_link_libraries(graph_shortest_path_visiting_all_nodes)

add_executable(graph_minimum_genetic_mutation graph_minimum_genetic_mutation.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_minimum_genetic_mutation PUBLIC cxx_auto_type)
target_link_libraries(graph_minimum_genetic_mutation)

add_executable(graph_reconstruct_itinerary graph_reconstruct_itinerary.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_reconstruct_itinerary PUBLIC cxx_auto_type)
target_link_libraries(graph_reconstruct_itinerary)

add_executable(graph_critical_connections_in_a_network graph_critical_connections_in_a_network.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_critical_connections_in_a_network PUBLIC cxx_auto_type)
target_link_libraries(graph_critical_connections_in_a_network)

add_executable(graph_scc_alg graph_scc_alg.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_scc_alg PUBLIC cxx_auto_type)
target_link_libraries(graph_scc_alg)

add_executable(graph_unique_path graph_unique_path.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_unique_path PUBLIC cxx_auto_type)
target_link_libraries(graph_unique_path)

add_executable(graph_region_cut_by_slash graph_region_cut_by_slash.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_region_cut_by_slash PUBLIC cxx_auto_type)
target_link_libraries(graph_region_cut_by_slash)

add_executable(graph_traversal_01 graph_traversal_01.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_traversal_01 PUBLIC cxx_auto_type)
target_link_libraries(graph_traversal_01)

add_executable(graph_clone graph_clone.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_clone PUBLIC cxx_auto_type)
target_link_libraries(graph_clone)

# BINARY SEARCH

add_executable(binary_search_search_insert_pos binary_search_search_insert_pos.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_search_insert_pos PUBLIC cxx_auto_type)
target_link_libraries(binary_search_search_insert_pos)

add_executable(binary_search_search_in_rotated_sorted_array binary_search_search_in_rotated_sorted_array.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_search_in_rotated_sorted_array PUBLIC cxx_auto_type)
target_link_libraries(binary_search_search_in_rotated_sorted_array)

add_executable(binary_search_matrix_search binary_search_matrix_search.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_matrix_search PUBLIC cxx_auto_type)
target_link_libraries(binary_search_matrix_search)

add_executable(binary_search_sqrt binary_search_sqrt.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_sqrt PUBLIC cxx_auto_type)
target_link_libraries(binary_search_sqrt)

add_executable(median_finder median_finder.cpp ${PUB_UTIL_FILES})
target_compile_features(median_finder PUBLIC cxx_auto_type)
target_link_libraries(median_finder)

add_executable(binary_search_kth_smallest_pair_dis binary_search_kth_smallest_pair_dis.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_kth_smallest_pair_dis PUBLIC cxx_auto_type)
target_link_libraries(binary_search_kth_smallest_pair_dis)

# BACKTRACE

add_executable(backtrace_combination backtrace_combination.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_combination PUBLIC cxx_auto_type)
target_link_libraries(backtrace_combination)

add_executable(backtrace_permutation backtrace_permutation.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_permutation PUBLIC cxx_auto_type)
target_link_libraries(backtrace_permutation)

add_executable(backtrace_generate_parenthesis backtrace_generate_parenthesis.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_generate_parenthesis PUBLIC cxx_auto_type)
target_link_libraries(backtrace_generate_parenthesis)

add_executable(backtrace_sudoku_solver backtrace_sudoku_solver.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_sudoku_solver PUBLIC cxx_auto_type)
target_link_libraries(backtrace_sudoku_solver)

add_executable(backtrace_word_search backtrace_word_search.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_word_search PUBLIC cxx_auto_type)
target_link_libraries(backtrace_word_search)

add_executable(backtrace_word_ladder backtrace_word_ladder.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_word_ladder PUBLIC cxx_auto_type)
target_link_libraries(backtrace_word_ladder)

add_executable(backtrace_01_matrix backtrace_01_matrix.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_01_matrix PUBLIC cxx_auto_type)
target_link_libraries(backtrace_01_matrix)

add_executable(backtrace_partition_array backtrace_partition_array.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_partition_array PUBLIC cxx_auto_type)
target_link_libraries(backtrace_partition_array)

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
target_compile_features(trie_tree_prefix_and_suffix_search PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_prefix_and_suffix_search)

add_executable(trie_tree_add_and_search_word trie_tree_add_and_search_word.cpp ${PUB_UTIL_FILES})
target_compile_features(trie_tree_add_and_search_word PUBLIC cxx_auto_type)
target_link_libraries(trie_tree_add_and_search_word)

add_executable(advanced_topics_sliding_window advanced_topics_sliding_window.cpp ${PUB_UTIL_FILES})
target_compile_features(advanced_topics_sliding_window PUBLIC cxx_auto_type)
target_link_libraries(advanced_topics_sliding_window)

add_executable(advanced_topics_online_stock_span advanced_topics_online_stock_span.cpp ${PUB_UTIL_FILES})
target_compile_features(advanced_topics_online_stock_span PUBLIC cxx_auto_type)
target_link_libraries(advanced_topics_online_stock_span)

add_executable(advanced_topics_range_sum_query advanced_topics_range_sum_query.cpp ${PUB_UTIL_FILES})
target_compile_features(advanced_topics_range_sum_query PUBLIC cxx_auto_type)
target_link_libraries(advanced_topics_range_sum_query)

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

# DP

add_executable(dp_range_query dp_range_query.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_range_query PUBLIC cxx_auto_type)
target_link_libraries(dp_range_query)

add_executable(dp_climb_stairs dp_climb_stairs.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_climb_stairs PUBLIC cxx_auto_type)
target_link_libraries(dp_climb_stairs)

add_executable(dp_maximum_subarray dp_maximum_subarray.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_maximum_subarray PUBLIC cxx_auto_type)
target_link_libraries(dp_maximum_subarray)

add_executable(dp_unique_paths dp_unique_paths.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_unique_paths PUBLIC cxx_auto_type)
target_link_libraries(dp_unique_paths)

add_executable(dp_max_rectangle dp_max_rectangle.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_max_rectangle PUBLIC cxx_auto_type)
target_link_libraries(dp_max_rectangle)

add_executable(dp_house_robber dp_house_robber.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_house_robber PUBLIC cxx_auto_type)
target_link_libraries(dp_house_robber)

add_executable(dp_perfect_squares dp_perfect_squares.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_perfect_squares PUBLIC cxx_auto_type)
target_link_libraries(dp_perfect_squares)

add_executable(dp_word_break dp_word_break.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_word_break PUBLIC cxx_auto_type)
target_link_libraries(dp_word_break)

add_executable(dp_longest_increasing_subsequence dp_longest_increasing_subsequence.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_longest_increasing_subsequence PUBLIC cxx_auto_type)
target_link_libraries(dp_longest_increasing_subsequence)

add_executable(dp_unique_bst dp_unique_bst.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_unique_bst PUBLIC cxx_auto_type)
target_link_libraries(dp_unique_bst)

add_executable(dp_min_shelf_height dp_min_shelf_height.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_min_shelf_height PUBLIC cxx_auto_type)
target_link_libraries(dp_min_shelf_height)

add_executable(dp_palindrome_partition dp_palindrome_partition.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_palindrome_partition PUBLIC cxx_auto_type)
target_link_libraries(dp_palindrome_partition)

add_executable(dp_edit_distance dp_edit_distance.cpp ${PUB_UTIL_FILES})
target_compile_features(dp_edit_distance PUBLIC cxx_auto_type)
target_link_libraries(dp_edit_distance)

# BINARY SEARCH 

add_executable(binary_tree_traversal binary_tree_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_traversal PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_traversal)

add_executable(binary_tree_level_order_traversal binary_tree_level_order_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_level_order_traversal PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_level_order_traversal)

add_executable(binary_tree_depth binary_tree_depth.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_depth PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_depth)

add_executable(binary_tree_lowest_common_ancestor binary_tree_lowest_common_ancestor.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_lowest_common_ancestor PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_lowest_common_ancestor)

add_executable(binary_tree_is_subtree binary_tree_is_subtree.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_is_subtree PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_is_subtree)

add_executable(binary_tree_vertical_traversal binary_tree_vertical_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_vertical_traversal PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_vertical_traversal)

add_executable(binary_tree_symmetric_tree binary_tree_symmetric_tree.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_symmetric_tree PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_symmetric_tree)

add_executable(binary_tree_leaf_similar binary_tree_leaf_similar.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_leaf_similar PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_leaf_similar)

add_executable(binary_tree_prune_tree binary_tree_prune_tree.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_prune_tree PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_prune_tree)

add_executable(binary_tree_path_sum binary_tree_path_sum.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_path_sum PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_path_sum)

add_executable(sum_root_to_leaf_numbers sum_root_to_leaf_numbers.cpp ${PUB_UTIL_FILES})
target_compile_features(sum_root_to_leaf_numbers PUBLIC cxx_auto_type)
target_link_libraries(sum_root_to_leaf_numbers)

add_executable(binary_tree_serialization_and_deserializaiton binary_tree_serialization_and_deserializaiton.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_tree_serialization_and_deserializaiton PUBLIC cxx_auto_type)
target_link_libraries(binary_tree_serialization_and_deserializaiton)

add_executable(list_traversal list_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(list_traversal PUBLIC cxx_auto_type)
target_link_libraries(list_traversal)

add_executable(list_cycle list_cycle.cpp ${PUB_UTIL_FILES})
target_compile_features(list_cycle PUBLIC cxx_auto_type)
target_link_libraries(list_cycle)

add_executable(list_sort list_sort.cpp ${PUB_UTIL_FILES})
target_compile_features(list_sort PUBLIC cxx_auto_type)
target_link_libraries(list_sort)

add_executable(singly_linked_list_tester singly_linked_list_tester.cpp ${PUB_UTIL_FILES})
target_compile_features(singly_linked_list_tester PUBLIC cxx_auto_type)
target_link_libraries(singly_linked_list_tester)

add_executable(divide_and_conquer_exercises divide_and_conquer_exercises.cpp ${PUB_UTIL_FILES})
target_compile_features(divide_and_conquer_exercises PUBLIC cxx_auto_type)
target_link_libraries(divide_and_conquer_exercises)

add_executable(bst_traversal bst_traversal.cpp ${PUB_UTIL_FILES})
target_compile_features(bst_traversal PUBLIC cxx_auto_type)
target_link_libraries(bst_traversal)

add_executable(bst_search bst_search.cpp ${PUB_UTIL_FILES})
target_compile_features(bst_search PUBLIC cxx_auto_type)
target_link_libraries(bst_search)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
