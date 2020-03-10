.PHONY: all clean

CXXFLAGS += -g -Wall -Wextra -std=c++11

PROC=path_finder
ALG_DIR=./alg
DATA_STRUCT_DIR=./data_struct
UTIL_DIR=./util
SRC_DIR=./src
OBJS=main.o util.o square_grid.o
INCLUDE= -I$(ALG_DIR) -I$(DATA_STRUCT_DIR) -I$(UTIL_DIR) -I$(SRC_DIR)

all: $(PROC)

main.o: $(SRC_DIR)/main.cpp $(UTIL_DIR)/util.h $(UTIL_DIR)/grid_drawer.h \
	$(ALG_DIR)/naive_bfs_alg.h $(ALG_DIR)/breadth_first_search.h \
	$(ALG_DIR)/dijkstra_alg.h $(ALG_DIR)/a_star_alg.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $(SRC_DIR)/main.cpp

$(ALG_DIR)/dijkstra_alg.h: $(DATA_STRUCT_DIR)/weighted_grid.h $(DATA_STRUCT_DIR)/priority_queue.h
$(ALG_DIR)/a_star_alg.h: $(DATA_STRUCT_DIR)/weighted_grid.h $(DATA_STRUCT_DIR)/priority_queue.h

util.o: $(UTIL_DIR)/util.cpp $(DATA_STRUCT_DIR)/simple_graph_struct.h \
	$(DATA_STRUCT_DIR)/square_grid.h \
	$(DATA_STRUCT_DIR)/weighted_grid.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $(UTIL_DIR)/util.cpp

square_grid.o: $(DATA_STRUCT_DIR)/square_grid.cpp $(DATA_STRUCT_DIR)/simple_graph_struct.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $(DATA_STRUCT_DIR)/square_grid.cpp

$(PROC): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	@rm -rf *.dSYM *.o a.out $(PROC)
	@file * | grep executable | grep -v script | cut -d: -f1 | xargs rm
