.PHONY: all clean insertionSort mergeSort

# Please tweak the following variable definitions as needed by your
# project, except GTEST_HEADERS, which you can use in your own targets
# but shouldn't modify.

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = ~/googletest/googletest

# Where to find user code.
USER_DIR = .

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wall -Wextra

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
INSERTION_SORT = insertion_sort_unittest
MERGE_SORT = merge_sort_unittest

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

all: $(INSERTION_SORT) $(MERGE_SORT)

insertionSort: $(INSERTION_SORT)
mergeSort: $(MERGE_SORT)


clean:
	@rm -rf *.o *.dSYM a.out *.i *.s
	@file * | grep executable | grep -v script | cut -d: -f1 | xargs rm -f
	
# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
#gtest-all.o : $(GTEST_SRCS_)
#	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
#            $(GTEST_DIR)/src/gtest-all.cc
#
#gtest.a : gtest-all.o
#	$(AR) $(ARFLAGS) $@ $^
#
#gtest_main.o : $(GTEST_SRCS_)
#	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
#            $(GTEST_DIR)/src/gtest_main.cc
#
#gtest_main.a : gtest-all.o gtest_main.o
#	$(AR) $(ARFLAGS) $@ $^

#main.o : $(GTEST_SRCS_)
#main.o : $(GTEST_HEADERS)

# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

insertion_sort_alg.o: $(USER_DIR)/sort_algs.h  $(USER_DIR)/util.h $(GTEST_HEADERS)
insertion_sort_unittest.o: $(USER_DIR)/sort_algs.h $(USER_DIR)/util.h  $(GTEST_HEADERS)
insertion_sort_unittest: main.o util.o insertion_sort_alg.o insertion_sort_unittest.o gtest.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

merge_sort_alg.o: $(USER_DIR)/sort_algs.h  $(USER_DIR)/util.h $(GTEST_HEADERS)
merge_sort_unittest.o: $(USER_DIR)/sort_algs.h $(USER_DIR)/util.h  $(GTEST_HEADERS)
merge_sort_unittest: main.o util.o merge_sort_alg.o merge_sort_unittest.o gtest.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@
