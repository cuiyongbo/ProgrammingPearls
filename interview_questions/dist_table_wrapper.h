/*
Copyright (c) 2017, Project OSRM contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DIST_TABLE_WRAPPER_H
#define DIST_TABLE_WRAPPER_H

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>
#include <cassert>

namespace osrm
{
namespace util
{

// This Wrapper provides an easier access to a distance table that is given as an linear vector

template <typename T> class DistanceTable
{
  public:
    using Iterator = typename std::vector<T>::iterator;
    using ConstIterator = typename std::vector<T>::const_iterator;

    DistanceTable(std::size_t number_of_nodes)
        : m_nodeCount(number_of_nodes)
    {
        m_table.resize(m_nodeCount * m_nodeCount);
    }

    std::size_t nodeCount() const { return m_nodeCount; }

    T operator()(int from, int to) const
    {
        assert(from < m_nodeCount && "from ID is out of bound");
        assert(to < m_nodeCount && "to ID is out of bound");

        const auto index = from * m_nodeCount + to;

        assert(index < m_table.size() && "index is out of bound");

        return m_table[index];
    }

    void setValue(int from, int to, T value)
    {
        assert(from < m_nodeCount && "from ID is out of bound");
        assert(to < m_nodeCount && "to ID is out of bound");

        const auto index = from * m_nodeCount + to;

        assert(index < m_table.size() && "index is out of bound");

        m_table[index] = value;
    }

    ConstIterator begin() const { return std::begin(m_table); }

    Iterator begin() { return std::begin(m_table); }

    ConstIterator end() const { return std::end(m_table); }

    Iterator end() { return std::end(m_table); }

    int getIndexOfMaxValue() const
    {
        return std::distance(m_table.begin(), std::max_element(m_table.begin(), m_table.end()));
    }

    std::vector<T> getTable() const { return m_table; }
    std::size_t size() const { return m_table.size(); }

  private:
    std::vector<T> m_table;
    const std::size_t m_nodeCount;
};
}
}

#endif // DIST_TABLE_WRAPPER_H
