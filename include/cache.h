#pragma once
#include <vector>
#include <tuple>
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

namespace details
{
    template< typename result_type, typename ...types, std::size_t ...indices >
    result_type
    make_struct(std::tuple< types... > t, std::index_sequence< indices... >) // &, &&, const && etc.
    {
        return {std::get< indices >(t)...};
    }
}

template< typename result_type, typename ...types >
result_type
make_struct(std::tuple< types... > t) // &, &&, const && etc.
{
    return details::make_struct< result_type, types... >(t, std::index_sequence_for< types... >{}); // if there is repeated types, then the change for using std::index_sequence_for is trivial
}

class SoACache;
template <class Target, class MemberType>
struct Mapping
{
    using Vector = std::vector<MemberType>;
    constexpr Mapping(MemberType Target::* tgt, Vector SoACache::*src): m_tgt(tgt), m_src(src) {}
    MemberType Target::* m_tgt;
    Vector SoACache::*   m_src;
};

template <class Target, class MemberType>
struct RefMapping
{
    using Vector = std::vector<MemberType>;
    constexpr RefMapping(Vector SoACache::*src): m_src(src) {}
    Vector SoACache::*   m_src;
};


struct SoACache
{
    using Id = int;
    using Idx = int;
    using View = std::vector<Idx>;
    View id;  // special, do not add to members

    std::vector<Id> index;
    std::vector<int> breadth;
    std::vector<void*> obj;


    static constexpr auto members_list = std::tuple{&SoACache::id, &SoACache::breadth, &SoACache::obj};
    enum class Member {Id, Breadth, obj};
    auto AsTuple()  const {
        return std::apply([this](auto ... args) {return std::make_tuple(this->*args...);}, members_list);
    }

    // move indexes around, without changing the logical meaning of the container
    void Swap(Idx a, Idx b)
    {
        if(a!=b)
        {
            Id id_a = id[a];
            Id id_b = id[b];
            std::apply([this,a,b](auto ... args) {(std::swap( (this->*args)[a], (this->*args)[b] ),...);}, members_list);
            std::swap(index[id_a], index[id_b]);
        }
    }

    bool CheckInvariants() const {
        for (size_t id_idx = 0; id_idx < id.size(); ++id_idx)
        {
            Id requiredID = id[id_idx];
            if (id[index[requiredID]] != requiredID)
                return false;

        }
        return std::apply( [this](auto ... args){
            std::vector<size_t> sizes{args.size()...};
            return std::all_of(sizes.begin(), sizes.end(), [&sizes](size_t s){return s==sizes[0];});
            }, AsTuple());
    }

    template<class T>
    T Fetch(int idx) const
    {
        return make_struct<T>(
            std::apply([this, idx](auto...mbrs){ return std::make_tuple(((this->*mbrs).at(idx)) ...);},T::member_map)
        );
    }

    template<class T>
    T RefFetch(int idx)
    {
        return make_struct<T>(
            std::apply([this, idx](auto...mbrs){ return std::forward_as_tuple(((this->*mbrs).at(idx)) ...);},T::member_map)
        );
    }
};

