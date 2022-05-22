#pragma once
#include <vector>
#include <tuple>
#include <optional>
#include <CL/sycl.hpp>
#define NDEBUG  // cannot handle aassert in kernels
#define BOOST_NO_EXCEPTIONS
#include <boost/variant2/variant.hpp>
namespace sycl = cl::sycl;

class class_test;

struct Rectangle
{
    double width_, height_;
    double area() const { return width_ * height_; }
};

struct Circle
{
    double radius_;
    double area() const { return M_PI * radius_ * radius_; }
};

using Shape = boost::variant2::variant<Circle, Rectangle>;

struct SoACache
{
private:
    template< typename result_type, typename ...types, std::size_t ...indices >
    static result_type make_struct_imp(std::tuple< types... > t, std::index_sequence< indices... >) // &, &&, const && etc.
    {
        return {std::get< indices >(t)...};
    }

    template< typename result_type, typename ...types >
    static result_type make_struct(std::tuple< types... > t) // &, &&, const && etc.
    {
        return make_struct_imp< result_type, types... >(t, std::index_sequence_for< types... >{});
    }
public:
    using Id = int;
    using Idx = int;
    using View = std::vector<Idx>;
    View id;  // special, do not add to members


    template<class T> class BufferedVector: public std::vector<T>
    {
    public:
        using ReadAccessor = sycl::accessor<T, 1, sycl::access::mode::read, sycl::COMPUTECPP_ACCESS_TARGET_DEVICE>;
        using WriteAccessor = sycl::accessor<T, 1, sycl::access::mode::write, sycl::COMPUTECPP_ACCESS_TARGET_DEVICE>;
        using RWAccessor = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::COMPUTECPP_ACCESS_TARGET_DEVICE>;
        std::optional<sycl::buffer<T, 1>> m_buffer;
        using std::vector<T>::vector;
        void load()     { 
            m_buffer = sycl::buffer<T,1>{ std::vector<T>::data(), sycl::range<1>(std::vector<T>::size())}; 
        }
        void sync()     { m_buffer.reset(); }
        inline ReadAccessor read_access(sycl::handler& cgh) const;
        inline WriteAccessor write_access(sycl::handler& cgh);
        inline RWAccessor read_write_access(sycl::handler& cgh);
    };

    BufferedVector<Id> index;
    BufferedVector<int> breadth;
    BufferedVector<void*> obj;
    BufferedVector<Shape> shapes;
    BufferedVector<double> area;


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
    
    template<class T>
    void Load()
    {
        std::apply([this](auto...mbrs){ (((this->*mbrs).load()), ...);}, T::member_map);
    }

    template<class T>
    T Access(sycl::handler& cgh)
    {
        auto temp{std::apply([this, &cgh](auto...mbrs){ return std::make_tuple(((this->*mbrs).read_write_access(cgh)) ...);}, T::member_map)};
        return SoACache::make_struct<T>(temp);
    }
    template<class T>
    T LoadAndAccess(sycl::handler& cgh)
    {
        Load<T>();
        return Access<T>(cgh);
    }

    template<class T>
    void Sync()
    {
        std::apply([this](auto...mbrs){ (((this->*mbrs).sync()), ...);},T::member_map);
    }

};

template<class T> 
inline typename SoACache::BufferedVector<T>::WriteAccessor SoACache::BufferedVector<T>::write_access(sycl::handler& cgh)
{ 
    return m_buffer.value().template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(cgh); 
}

template<class T> 
inline typename SoACache::BufferedVector<T>::ReadAccessor SoACache::BufferedVector<T>::read_access(sycl::handler& cgh) const  
{
    return m_buffer.value().template get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(cgh); 
}

template<class T> 
inline typename SoACache::BufferedVector<T>::RWAccessor SoACache::BufferedVector<T>::read_write_access(sycl::handler& cgh)  { 
    return m_buffer.value().template get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh); 
}
