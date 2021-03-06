#define CL_TARGET_OPENCL_VERSION 300
#include <gtest/gtest.h>
#include "cache.h"

class assign_elements;
class assign_ids;
class access_test;
class shape_test;

namespace boost
{
	void throw_exception(const std::exception&)
	{

	}
}
auto InitialiseGPU()
{
    using namespace sycl;
    default_selector selector;
    queue myQueue(selector, [](exception_list l) {
        for (auto ep : l) {
            try {
                std::rethrow_exception(ep);
            } catch (const exception& e) {
                std::cout << "Asynchronous exception caught:\n" << e.what();
            }
        }
    });
    return myQueue;
}

struct GPUAccess{
    typename SoACache::BufferedVector<int>::RWAccessor breadth; 
    typename SoACache::BufferedVector<void*>::RWAccessor obj;
    static constexpr auto member_map = std::tuple{
                    &SoACache::breadth,
                    &SoACache::obj
            };
};


struct BreadthObj{
    int& breadth; 
    void*& obj;
    static constexpr auto member_map = std::tuple{
                    &SoACache::breadth,
                    &SoACache::obj
            };
};

struct ConstBreadthObj{
    int breadth; 
    void* obj;
    static constexpr auto member_map = std::tuple{
                    &SoACache::breadth,
                    &SoACache::obj
            };
};


TEST(SoA, Fetch)
{
    const SoACache    test_obj{{0}, {0}, {6}, {nullptr}};
    const auto sizepos = test_obj.Fetch<ConstBreadthObj>(0);
    EXPECT_EQ(sizepos.breadth,6);
    EXPECT_THROW(test_obj.Fetch<ConstBreadthObj>(1), std::out_of_range);
}

TEST(SoA, RefFetch)
{
    SoACache    test_obj{{0,1,2,3}, {0,1,2,3}, {1,2,3,4}, {nullptr, nullptr, nullptr, nullptr}};
    auto obj0 = test_obj.RefFetch<BreadthObj>(0);
    EXPECT_EQ(obj0.breadth,1);
    obj0.breadth = 2;
    EXPECT_EQ(test_obj.breadth[1], 2);
}


TEST(SoA, Invariants)
{
    SoACache    test_obj{{0}, {0}, {6}, {nullptr}};
    EXPECT_TRUE(test_obj.CheckInvariants());
    test_obj.breadth.push_back(8);
    EXPECT_FALSE(test_obj.CheckInvariants());
}



TEST(SoA, swap)
{
    SoACache    test_obj{{0,1,2,3}, {0,1,2,3}, {1,2,3,4}, {nullptr, nullptr, nullptr, nullptr}};
    const std::vector<std::pair<SoACache::Idx, SoACache::Idx>> swap_pairs = {{0,1}, {0,3}, {0,0}, {1,2}, {2,3}};

    for (const auto swap_pair : swap_pairs)
    {
        EXPECT_TRUE(test_obj.CheckInvariants());
        test_obj.Swap(swap_pair.first, swap_pair.second);
        EXPECT_TRUE(test_obj.CheckInvariants());
    }
    for (const auto swap_pair : swap_pairs)
    {
        EXPECT_TRUE(test_obj.CheckInvariants());
        test_obj.Swap(swap_pair.first, swap_pair.second);
        EXPECT_TRUE(test_obj.CheckInvariants());
    }
    EXPECT_TRUE(test_obj.CheckInvariants());
}


TEST(ComputeCPP, SimpleLoop)
{
    using namespace sycl;
    const int nElems = 64u;
    /* We define and initialize data to be copied to the device. */
    int data[nElems] = {0};

    try {
        auto myQueue = InitialiseGPU();
        buffer<int, 1> buf(data, range<1>(nElems));

        myQueue.submit([&](handler& cgh) {
            auto ptr = buf.get_access<access::mode::read_write>(cgh);

            /* We create an nd_range to describe the work space that the kernel is
            * to be executed across. Here we create a linear (one dimensional)
            * nd_range, which creates a work item per element of the vector. The
            * first parameter of the nd_range is the range of global work items
            * and the second is the range of local work items (i.e. the work group
            * range). */
            auto myRange = nd_range<1>(range<1>(nElems), range<1>(nElems / 4));

            /* We construct the lambda outside of the parallel_for function call,
            * though it can be inline inside the function call too. For this
            * parallel_for API the lambda is required to take a single parameter;
            * an item<N> of the same dimensionality as the nd_range - in this
            * case one. Other kernel dispatches might have different parameters -
            * for example, the single_task takes no arguments. */
            auto myKernel = ([=](nd_item<1> item) {
                /* Items have various methods to extract ids and ranges. The
                    * specification has full details of these. Here we use the
                    * item::get_global() to retrieve the global id as an id<1>.
                    * This particular kernel will set the ith element to the value
                    * of i. */
                ptr[item.get_global_id()] = item.get_global_id()[0];
            });

            /* We call the parallel_for() API with two parameters; the nd_range
            * we constructed above and the lambda that we constructed. Because
            * the kernel is a lambda we *must* specify a template parameter to
            * use as a name. */
            cgh.parallel_for<class assign_elements>(myRange, myKernel);
        });

    } catch (const exception& e) {
    std::cout << "Synchronous exception caught:\n" << e.what();
    return;
    }

    /* Check the result is correct. */
    for (int i = 0; i < nElems; i++) {
        EXPECT_EQ(i, data[i]);
        }
}

TEST(ComputeCPP, WriteToSoA)
{
    using namespace sycl;

    SoACache test_obj;
    test_obj.id.resize(1024*1);
    try 
    {
        auto myQueue = InitialiseGPU();

        buffer<int, 1> buf(test_obj.id.data(), range<1>(test_obj.id.size()));
        myQueue.submit([&](handler& cgh) {
            auto ptr = buf.get_access<access::mode::read_write>(cgh);
            auto myRange = nd_range<1>(range<1>(test_obj.id.size()), range<1>(test_obj.id.size() / 4));
            auto myKernel = ([=](nd_item<1> item) {
                ptr[item.get_global_id()] = item.get_global_id()[0];
            });
            cgh.parallel_for<class assign_ids>(myRange, myKernel);
        });

    } catch (const exception& e) 
    {
        std::cout << "Synchronous exception caught:\n" << e.what();
        return;
    }

    /* Check the result is correct. */
    for (int i = 0; i < (int)test_obj.id.size(); i++) {
        if (i != test_obj.id[i])
            EXPECT_EQ(i, test_obj.id[i]);
    }
}


TEST(ComputeCPP, ReadObj)
{
    using namespace sycl;

    SoACache test_obj;
    test_obj.index.resize(1024*1);
    try 
    {
        auto myQueue = InitialiseGPU();

        test_obj.index.load();
        myQueue.submit([&](handler& cgh) {
            auto ptr = test_obj.index.write_access(cgh);
            auto myRange = nd_range<1>(range<1>(test_obj.index.size()), range<1>(test_obj.index.size() / 4));
            auto myKernel = ([ptr](nd_item<1> item) {
                ptr[item.get_global_id()] = item.get_global_id()[0];
            });
            cgh.parallel_for<access_test>(myRange, myKernel);
        });
       test_obj.index.sync();
    } catch (const exception& e) 
    {
        std::cout << "Synchronous exception caught:\n" << e.what();
        return;
    }

    /* Check the result is correct. */
    for (int i = 0; i < (int)test_obj.index.size(); i++) {
        EXPECT_EQ(i, test_obj.index[i]);
    }
}

TEST(ComputeCPP, ThrowOnEmptyVector)
{
    SoACache test_obj;
    auto myQueue = InitialiseGPU();

    EXPECT_THROW(test_obj.Load<GPUAccess>(), sycl::exception);
}

TEST(ComputeCPP, Execute)
{
    using namespace sycl;

    SoACache test_obj;
    test_obj.breadth.resize(1024);
    test_obj.obj.resize(1024);
    try 
    {
        auto myQueue = InitialiseGPU();

        test_obj.Load<GPUAccess>();
        myQueue.submit([&](handler& cgh) {
            auto access = test_obj.Access<GPUAccess>(cgh);
            auto myRange = nd_range<1>(range<1>(test_obj.breadth.size()), range<1>(128));
            auto myKernel = ([access](nd_item<1> item) {
               access.breadth[item.get_global_id()] = item.get_global_id()[0];
               access.obj[item.get_global_id()] = (void*)item.get_global_id()[0];
            });
            cgh.parallel_for<GPUAccess>(myRange, myKernel);
        });
       test_obj.Sync<GPUAccess>();
    } catch (const exception& e) 
    {
        std::cout << "Synchronous exception caught:\n" << e.what();
        return;
    }

    /* Check the result is correct. */
    for (int i = 0; i < (int)test_obj.breadth.size(); i++)
//        if (i != test_obj.breadth[i])
		{
			EXPECT_EQ(i, test_obj.breadth[i]);
			EXPECT_EQ((void*)i, test_obj.obj[i]);
			if (i!=test_obj.breadth[i])
				break;
		}
}



struct ShapeAccess{
    typename SoACache::BufferedVector<Shape>::RWAccessor shape; 
    typename SoACache::BufferedVector<double>::RWAccessor area;
    static constexpr auto member_map = std::tuple{
                    &SoACache::shapes,
                    &SoACache::area
            };
};

TEST(ComputeCPP, Variants)
{
    using namespace sycl;

    SoACache test_obj;
    test_obj.breadth.resize(1024);
    test_obj.obj.resize(1024);

    for (int i = 0; i < 1024; ++i)
        test_obj.shapes.push_back(Rectangle{1,1});

    test_obj.shapes[0] = Rectangle{1,1};
    test_obj.shapes[1] = Rectangle{2,1};
    test_obj.shapes[2] = Rectangle{1,2};
    test_obj.shapes[3] = Circle{1};
    test_obj.area.resize(test_obj.shapes.size());
    try 
    {
        auto myQueue = InitialiseGPU();

        test_obj.Load<ShapeAccess>();
        myQueue.submit([&](handler& cgh) {
            auto access = test_obj.Access<ShapeAccess>(cgh);
            auto myRange = nd_range<1>(range<1>(test_obj.shapes.size()), range<1>(128));
            auto myKernel = ([access](nd_item<1> item) {
                const auto idx = item.get_global_id();
	            access.area[idx] = boost::variant2::visit([](const auto& s){return s.area();}, access.shape[idx]);
            });
            cgh.parallel_for<shape_test>(myRange, myKernel);
        });
       test_obj.Sync<ShapeAccess>();
    } catch (const exception& e) 
    {
        std::cout << "Synchronous exception caught:\n" << e.what();
        return;
    }

    const auto shape = boost::variant2::get<Rectangle>(test_obj.shapes[0]);
    EXPECT_EQ(test_obj.area[0], 1.0);
    EXPECT_EQ(test_obj.area[1], 2.0);
    EXPECT_EQ(test_obj.area[2], 2.0);
    EXPECT_EQ(test_obj.area[3], M_PI);

}