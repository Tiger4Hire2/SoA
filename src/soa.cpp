#include <gtest/gtest.h>
#include "cache.h"


struct BreadthObj{
    int& breadth; 
    void*& obj;
    static constexpr auto member_map = std::tuple{
                    RefMapping{&SoACache::obj},
                    RefMapping{&SoACache::breadth}
            };
};

struct ConstBreadthObj{
    int breadth; 
    void* obj;
    static constexpr auto member_map = std::tuple{
                    Mapping{&ConstBreadthObj::obj, &SoACache::obj},
                    Mapping{&ConstBreadthObj::breadth, &SoACache::breadth}
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
    auto obj0 = test_obj.Fetch<BreadthObj>(0);
    EXPECT_EQ(sizepos.breadth,1);
    sizepos.breadth = 2;
    EXPECT_EQ(test_obj.breadth[1], 1);
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
