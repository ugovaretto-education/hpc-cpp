// Author: Ugo Varetto
// Fast(er) allocators with memory alignment and default initialization:
// - uninitialized
// - initialize with calloc: using calloc to initialize to zero is orders of
// magnitude faster than through std::vector default initialization
// - Lazy allocation for non-POD types with trivial destructor: memory is
// allocated but constructors are not called
//
// All operations implemented in std::allcator_traits specializations
//
// Hardware:
//  model name      : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
//  6 physical cores, hyperthreading enabled
//  64 GB memory: Configured Memory Speed: 2667 MT/s (2x32 GB banks)
// Software:
//  PopOS 21.04
//  clang version 12.0.0-3ubuntu1~21.04.1, GNU STL
//  Target: x86_64-pc-linux-gnu
//  Thread model: posix
//  clang++ -O2 uninitialized_vector_test.cpp
//
// ./a.out
//
// SIZE: 64 MB
// malloc (s): 1.4054e-05
// free (s): 1.07e-07
// vector<int> (s): 0.0229717
// vector<int>.resize(0) (s): 0.00352966
// vector<int> custom allocator default initialized (s): 2.5944e-05
// vector<int>.resize(0) custom allocator default initialized (s): 2.381e-05
// vector<int> custom allocator (s): 2.16e-06
// vector<int>.resize(0) custom allocator (s): 3.248e-06
// vector<Uninitialized<int>> (s): 1.9735e-05
// vector<Uninitialized<int>>.resize(0) (s): 3.418e-06
// vector<Large> (s): 2.4341
// vector<Wrapper<Large>> (s): 9.158e-06
// vector<Large> custom allocator (s): 5.711e-06
//
// Page locked:
// vector<int> custom allocator (s): 0.00990538
// vector<int>.resize(0) custom allocator (s): 0.00427079

#include <sys/mman.h>
#include <unistd.h>

#include <memory>
#include <type_traits>
#include <limits>

template <typename T>
struct BaseAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
};

template <typename T>
struct RawAllocator : BaseAllocator<T> {
    bool init;    // cannot combine with others
    int align;    // not initialised
    bool pinned;  // page locked memory requested; page aligned, not initialised
    RawAllocator(bool i = false, int a = 0, bool p = false)
        : init(i), align(a), pinned(p) {}
    RawAllocator(const RawAllocator&) = default;
};

template <typename T>
struct LazyAllocator : BaseAllocator<T> {
    int align;
    LazyAllocator(int a = 0) : align(a) {}
    LazyAllocator(const LazyAllocator&) = default;
};

template <typename T>
requires(std::is_standard_layout<T>::value && std::is_trivial<
         T>::value) struct std::allocator_traits<RawAllocator<T>> {
    using allocator_type = RawAllocator<T>;
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using difference_type = ptrdiff_t;
    using size_type = std::make_unsigned<difference_type>::type;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;

    template <typename U>
    using rebind_alloc = RawAllocator<U>;
    [[nodiscard]] static constexpr pointer allocate(allocator_type& a,
                                                    size_type n) {
        const size_t size = n * sizeof(value_type);
        if (a.init) {
            return reinterpret_cast<pointer>(calloc(sizeof(value_type), n));
        } else if (a.align > 0) {
            return reinterpret_cast<pointer>(aligned_alloc(a.align, size));
        } else if (a.pinned) {
            void* p = aligned_alloc(getpagesize(), size);
            if(mlock(p, size)) return nullptr;
            return reinterpret_cast<pointer>(p);
        } else {
            return reinterpret_cast<pointer>(malloc(size));
        }
    }
    static constexpr void deallocate(allocator_type& a, pointer p,
                                     size_type n) {
        if (a.pinned) munlock(p, n * sizeof(value_type));
        free(p);
        // unmap(p,n);
    }
    template <typename... ArgsT>
    static constexpr void construct(allocator_type&, pointer, ArgsT&&...) {}
    static constexpr void destroy(allocator_type&, pointer) {}
    static constexpr size_t max_size(const allocator_type&) {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }
};

template <typename T>
requires(std::is_trivially_destructible<T>::value) struct std::allocator_traits<
    LazyAllocator<T>> {
    using allocator_type = LazyAllocator<T>;
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using difference_type = ptrdiff_t;
    using size_type = std::make_unsigned<difference_type>::type;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;

    template <typename U>
    using rebind_alloc = LazyAllocator<U>;
    [[nodiscard]] static constexpr pointer allocate(allocator_type& a,
                                                    size_type n) {
        if (a.align > 0) {
            return reinterpret_cast<pointer>(
                aligned_alloc(a.align, (n * sizeof(value_type))));
        } else {
            return reinterpret_cast<pointer>(malloc(n * sizeof(value_type)));
        }
    }
    static constexpr void deallocate(allocator_type& a, pointer p, size_type) {
        free(p);
    }
    template <typename... ArgsT>
    static constexpr void construct(allocator_type&, pointer, ArgsT&&...) {}
    static constexpr void destroy(allocator_type&, pointer) {}
    static constexpr size_t max_size(const allocator_type&) {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }
};

// MMAP:
// return reinterpret_cast<pointer>(mmap(0, n * sizeof(value_type),
//        PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));