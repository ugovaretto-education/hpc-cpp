// Author: Ugo Varetto
// Memory aligned and page locked allocators

#include <sys/mman.h>
#include <unistd.h>

#include <memory>
#include <type_traits>

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    int align;
    AlignedAllocator(int a = 64) : align(a) {}
    AlignedAllocator(const AlignedAllocator&) = default;
    AlignedAllocator(AlignedAllocator&&) = default;
};

template <typename T>
struct PinnedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
};

template <typename T>
struct std::allocator_traits<AlignedAllocator<T>> {
    using allocator_type = AlignedAllocator<T>;
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
    using rebind_alloc = AlignedAllocator<U>;
    [[nodiscard]] static constexpr pointer allocate(allocator_type& a,
                                                    size_type n) {
        return reinterpret_cast<pointer>(
            aligned_alloc(a.align, n * sizeof(value_type)));
    }
    static constexpr void deallocate(allocator_type& a, pointer p,
                                     size_type n) {
        free(p);
    }
    template <typename... ArgsT>
    static constexpr void construct(allocator_type&, pointer p,
                                    ArgsT&&... args) {
        new (p) value_type(std::forward<ArgsT>(args)...);
    }
    static constexpr void destroy(allocator_type&, pointer p) {
        p->~value_type();
    }
    static constexpr size_t max_size(const allocator_type&) {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }
};

template <typename T>
struct std::allocator_traits<PinnedAllocator<T>> {
    using allocator_type = PinnedAllocator<T>;
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
    using rebind_alloc = PinnedAllocator<U>;
    [[nodiscard]] static constexpr pointer allocate(allocator_type& a,
                                                    size_type n) {
        const size_t size = n * sizeof(value_type);
        void* p = aligned_alloc(getpagesize(), size);
        if(mlock(p, size)) return nullptr;
        return reinterpret_cast<pointer>(p);
    }
    static constexpr void deallocate(allocator_type& a, pointer p,
                                     size_type n) {
        munlock(p, n * sizeof(value_type));
    }
    template <typename... ArgsT>
    static constexpr void construct(allocator_type&, pointer p,
                                    ArgsT&&... args) {
        new (p) value_type(std::forward<ArgsT>(args)...);
    }
    static constexpr void destroy(allocator_type&, pointer p) {
        p->~value_type();
    }
    static constexpr size_t max_size(const allocator_type&) {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }
};
