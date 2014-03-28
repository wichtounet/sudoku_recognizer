
#ifndef ALGO_HPP
#define ALGO_HPP

template<typename Iterator, typename Functor>
void pairwise_foreach(Iterator it, Iterator end, Functor&& fun){
    for(; it != end; ++it){
        for(Iterator next = std::next(it); next != end; ++next){
            fun(*it, *next);
        }
    }
}

template<typename Iterator, typename Functor>
void pairwise_foreach_it(Iterator it, Iterator end, Functor&& fun){
    for(; it != end; ++it){
        for(Iterator next = std::next(it); next != end; ++next){
            fun(it, next);
        }
    }
}

template<typename Iterator, typename Functor>
auto vector_transform(Iterator it, Iterator end, Functor&& fun){
    std::vector<decltype(fun(*it))> transformed;
    std::transform(it, end, std::back_inserter(transformed), std::forward<Functor>(fun));
    return transformed;
}

#endif
