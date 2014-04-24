#ifndef TRIG_UTILS_HPP
#define TRIG_UTILS_HPP

//Manhattan distance between two points
template<typename Point>
auto manhattan_distance(const Point& p1, const Point& p2){
    auto dx = p1.x - p2.x;
    auto dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

//Euclidean distance between two points
template<typename Point>
auto euclidean_distance(const Point& p1, const Point& p2){
    return std::sqrt(manhattan_distance(p1, p2));
}

template<typename Point, typename Line>
auto manhattan_distance(const Point& p, const Line& line){
    return
        std::abs((line.second.x - line.first.x) * (line.first.y - p.y) - (line.first.x - p.x) * (line.second.y - line.first.y))
        / sqrt((line.second.x - line.first.x) * (line.second.x - line.first.x) + (line.second.y - line.first.y) * (line.second.y - line.first.y));
}

template<typename Point>
Point gravity(const std::vector<Point>& vec){
    if(vec.size() == 1){
        return vec.front();
    }

    auto x = static_cast<decltype(Point::x)>(0);
    auto y = static_cast<decltype(Point::y)>(0);

    for(auto& v : vec){
        x += v.x;
        y += v.y;
    }

    return Point(x / vec.size(), y / vec.size());
}

template<typename Point>
std::vector<Point> gravity_points(const std::vector<std::vector<Point>>& clusters){
    return vector_transform(begin(clusters), end(clusters), [](auto& cluster) -> Point {return gravity(cluster);});
}

template<typename Point>
float distance_to_gravity(const Point& p, const std::vector<Point>& vec){
    return euclidean_distance(p, gravity(vec));
}

template<typename Rect>
bool overlap(const Rect& a, const Rect& b){
    return
            a.contains({b.x, b.y}) || a.contains({b.x + b.width, b.y})
        ||  a.contains({b.x, b.y + b.height}) || a.contains({b.x + b.width, b.y + b.height})
        ||  b.contains({a.x, a.y}) || b.contains({a.x + a.width, a.y})
        ||  b.contains({a.x, a.y + a.height}) || b.contains({a.x + a.width, a.y + a.height});
}

template<typename Rect>
void ensure_inside(const cv::Mat& image, Rect& rect){
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);

    rect.width = std::min(image.cols - rect.x, rect.width);
    rect.height = std::min(image.rows - rect.y, rect.height);
}

#endif
