//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <numeric>
#include <array>

#include <opencv2/opencv.hpp>

#include "cpp_utils/algorithm.hpp"

#include "detector.hpp"
#include "data.hpp"
#include "trig_utils.hpp"
#include "image_utils.hpp"

namespace {

constexpr const bool DEBUG = false;

constexpr const bool SHOW_LINE_SEGMENTS = false;
constexpr const bool SHOW_MERGED_LINE_SEGMENTS = false;
constexpr const bool SHOW_LONG_LINES = false;
constexpr const bool SHOW_FINAL_LINES = false;

constexpr const bool SHOW_INTERSECTIONS = false;
constexpr const bool SHOW_CLUSTERED_INTERSECTIONS = false;
constexpr const bool SHOW_HULL = false;
constexpr const bool SHOW_HULL_FILL = false;
constexpr const bool SHOW_TL_BR = false;
constexpr const bool SHOW_CELLS = true;
constexpr const bool SHOW_GRID_NUMBERS= false;
constexpr const bool SHOW_CHAR_CELLS = true;
constexpr const bool SHOW_REGRID = true;

#define IF_DEBUG if(DEBUG)

void sudoku_binarize(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::medianBlur(gray_image, gray_image, 5);

    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);

    cv::medianBlur(dest_image, dest_image, 5);

    auto structure_elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(dest_image, dest_image, cv::MORPH_DILATE, structure_elem);
}

void cell_binarize(const cv::Mat& gray_image, cv::Mat& dest_image){
    dest_image = gray_image.clone();
    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);

    cv::medianBlur(dest_image, dest_image, 3);
}

cv::Point2f find_intersection(const line_t& p1, const line_t& p2){
    float denom = (p1.first.x - p1.second.x)*(p2.first.y - p2.second.y) - (p1.first.y - p1.second.y)*(p2.first.x - p2.second.x);
    return {
            ((p1.first.x*p1.second.y - p1.first.y*p1.second.x)*(p2.first.x - p2.second.x) -
                (p1.first.x - p1.second.x)*(p2.first.x*p2.second.y - p2.first.y*p2.second.x)) / denom,
            ((p1.first.x*p1.second.y - p1.first.y*p1.second.x)*(p2.first.y - p2.second.y) -
                (p1.first.y - p1.second.y)*(p2.first.x*p2.second.y - p2.first.y*p2.second.x)) / denom
        };
}

std::vector<cv::Point2f> find_intersections(const std::vector<line_t>& lines, const cv::Mat& source_image){
    std::vector<cv::Point2f> intersections;

    //Detect intersections
    cpp::pairwise_foreach(lines.begin(), lines.end(), [&intersections](auto& p1, auto& p2){
        intersections.emplace_back(find_intersection(p1, p2));
    });

    constexpr const float CLOSE_INTERSECTION_THRESHOLD = 15.0f;
    constexpr const float CLOSE_INNER_MARGIN = 0.1f;

    //Put the points out of the image but very close to it inside the image
    for(auto& i : intersections){
        if(i.x >= -CLOSE_INTERSECTION_THRESHOLD && i.x < CLOSE_INNER_MARGIN){
            i.x = CLOSE_INNER_MARGIN;
        }

        if(i.y >= -CLOSE_INTERSECTION_THRESHOLD && i.y < CLOSE_INNER_MARGIN){
            i.y = CLOSE_INNER_MARGIN;
        }

        if(i.x >= source_image.cols - CLOSE_INNER_MARGIN && i.x <= source_image.cols + CLOSE_INTERSECTION_THRESHOLD){
            i.x = source_image.cols - CLOSE_INNER_MARGIN;
        }

        if(i.y >= source_image.rows - CLOSE_INNER_MARGIN && i.y <= source_image.rows + CLOSE_INTERSECTION_THRESHOLD){
            i.y = source_image.rows - CLOSE_INNER_MARGIN;
        }
    }

    //Filter bad points
    intersections.erase(std::remove_if(intersections.begin(), intersections.end(), [&source_image](const auto& p){
        return
                std::isnan(p.x) || std::isnan(p.y) || std::isinf(p.x) || std::isinf(p.y)
            ||  p.x < 0 || p.y < 0
            ||  p.x > source_image.cols || p.y > source_image.rows;
    }), intersections.end());

    //Make sure there are no duplicates
    std::sort(intersections.begin(), intersections.end(), [](auto& a, auto& b){ return a.x < b.x && a.y < b.y; });
    intersections.erase(std::unique(intersections.begin(), intersections.end()), intersections.end());

    return intersections;
}

constexpr bool almost_equals(float a, float b, float epsilon){
    return a >= (1.0f - epsilon) * b && a <= (1.0f + epsilon) * b;
}

float approximate_parallel_distance(const line_t& l1, const line_t& l2){
    //Direction vectors
    cv::Point2f l1_d(l1.second.x - l1.first.x, l1.second.y - l1.first.y);
    cv::Point2f l2_d(l2.second.x - l2.first.x, l2.second.y - l2.first.y);

    //Unit vector
    auto l1_n = cv::Vec2f(l1_d.x, l1_d.y) * (1.0f / norm(l1_d));
    auto l2_n = cv::Vec2f(l2_d.x, l2_d.y) * (1.0f / norm(l2_d));

    //Compute all the possible distances

    auto d1 = norm(cv::Vec2f(l1.first - l2.first) - (cv::Vec2f(l1.first - l2.first).dot(l1_n)) * l1_n);
    auto d2 = norm(cv::Vec2f(l1.first - l2.second) - (cv::Vec2f(l1.first - l2.second).dot(l1_n)) * l1_n);

    auto d3 = norm(cv::Vec2f(l2.first - l1.first) - (cv::Vec2f(l2.first - l1.first).dot(l2_n)) * l2_n);
    auto d4 = norm(cv::Vec2f(l2.first - l1.second) - (cv::Vec2f(l2.first - l1.second).dot(l2_n)) * l2_n);

    //Get the mean of the distances

    return (d1 + d2 + d3 + d4) / 4.0f;
}

bool on_same_line(const cv::Vec4i& v1, const cv::Vec4i& v2){
    cv::Point2f a(v1[0] - v1[2], v1[1] - v1[3]);
    cv::Point2f b(v2[0] - v2[2], v2[1] - v2[3]);

    float angle = atan(a.cross(b) / a.dot(b));

    if(std::fabs(angle) < 0.10f){
        //Unit vector of line format by v1
        cv::Vec2f na(a.x, a.y);
        na *= (1.0 / norm(na));

        //A point on line format by v2
        cv::Point2f p(v2[0], v2[1]);

        cv::Vec2f ap = cv::Point2f(v1[0], v1[1]) - p;
        cv::Vec2f dist_v = ap - (ap.dot(na)) * na;
        auto distance = norm(dist_v);

        if(distance < 10.0f){
            return true;
        }
    }

    return false;
}

bool intersects(const cv::Vec4i& v1, const cv::Vec4i& v2){
    float x1 = v1[0], x2 = v1[2], y1 = v1[1], y2 = v1[3];
    float x3 = v2[0], x4 = v2[2], y3 = v2[1], y4 = v2[3];

    if(x1 == x2){
        //Parallel lines
        if(x3 == x4){
            return false;
        }

        auto a2 = (y4 - y3) / (x4 - x3);
        auto b2 = y3 - a2 * x3;

        auto x0 = x1;
        auto y0 = a2 * x0 + b2;

        return
                std::min(x1, x2) <= x0 && std::max(x1, x2) >= x0
            &&  std::min(x3, x4) <= x0 && std::max(x3, x4) >= x0
            &&  std::min(y1, y2) <= y0 && std::max(y1, y2) >= y0
            &&  std::min(y3, y4) <= y0 && std::max(y3, y4) >= y0;
    } else if(x3 == x4){
        auto a1 = (y2 - y1) / (x2 - x1);
        auto b1 = y1 - a1 * x1;

        auto x0 = x3;
        auto y0 = a1 * x0 + b1;

        return
                std::min(x1, x2) <= x0 && std::max(x1, x2) >= x0
            &&  std::min(x3, x4) <= x0 && std::max(x3, x4) >= x0
            &&  std::min(y1, y2) <= y0 && std::max(y1, y2) >= y0
            &&  std::min(y3, y4) <= y0 && std::max(y3, y4) >= y0;
    }

    auto a1 = (y2 - y1) / (x2 - x1);
    auto b1 = y1 - a1 * x1;
    auto a2 = (y4 - y3) / (x4 - x3);
    auto b2 = y3 - a2 * x3;

    //The lines are parallel, consider no intersection
    if(a1 == a2){
        return false;
    }

    auto x0 = -(b1 - b2) / (a1 - a2);

    return
            std::min(x1, x2) < x0 && std::max(x1, x2) > x0
        &&  std::min(x3, x4) < x0 && std::max(x3, x4) > x0;
}

float angle_rad(const cv::Point2f& a, const cv::Point2f& b){
    return acos(a.dot(b) / (norm(a) * norm(b)));
}

float angle_deg(const cv::Point2f& a, const cv::Point2f& b){
    return angle_rad(a,b) * 180.0f / CV_PI;
}

float angle(const line_t& l){
    return std::fabs(atan((l.second.y - l.first.y) / (l.second.x - l.first.x)) * 180 / CV_PI);
}

//Only here for convenience, it is not an efficient way to create vector
template<typename T>
std::vector<T> make_vector(std::initializer_list<T> list){
    return {list};
}

std::vector<std::vector<cv::Point2f>> cluster(const std::vector<cv::Point2f>& intersections){
    std::vector<std::vector<cv::Point2f>> clusters;

    for(auto& i : intersections){
        auto it = std::find_if(clusters.begin(), clusters.end(), [&i](auto& cluster){
            return distance_to_gravity(i, cluster) < 10.0f;
        });

        if(it == clusters.end()){
            clusters.push_back({i});
        } else {
            it->push_back(i);
        }
    }

    return clusters;
}

void draw_points(cv::Mat& dest_image, const std::vector<cv::Point2f>& points, const cv::Scalar& color){
    for(auto& point : points){
        cv::circle(dest_image, point, 1, color, 3);
    }
}

std::vector<cv::Point2f> compute_hull(const std::vector<cv::Point2f>& points, cv::Mat& dest_image){
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull, false);

    IF_DEBUG std::cout << "Hull of size " << hull.size() << " found" << std::endl;

    if(SHOW_HULL){
        for(std::size_t i = 0; i < hull.size(); ++i){
            cv::line(dest_image, hull[i], hull[(i+1)%hull.size()], cv::Scalar(128,128,128), 2, CV_AA);
        }
    }

    if(SHOW_HULL_FILL){
        auto hull_i = cpp::vector_transform(hull.begin(), hull.end(),
            [](auto& p) -> cv::Point2i {return {static_cast<int>(p.x), static_cast<int>(p.y)};});
        std::vector<decltype(hull_i)> contours = {hull_i};
        cv::fillPoly(dest_image, contours, cv::Scalar(128, 128, 0));
    }

    return hull;
}

std::vector<cv::Rect> compute_grid(const std::vector<cv::Point2f>& hull, cv::Mat& dest_image){
    std::vector<cv::Point2f> corners;

    float prev = 0.0;
    for(std::size_t i = 0; i < hull.size(); ++i){
        auto j = (i + 1) % hull.size();
        auto k = (i + 2) % hull.size();

        auto angle = angle_deg(hull[i] - hull[j], hull[j] - hull[k]);

        if(angle > 70.0f && angle < 110.0f){
            corners.push_back(hull[j]);
            prev = 0.0f;
        } else {
            if((angle+prev) > 70.0f && (angle+prev) < 110.0f){
                corners.push_back(find_intersection({hull[(i-1)%hull.size()], hull[i]}, {hull[j], hull[k]}));
                prev = 0.0f;
            } else {
                prev = angle;
            }
        }
    }

    assert(corners.size() == 4);

    std::size_t tl = 0;
    cv::Point2f origin(0.0f, 0.0f);
    float min_dist = euclidean_distance(corners[tl], origin);

    for(std::size_t i = 1; i < 4; ++i){
        if(euclidean_distance(corners[i], origin) < min_dist){
            min_dist = euclidean_distance(corners[i], origin);
            tl = i;
        }
    }

    std::size_t br = (tl + 2) % 4;

    if(SHOW_TL_BR){
        cv::putText(dest_image, "TL", corners[tl], cv::FONT_HERSHEY_PLAIN, 0.5f, cv::Scalar(0,255,25));
        cv::putText(dest_image, "BR", corners[br], cv::FONT_HERSHEY_PLAIN, 0.5f, cv::Scalar(0,255,25));
    }

    cv::Point2f a_vec;
    cv::Point2f b_vec;
    cv::Point2f a_p;
    cv::Point2f b_p;

    if(std::fabs(corners[tl].y - corners[(tl+1) % 4].y) > std::fabs(corners[tl].y - corners[(tl+3)%4].y)){
        a_p = corners[(tl+1) % 4];
        b_p = corners[(tl+2) % 4];
        a_vec = corners[(tl+0) % 4] - corners[(tl + 1) % 4];
        b_vec = corners[(tl+3) % 4] - corners[(tl + 2) % 4];
    } else {
        a_p = corners[(tl+0) % 4];
        b_p = corners[(tl+1) % 4];
        a_vec = corners[(tl + 3) % 4] - corners[(tl+0) % 4];
        b_vec = corners[(tl + 2) % 4] - corners[(tl + 1) % 4];
    }

    std::array<line_t, 10> vectors;

    auto cell_factor = 1.0f / 9.0f;

    for(std::size_t i = 0; i < 10; ++i){
        auto a_a = a_p + a_vec * (cell_factor * i);
        auto b_b = b_p + b_vec * (cell_factor * i);

        vectors[i] = {a_a, b_b};
    }

    std::vector<cv::Rect> cells(9 * 9);

    for(std::size_t i = 0; i < 9; ++i){
        for(std::size_t j = 0; j < 9; ++j){
            auto p1 = vectors[j].first + (vectors[j].second - vectors[j].first) * (cell_factor * i);
            auto p2 = vectors[j].first + (vectors[j].second - vectors[j].first) * (cell_factor * (i + 1));

            auto p3 = vectors[j+1].first + (vectors[j+1].second - vectors[j+1].first) * (cell_factor * i);
            auto p4 = vectors[j+1].first + (vectors[j+1].second - vectors[j+1].first) * (cell_factor * (i + 1));

            std::vector<cv::Point2f> pts({p1, p2, p3, p4});
            cells[i + j * 9] = cv::boundingRect(pts);
        }
    }

    if(SHOW_CELLS){
        for(auto& cell : cells){
            cv::rectangle(dest_image, cell, cv::Scalar(0, 0, 255), 1, 8, 0);
        }
    }

    if(SHOW_GRID_NUMBERS){
        for(size_t i = 0; i < cells.size(); ++i){
            auto center_x = cells[i].x + cells[i].width / 2.0f - 12;
            auto center_y = cells[i].y + cells[i].height / 2.0f + 5;
            cv::putText(dest_image, std::to_string(i + 1), cv::Point2f(center_x, center_y),
                cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0,255,25));
        }
    }

    return cells;
}

std::vector<cv::Point2f> to_float_points(const std::vector<cv::Point>& vec){
    return cpp::vector_transform(vec.begin(), vec.end(), [](auto& i){return cv::Point2f(i.x, i.y);});
}

} //end of anonymous namespace

std::vector<line_t> detect_lines(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat binary_image;
    sudoku_binarize(source_image, binary_image);

    return detect_lines_binary(binary_image, dest_image);
}

std::vector<line_t> detect_lines_binary(const cv::Mat& binary_image, cv::Mat& dest_image){
    std::vector<line_t> final_lines;

    //1. Detect lines

    cv::Mat lines_image;
    constexpr const size_t CANNY_THRESHOLD = 60;
    cv::Canny(binary_image, lines_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 5);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(lines_image, lines, 1, CV_PI/180, 50, 50, 12);

    IF_DEBUG std::cout << lines.size() << " lines found" << std::endl;

    //If Hough failed, there is no sense filtering it
    if(lines.empty()){
        return {};
    }

    //2. Cluster lines

    //Enlarge a bit the lines to enhance the clusters
    for(auto& l : lines){
        cv::Vec2f u(l[2] - l[0], l[3] - l[1]);
        u *= 0.02;

        l[2] += u[0];
        l[3] += u[1];

        l[0] -= u[0];
        l[1] -= u[1];
    }

    auto clusters = cpp::vector_transform(lines.begin(), lines.end(), [](auto& v) -> std::vector<cv::Vec4i> {
        return {v};
    });

    bool merged_cluster;
    do {
        merged_cluster = false;

        cpp::pairwise_foreach(clusters.begin(), clusters.end(), [&merged_cluster](auto& c1, auto& c2){
            for(auto& v1 : c1){
                for(auto& v2 : c2){
                    if(intersects(v1, v2)){
                        std::copy(c2.begin(), c2.end(), std::back_inserter(c1));
                        c2.clear();
                        merged_cluster = true;
                        return;
                    }
                }
            }
        });
    } while(merged_cluster);

    IF_DEBUG std::cout << clusters.size() << " clusters found" << std::endl;

    auto& max_cluster = *std::max_element(clusters.begin(), clusters.end(), [](auto& lhs, auto& rhs){return lhs.size() < rhs.size();});

    if(SHOW_LINE_SEGMENTS){
        for(auto& l : lines){
            cv::line(dest_image, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 255, 255), 2, CV_AA);
        }

        for(auto& l : max_cluster){
            cv::line(dest_image, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 0, 255), 2, CV_AA);
        }
    }

    IF_DEBUG std::cout << "Cluster of " << max_cluster.size() << " lines found" << std::endl;

    //3. Merge line segments into bigger segments

    bool merged;
    do {
        merged = false;

        auto it = max_cluster.begin();

        while(it != max_cluster.end()){
            auto& v1 = *it;

            auto before = max_cluster.size();
            max_cluster.erase(std::remove_if(std::next(it), max_cluster.end(), [&v1](auto& v2){
                if(on_same_line(v1, v2)){
                    cv::Point2f a(v1[0], v1[1]);
                    cv::Point2f b(v1[2], v1[3]);
                    cv::Point2f c(v2[0], v2[1]);
                    cv::Point2f d(v2[2], v2[3]);

                    auto dab = euclidean_distance(a, b);
                    auto dac = euclidean_distance(a, c);
                    auto dad = euclidean_distance(a, d);
                    auto dbc = euclidean_distance(b, c);
                    auto dbd = euclidean_distance(b, d);
                    auto dcd = euclidean_distance(c, d);

                    auto max = std::max(dab, std::max(dac, std::max(dad, std::max(dbc, std::max(dbd, dcd)))));

                    if(dab  == max){
                        //No change in v1
                    } else if(dac == max){
                        v1[2] = v2[0];
                        v1[3] = v2[1];
                    } else if(dad == max){
                        v1[2] = v2[2];
                        v1[3] = v2[3];
                    } else if(dbc == max){
                        v1[0] = v1[2];
                        v1[1] = v1[3];
                        v1[2] = v2[0];
                        v1[3] = v2[1];
                    } else if(dbd == max){
                        v1[0] = v1[2];
                        v1[1] = v1[3];
                        v1[2] = v2[2];
                        v1[3] = v2[3];
                    } else if(dcd == max){
                        v1 = v2;
                    }

                    return true;
                } else {
                    return false;
                }
            }), max_cluster.end());

            if(max_cluster.size() != before){
                merged = true;
                break;
            }

            ++it;
        }
    } while(merged);

    if(SHOW_MERGED_LINE_SEGMENTS){
        for(auto& l : max_cluster){
            cv::line(dest_image, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 0, 255), 2, CV_AA);
        }
    }

    IF_DEBUG std::cout << "Cluster reduced to " << max_cluster.size() << " lines" << std::endl;

    //4. Transform segments into lines

    std::vector<line_t> long_lines;

    for(auto& l : max_cluster){
        cv::Point2f a(l[0], l[1]);
        cv::Point2f b(l[2], l[3]);

        cv::Vec2f u(b.x - a.x, b.y - a.y);
        u /= norm(u);

        while(b.x < binary_image.cols && b.y < binary_image.rows && b.x > 0 && b.y > 0){
            b.x += u[0];
            b.y += u[1];
        }

        b.x -= u[0];
        b.y -= u[1];

        while(a.x < binary_image.cols && a.y < binary_image.rows && a.x > 0 && a.y > 0){
            a.x -= u[0];
            a.y -= u[1];
        }

        a.x += u[0];
        a.y += u[1];

        long_lines.emplace_back(a, b);
    }

    if(SHOW_LONG_LINES){
        for(auto& l : long_lines){
            cv::line(dest_image, l.first, l.second, cv::Scalar(255, 0, 0), 2, CV_AA);
        }
    }

    //5. Filter intruders

    std::copy_if(long_lines.begin(), long_lines.end(), std::back_inserter(final_lines), [&long_lines](auto& l1){
        auto theta1 = angle(l1);

        auto similar = std::count_if(long_lines.begin(), long_lines.end(), [theta1](auto& l2){
            return std::fabs(angle(l2) - theta1) <= 2.0f;
        });

        return similar >= 3;
    });

    //6. Filter extreme outliers

    //20 is the optimal number for a Sudoku
    //If there is less, we cannot do anything to add more
    if(final_lines.size() > 20){
        std::vector<std::vector<line_t>> p_clusters;

        for(auto& l1 : final_lines){
            auto theta1 = angle(l1);

            auto it = std::find_if(p_clusters.begin(), p_clusters.end(), [theta1](const auto& cluster){
                for(auto& l2 : cluster){
                    if(std::fabs(angle(l2) - theta1) <= 10.0f){
                        return true;
                    }
                }

                return false;
            });

            if(it == p_clusters.end()){
                p_clusters.push_back({l1});
            } else {
                it->push_back(l1);
            }
        }

        bool cleaned_once = false;

        for(auto& cluster : p_clusters){
            bool cleaned;

            do {
                cleaned = false;

                //10 is the optimal size for a cluster
                if(cluster.size() > 10){
                    auto theta = angle(cluster.front());

                    bool vertical = std::fabs(theta - 90.0f) < 5.0f;
                    bool horizontal = std::fabs(theta - 0.0f) < 5.0f;

                    bool sorted = false;
                    if(vertical){
                        line_t base_line(cv::Point2f(0,0), cv::Point2f(0,100));

                        std::sort(cluster.begin(), cluster.end(), [&base_line](const auto& lhs, const auto& rhs){
                            return approximate_parallel_distance(lhs, base_line) < approximate_parallel_distance(rhs, base_line);
                        });

                        sorted = true;
                    } else if(horizontal){
                        line_t base_line(cv::Point2f(0,0), cv::Point2f(100,0));

                        std::sort(cluster.begin(), cluster.end(), [&base_line](const auto& lhs, const auto& rhs){
                            return approximate_parallel_distance(lhs, base_line) < approximate_parallel_distance(rhs, base_line);
                        });

                        sorted = true;
                    } else {
                        //TODO We need a rotation mechanism to handle such lines
                        //Or create a base line with the correct angle
                    }

                    if(sorted){
                        auto total = 0.0f;
                        for(size_t i = 0; i < cluster.size() - 1; ++i){
                            total += approximate_parallel_distance(cluster[i], cluster[i+1]);
                        }

                        auto& first = cluster[0];
                        auto& second = cluster[1];
                        auto& third = cluster[2];

                        auto d12 = approximate_parallel_distance(first, second);
                        auto d23 = approximate_parallel_distance(second, third);
                        auto mean_first = (total - d12) / (cluster.size() - 1);

                        if((d12 < 0.6f * mean_first || d12 < 0.40f * d23) && almost_equals(d23, mean_first, 0.25f)){
                            auto inter = find_intersection(first, second);

                            if(inter.x > 0 && inter.y > 0 && inter.x < binary_image.cols && inter.y < binary_image.rows){
                                second.first = gravity(make_vector({first.first, second.first}));
                                second.second = gravity(make_vector({first.second, second.second}));

                                if(horizontal){
                                    second.first.y *= 0.95;
                                    second.second.y *= 0.95;
                                } else if(vertical){
                                    second.first.x *= 0.95;
                                    second.second.x *= 0.95;
                                }
                            }

                            cluster.erase(cluster.begin(), std::next(cluster.begin()));
                            cleaned_once = cleaned = true;
                        } else {
                            auto& last = cluster.back();
                            auto& pen = cluster[cluster.size() - 2];
                            auto& ante = cluster[cluster.size() - 3];

                            auto dlp = approximate_parallel_distance(pen, last);
                            auto dpa = approximate_parallel_distance(ante, pen);
                            auto mean_last = (total - dlp) / (cluster.size() - 1);

                            if((dlp < 0.6f * mean_last || dlp < 0.40f * dpa) && almost_equals(dpa, mean_last, 0.20f)){
                                auto inter = find_intersection(pen, last);

                                if(inter.x > 0 && inter.y > 0 && inter.x < binary_image.cols && inter.y < binary_image.rows){
                                    pen.first = gravity(make_vector({last.first, pen.first}));
                                    pen.second = gravity(make_vector({last.second, pen.second}));

                                    if(horizontal){
                                        pen.first.y *= 1.005;
                                        pen.second.y *= 1.005;
                                    } else if(vertical){
                                        pen.first.x *= 1.005;
                                        pen.second.x *= 1.005;
                                    }
                                }

                                cluster.erase(std::prev(cluster.end()), cluster.end());
                                cleaned_once = cleaned = true;
                            }
                        }
                    } else {
                        std::cout << "Failed to sort" << std::endl;
                    }
                }
            } while(cleaned);
        }

        if(cleaned_once){
            final_lines.clear();
            for(auto& cluster : p_clusters){
                std::copy(cluster.begin(), cluster.end(), std::back_inserter(final_lines));
            }
        }
    }

    IF_DEBUG std::cout << "Final lines: " << final_lines.size() << std::endl;

    if(final_lines.size() == 20){
        IF_DEBUG std::cout << "LINES PERFECT" << std::endl;
    }

    if(SHOW_FINAL_LINES){
        for(auto& l : final_lines){
            cv::line(dest_image, l.first, l.second, cv::Scalar(0, 255, 0), 2, CV_AA);
        }
    }

    return final_lines;
}

std::vector<cv::Rect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image, std::vector<line_t>& lines){
    if(lines.empty()){
        return {};
    }

    auto intersections = find_intersections(lines, source_image);

    if(SHOW_INTERSECTIONS){
        draw_points(dest_image, intersections, cv::Scalar(0,0,255));
    }

    IF_DEBUG std::cout << intersections.size() << " intersections found" << std::endl;

    auto clusters = cluster(intersections);
    auto points = gravity_points(clusters);

    if(SHOW_CLUSTERED_INTERSECTIONS){
        draw_points(dest_image, points, cv::Scalar(255,0,0));
    }

    IF_DEBUG std::cout << points.size() << " clustered intersections found" << std::endl;

    //If the detected lines are optimal, the number of intersection is 100
    //In that case, no need to more post processing, just get the grid around
    //the points
    if(points.size() == 100){
        IF_DEBUG std::cout << "POINTS PERFECT" << std::endl;

        auto hull = compute_hull(points, dest_image);

        return compute_grid(hull, dest_image);
    } else {
        cv::Mat dest_image_gray;
        sudoku_binarize(source_image, dest_image_gray);

        std::size_t CANNY = 150;
        cv::Canny(dest_image_gray, dest_image_gray, CANNY, CANNY * 4, 5);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(dest_image_gray, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));

        std::size_t max_c = 0;

        for(std::size_t i = 1; i < contours.size(); ++i){
            if(cv::contourArea(contours[i]) > cv::contourArea(contours[max_c])){
                max_c = i;
            }
        }

        auto clusters = cluster(to_float_points(contours[max_c]));
        auto points = gravity_points(clusters);

        auto hull = compute_hull(points, dest_image);

        return compute_grid(hull, dest_image);
    }
}

sudoku_grid split(const cv::Mat& source_image, cv::Mat& dest_image, const std::vector<cv::Rect>& cells, std::vector<line_t>& lines){
    sudoku_grid grid;
    grid.source_image = source_image.clone(); //TODO constructor

    if(cells.empty()){
        IF_DEBUG std::cout << "No cell provided, no splitting" << std::endl;
        return grid;
    }

    cv::Mat source;
    if(source_image.type() == CV_8U){
        source = source_image;
    } else {
        sudoku_binarize(source_image, source);
    }

    if(lines.size() > 20){
        lines.erase(std::remove_if(lines.begin(), lines.end(), [&cells](auto& line){
            std::size_t near = 0;
            for(auto& rect : cells){
                if(manhattan_distance(cv::Point2f(rect.x, rect.y), line) < 10.0f){
                    ++near;
                }
            }

            return !near;
        }), lines.end());
    }

    for(auto& line : lines){
        cv::line(source, line.first, line.second, cv::Scalar(255, 255, 255), 7, CV_AA);
    }

    //TODO Clean

    for(size_t n = 0; n < cells.size(); ++n){
        //Create a new cell
        grid.cells.emplace_back();
        auto& cell = grid.cells.back();

        cell.final_mat = cv::Mat(cv::Size(CELL_SIZE, CELL_SIZE), source.type());
        cell.bounding = ensure_inside(source, cells[n]);

        auto& cell_mat = cell.final_mat;
        const auto& bounding = cell.bounding;

        cell_mat = cv::Scalar(255);

        cv::Mat rect_image_clean(source, bounding);
        cv::Mat rect_image = rect_image_clean.clone();

        cv::Canny(rect_image, rect_image, 4, 12);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(rect_image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        IF_DEBUG std::cout << "n=" << (n+1) << std::endl;
        IF_DEBUG std::cout << contours.size() << " contours found" << std::endl;

        auto width = rect_image.cols * 0.75f;
        auto height = rect_image.rows * 0.75f;

        std::vector<cv::Rect> candidates;

        //Get all interesting candidates
        for(std::size_t i = 0; i < contours.size(); ++i){
            auto rect = cv::boundingRect(contours[i]);

            if(rect.height <= height && rect.width <= width && std::find(candidates.begin(), candidates.end(), rect) == candidates.end()){
                candidates.push_back(rect);
            }
        }

        IF_DEBUG std::cout << candidates.size() << " filtered candidates found" << std::endl;

        bool merged;
        do {
            merged = false;
            for(std::size_t i = 0; i < candidates.size() && !merged; ++i){
                auto& a = candidates[i];

                for(std::size_t j = i + 1; j < candidates.size() && !merged; ++j){
                    auto& b = candidates[j];

                    if(overlap(a, b)){
                        std::vector<cv::Point2i> all_points({
                            {a.x, a.y},{a.x + a.width, a.y},{a.x,a.y + a.height},{a.x + a.width, a.y + a.height},
                            {b.x, b.y},{b.x + b.width, b.y},{b.x, b.y + b.height},{b.x + b.width, b.y + b.height}});

                        auto result = cv::boundingRect(all_points);

                        if(result.height > height || result.width > width || result.width > 2.0 * result.height){
                            continue;
                        }

                        a = result;

                        candidates.erase(candidates.begin() + j);

                        merged = true;
                    }
                }
            }
        } while(merged);

        IF_DEBUG std::cout << candidates.size() << " merged candidates found" << std::endl;

        candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&rect_image_clean,height,width](auto rect){
            ensure_inside(rect_image_clean, rect);

            auto dim = std::max(rect.width, rect.height);
            cv::Mat tmp_rect(rect_image_clean, rect);
            cv::Mat tmp_square(cv::Size(dim, dim), tmp_rect.type());
            tmp_square = cv::Scalar(255,255,255);
            tmp_rect.copyTo(tmp_square(cv::Rect((dim - rect.width) / 2, (dim - rect.height) / 2, rect.width, rect.height)));

            if(fill_factor(tmp_square) > 0.95f){
                return true;
            }

            if(rect.height < 10 || rect.width < 5 || rect.height > height || rect.width > width){
                return true;
            }

            //Horizontal
            if(rect.width > 1.5 * rect.height){
                if(rect.y < 5 || rect.y + rect.height > rect_image_clean.rows - 5){
                    return true;
                }
            }

            //Vertical
            if(rect.height > 2.0 * rect.width && rect.width < 8){
                if(rect.x < 5 || rect.x + rect.width > rect_image_clean.cols - 5){
                    return true;
                }
            }

            return false;
        }), candidates.end());

        IF_DEBUG std::cout << candidates.size() << " filtered  bounding rect found" << std::endl;

        std::size_t max_i = 0;
        decltype(bounding.area()) max = 0;

        for(std::size_t i = 0; i < candidates.size(); ++i){
            auto& rect = candidates[i];
            auto area = rect.area();

            if(area > max){
                max_i = i;
                max = area;
            }
        }

        if(max > 100){
            auto& rect = candidates[max_i];

            rect.x -= 2;
            rect.width += 4;
            rect.y += 1;
            rect.height += 2;

            ensure_inside(rect_image, rect);

            IF_DEBUG std::cout << "Final rect " << rect << std::endl;

            auto big_rect = rect;
            big_rect.x += bounding.x;
            big_rect.y += bounding.y;

            auto dim = std::max(rect.width, rect.height);

            //Extract the cell from the source image (binary)
            const cv::Mat final_rect(source, big_rect);

            //Make the image square
            cv::Mat final_square(cv::Size(dim, dim), final_rect.type());
            final_square = cv::Scalar(255,255,255);
            final_rect.copyTo(final_square(cv::Rect((dim - rect.width) / 2, (dim - rect.height) / 2, rect.width, rect.height)));

            auto fill = fill_factor(final_square);

            IF_DEBUG std::cout << "\tfill_factor=" << fill << std::endl;

            if(fill < 0.95f){
                auto min_distance = 1000000.0f;

                if(fill > 0.85f){
                    for(auto& line : lines){
                        auto local_distance = 0.0f;

                        local_distance += manhattan_distance(cv::Point2f(big_rect.x, big_rect.y), line);
                        local_distance += manhattan_distance(cv::Point2f(big_rect.x + big_rect.width, big_rect.y), line);
                        local_distance += manhattan_distance(cv::Point2f(big_rect.x, big_rect.y + big_rect.height), line);
                        local_distance += manhattan_distance(cv::Point2f(big_rect.x + big_rect.width, big_rect.y + big_rect.height), line);

                        min_distance = std::min(min_distance, local_distance);
                    }
                }

                IF_DEBUG std::cout << "\tmin_distance=" << min_distance << std::endl;

                if(min_distance >= 50.0f){
                    //Resize the square to the cell size
                    cv::Mat big_square(cv::Size(CELL_SIZE, CELL_SIZE), final_square.type());
                    cv::resize(final_square, big_square, big_square.size(), 0, 0, cv::INTER_CUBIC);

                    //Binarize again because resize goes back to GRAY
                    cell_binarize(big_square, cell_mat);

                    if(SHOW_CHAR_CELLS){
                        cv::rectangle(dest_image, big_rect, cv::Scalar(255, 0, 0), 2);
                    }
                }
            }

            if(fill_factor(cell_mat) == 1.0f){
                cell.m_empty = true;
            } else {
                cell.m_empty = false;
            }
        }
    }

    if(SHOW_REGRID){
        cv::Mat remat(cv::Size(CELL_SIZE * 9, CELL_SIZE * 9), grid(0,0).final_mat.type());

        for(std::size_t i = 0; i < 9; ++i){
            for(std::size_t j = 0; j < 9; ++j){
                const auto& cell = grid(i, j);
                const auto& mat = cell.final_mat;

                mat.copyTo(remat(cv::Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE)));
            }
        }

        cv::namedWindow("Sudoku Final", cv::WINDOW_AUTOSIZE);
        cv::imshow("Sudoku Final", remat);
    }

    return grid;
}

sudoku_grid detect(const cv::Mat& source_image, cv::Mat& dest_image){
    dest_image = source_image.clone();

    auto lines = detect_lines(source_image, dest_image);
    auto cells = detect_grid(source_image, dest_image, lines);
    return split(source_image, dest_image, cells, lines);
}

sudoku_grid detect_binary(const cv::Mat& source_image, cv::Mat& dest_image){
    dest_image = source_image.clone();

    auto lines = detect_lines_binary(source_image, dest_image);
    auto cells = detect_grid(source_image, dest_image, lines);
    return split(source_image, dest_image, cells, lines);
}
