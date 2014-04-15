#include <opencv2/opencv.hpp>

#include <iostream>
#include <numeric>

#include "detector.hpp"
#include "stop_watch.hpp"
#include "algo.hpp"
#include "data.hpp"

namespace {

typedef std::pair<cv::Point2f, cv::Point2f> line_t;
typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
typedef std::pair<cv::Point2f, cv::Point2f> grid_cell;

constexpr const bool SHOW_LINE_SEGMENTS = false;
constexpr const bool SHOW_MERGED_LINE_SEGMENTS = false;
constexpr const bool SHOW_LONG_LINES = false;
constexpr const bool SHOW_FINAL_LINES = false;

constexpr const bool SHOW_INTERSECTIONS = false;
constexpr const bool SHOW_CLUSTERED_INTERSECTIONS = false;
constexpr const bool SHOW_SQUARES = false;
constexpr const bool SHOW_MAX_SQUARES = false;
constexpr const bool SHOW_FINAL_SQUARES = false;
constexpr const bool SHOW_HULL = true;
constexpr const bool SHOW_HULL_FILL = false;
constexpr const bool SHOW_GRID = false;
constexpr const bool SHOW_TL_BR = false;
constexpr const bool SHOW_GRID_NUMBERS= false;
constexpr const bool SHOW_REGRID = false;

constexpr const bool SHOW_CELLS = true;
constexpr const bool SHOW_FINAL_CELLS = true;

void sudoku_binarize(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::medianBlur(gray_image, gray_image, 5);

    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);

    cv::medianBlur(dest_image, dest_image, 5);

    auto structure_elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(dest_image, dest_image, cv::MORPH_DILATE, structure_elem);
}

//TODO This should be improved
void cell_binarize(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    dest_image = gray_image.clone();
    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);

    cv::medianBlur(dest_image, dest_image, 5);

    auto structure_elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(dest_image, dest_image, cv::MORPH_DILATE, structure_elem);
}

constexpr bool almost_equals(float a, float b, float epsilon){
    return a >= (1.0f - epsilon) * b && a <= (1.0f + epsilon) * b;
}

//Manhattan distance between two points
float manhattan_distance(const cv::Point2f& p1, const cv::Point2f& p2){
    auto dx = p1.x - p2.x;
    auto dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

//Euclidean distance between two points
float euclidean_distance(const cv::Point2f& p1, const cv::Point2f& p2){
    return sqrt(manhattan_distance(p1, p2));
}

void draw_square(cv::Mat& dest_image, const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    cv::line(dest_image, p1, p2, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p1, p3, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p1, p4, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p2, p3, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p2, p4, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p3, p4, cv::Scalar(255, 0, 0), 3);
}

cv::Point2f gravity(const std::vector<cv::Point2f>& vec){
    if(vec.size() == 1){
        return vec.front();
    }

    float x = 0.0;
    float y = 0.0;

    for(auto& v : vec){
        x += v.x;
        y += v.y;
    }

    return cv::Point2f(x / vec.size(), y / vec.size());
}

float distance_to_gravity(const cv::Point2f& p, const std::vector<cv::Point2f>& vec){
    return euclidean_distance(p, gravity(vec));
}

bool is_square(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    auto d12 = manhattan_distance(p1, p2);
    auto d13 = manhattan_distance(p1, p3);
    auto d14 = manhattan_distance(p1, p4);
    auto d23 = manhattan_distance(p2, p3);
    auto d24 = manhattan_distance(p2, p4);
    auto d34 = manhattan_distance(p3, p4);

    auto s = std::min(d12, std::min(d13, std::min(d14, std::min(d23, std::min(d24, d34)))));
    auto d = std::max(d12, std::max(d13, std::max(d14, std::max(d23, std::max(d24, d34)))));

    if(almost_equals(d, 2.0f * s, 0.5f)){
        cv::Point2f g((p1.x + p2.x + p3.x + p4.x) / 4.0f, (p1.y + p2.y + p3.y + p4.y) / 4.0f);

        auto d1 = manhattan_distance(p1, g);
        auto d2 = manhattan_distance(p2, g);
        auto d3 = manhattan_distance(p3, g);
        auto d4 = manhattan_distance(p4, g);

        return
            almost_equals(d1, d2, 0.5f) && almost_equals(d1, d3, 0.5f) && almost_equals(d1, d4, 0.5f) &&
            almost_equals(d2, d3, 0.5f) && almost_equals(d2, d4, 0.5f) && almost_equals(d3, d4, 0.5f);
    }

    return false;
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

float angle(const line_t& l){
    return std::fabs(atan((l.second.y - l.first.y) / (l.second.x - l.first.x)) * 180 / CV_PI);
}

std::vector<line_t> detect_lines(const cv::Mat& source_image, cv::Mat& dest_image){
    std::vector<line_t> final_lines;

    //1. Detect lines

    cv::Mat binary_image;
    sudoku_binarize(source_image, binary_image);

    cv::Mat lines_image;
    constexpr const size_t CANNY_THRESHOLD = 60;
    cv::Canny(binary_image, lines_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 5);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(lines_image, lines, 1, CV_PI/180, 50, 50, 12);

    std::cout << lines.size() << " lines found" << std::endl;

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

    auto clusters = vector_transform(lines.begin(), lines.end(), [](auto& v) -> std::vector<cv::Vec4i> {
        return {v};
    });

    bool merged_cluster;
    do {
        merged_cluster = false;

        pairwise_foreach(clusters.begin(), clusters.end(), [&merged_cluster](auto& c1, auto& c2){
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

    auto& max_cluster = *std::max_element(clusters.begin(), clusters.end(), [](auto& lhs, auto& rhs){return lhs.size() < rhs.size();});

    if(SHOW_LINE_SEGMENTS){
        for(auto& l : lines){
            cv::line(dest_image, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 255, 255), 2, CV_AA);
        }

        for(auto& l : max_cluster){
            cv::line(dest_image, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 0, 255), 2, CV_AA);
        }
    }

    std::cout << "Cluster of " << max_cluster.size() << " lines found" << std::endl;

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

    std::cout << "Cluster reduced to " << max_cluster.size() << " lines" << std::endl;

    //4. Transform segments into lines

    std::vector<line_t> long_lines;

    for(auto& l : max_cluster){
        cv::Point2f a(l[0], l[1]);
        cv::Point2f b(l[2], l[3]);

        cv::Vec2f u(b.x - a.x, b.y - a.y);
        u /= norm(u);

        while(b.x < source_image.cols && b.y < source_image.rows && b.x > 0 && b.y > 0){
            b.x += u[0];
            b.y += u[1];
        }

        b.x -= u[0];
        b.y -= u[1];

        while(a.x < source_image.cols && a.y < source_image.rows && a.x > 0 && a.y > 0){
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

                    bool sorted = false;
                    if(std::fabs(theta - 90.0f) < 5.0f){
                        line_t base_line(cv::Point2f(0,0), cv::Point2f(0,100));

                        std::sort(cluster.begin(), cluster.end(), [&base_line](const auto& lhs, const auto& rhs){
                            return approximate_parallel_distance(lhs, base_line) < approximate_parallel_distance(rhs, base_line);
                        });

                        sorted = true;
                    } else if(std::fabs(theta - 0.0f) < 5.0f){
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

                        auto d_first = approximate_parallel_distance(cluster[0], cluster[1]);
                        auto d_first_next = approximate_parallel_distance(cluster[1], cluster[2]);
                        auto mean_first = (total - d_first) / (cluster.size() - 1);

                        auto d_last = approximate_parallel_distance(cluster[cluster.size() - 2], cluster[cluster.size() - 1]);
                        auto d_last_prev = approximate_parallel_distance(cluster[cluster.size() - 3], cluster[cluster.size() - 2]);
                        auto mean_last = (total - d_last) / (cluster.size() - 1);

                        if((d_first < 0.6f * mean_first || d_first < 0.40f * d_first_next) && almost_equals(d_first_next, mean_first, 0.25f)){
                            cluster.erase(cluster.begin(), std::next(cluster.begin()));
                            cleaned_once = cleaned = true;
                        } else if((d_last < 0.6f * mean_last || d_last < 0.40f * d_last_prev) && almost_equals(d_last_prev, mean_last, 0.20f)){
                            cluster.erase(std::prev(cluster.end()), cluster.end());
                            cleaned_once = cleaned = true;
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

    std::cout << "Final lines: " << final_lines.size() << std::endl;

    if(SHOW_FINAL_LINES){
        for(auto& l : final_lines){
            cv::line(dest_image, l.first, l.second, cv::Scalar(0, 255, 0), 2, CV_AA);
        }
    }

    return final_lines;
}

std::vector<cv::Point2f> find_intersections(const std::vector<line_t>& lines, const cv::Mat& source_image){
    std::vector<cv::Point2f> intersections;

    //Detect intersections
    pairwise_foreach(lines.begin(), lines.end(), [&intersections](auto& p1, auto& p2){
        float denom = (p1.first.x - p1.second.x)*(p2.first.y - p2.second.y) - (p1.first.y - p1.second.y)*(p2.first.x - p2.second.x);
        intersections.emplace_back(
            ((p1.first.x*p1.second.y - p1.first.y*p1.second.x)*(p2.first.x - p2.second.x) -
                (p1.first.x - p1.second.x)*(p2.first.x*p2.second.y - p2.first.y*p2.second.x)) / denom,
            ((p1.first.x*p1.second.y - p1.first.y*p1.second.x)*(p2.first.y - p2.second.y) -
                (p1.first.y - p1.second.y)*(p2.first.x*p2.second.y - p2.first.y*p2.second.x)) / denom);
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

std::vector<cv::Point2f> gravity_points(const std::vector<std::vector<cv::Point2f>>& clusters){
    return vector_transform(begin(clusters), end(clusters), [](auto& cluster) -> cv::Point2f {return gravity(cluster);});
}

void draw_points(cv::Mat& dest_image, const std::vector<cv::Point2f>& points, const cv::Scalar& color){
    for(auto& point : points){
        cv::circle(dest_image, point, 1, color, 3);
    }
}

double mse(const square_t& s, const std::vector<cv::Point2f>& points){
    auto& p1 = points[std::get<0>(s)];
    auto& p2 = points[std::get<1>(s)];
    auto& p3 = points[std::get<2>(s)];
    auto& p4 = points[std::get<3>(s)];

    auto d12 = euclidean_distance(p1, p2);
    auto d13 = euclidean_distance(p1, p3);
    auto d14 = euclidean_distance(p1, p4);
    auto d23 = euclidean_distance(p2, p3);
    auto d24 = euclidean_distance(p2, p4);
    auto d34 = euclidean_distance(p3, p4);

    return (d12 + d13 + d14 + d23 + d24 + d34) / 6.0f;
}

double mse(const std::vector<square_t>& squares, const std::vector<cv::Point2f>& points){
    if(squares.empty()){
        return 0.0;
    }

    auto mse_sum = std::accumulate(squares.begin(), squares.end(), 0.0f, [&points](auto& lhs, auto& rhs) -> float {
        return lhs + mse(rhs, points);
    });

    return mse_sum / squares.size();
}

std::vector<square_t> detect_squares(const cv::Mat& source_image, const std::vector<cv::Point2f>& points){
    std::vector<square_t> squares;

    auto limit = std::max(source_image.rows, source_image.cols) / 9.0f;

    for(size_t i = 0; i < points.size(); ++i){
        for(size_t j = i + 1; j < points.size(); ++j){
            auto dij = euclidean_distance(points[i], points[j]);

            if(dij > limit){
                continue;
            }

            for(size_t k = j + 1; k < points.size(); ++k){
                for(size_t l = k + 1; l < points.size(); ++l){
                    if(is_square(points[i], points[j], points[k], points[l])){
                        squares.emplace_back(i,j,k,l);
                    }
                }
            }
        }
    }

    std::cout << "Found " << squares.size() << " squares" << std::endl;

    return squares;
}

std::vector<square_t> find_max_square(const std::vector<square_t>& squares, const std::vector<cv::Point2f>& points){
    std::vector<std::vector<square_t>> square_set;
    square_set.reserve(squares.size());

    for(auto& s1 : squares){
        square_set.push_back({s1});
    }

    bool merged;
    do {
        merged = false;

        pairwise_foreach(begin(square_set), end(square_set), [&points,&merged](auto& ss1, auto& ss2){
            if(ss1.empty() || ss2.empty()){
                return;
            }

            auto d1 = mse(ss1, points);
            auto d2 = mse(ss2, points);

            if(almost_equals(d1, d2, 0.10f)){
                std::copy(ss2.begin(), ss2.end(), std::back_inserter(ss1));
                ss2.clear();

                merged = true;
            }
        });
    } while(merged);

    auto max_square = *std::max_element(square_set.begin(), square_set.end(), [](auto& lhs, auto& rhs){return lhs.size() < rhs.size();});

    std::cout << "Biggest square set size: " << max_square.size() << std::endl;

    return max_square;
}

void remove_unsquare(std::vector<square_t>& squares, const std::vector<cv::Point2f>& points){
    squares.erase(std::remove_if(squares.begin(), squares.end(), [&points](auto& square){
        auto& p1 = points[std::get<0>(square)];
        auto& p2 = points[std::get<1>(square)];
        auto& p3 = points[std::get<2>(square)];
        auto& p4 = points[std::get<3>(square)];

        cv::Point2f g((p1.x + p2.x + p3.x + p4.x) / 4.0f, (p1.y + p2.y + p3.y + p4.y) / 4.0f);

        auto d1 = manhattan_distance(p1, g);
        auto d2 = manhattan_distance(p2, g);
        auto d3 = manhattan_distance(p3, g);
        auto d4 = manhattan_distance(p4, g);

        auto diffs = std::fabs(d1 - d2) + std::fabs(d1 - d3) + std::fabs(d1 - d4) + std::fabs(d2 - d3) + std::fabs(d2 - d4) + std::fabs(d3 - d4);
        auto norm = d1 + d2 + d3 + d4;

        auto squareness = diffs / norm;

        return squareness > 0.33;
    }), squares.end());
}

std::vector<cv::Point2f> compute_hull(const std::vector<cv::Point2f>& points, cv::Mat& dest_image){
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull, false);

    std::cout << "Hull of size " << hull.size() << " found" << std::endl;

    if(SHOW_HULL){
        for(std::size_t i = 0; i < hull.size(); ++i){
            cv::line(dest_image, hull[i], hull[(i+1)%hull.size()], cv::Scalar(128,128,128), 2, CV_AA);
        }
    }

    if(SHOW_HULL_FILL){
        auto hull_i = vector_transform(hull.begin(), hull.end(),
            [](auto& p) -> cv::Point2i {return {static_cast<int>(p.x), static_cast<int>(p.y)};});
        std::vector<decltype(hull_i)> contours = {hull_i};
        cv::fillPoly(dest_image, contours, cv::Scalar(128, 128, 0));
    }

    return hull;
}

std::vector<cv::RotatedRect> compute_grid(const std::vector<cv::Point2f>& hull, cv::Mat& dest_image){
    auto bounding = cv::minAreaRect(hull);

    cv::Point2f bounding_v[4];
    bounding.points(bounding_v);

    if(SHOW_GRID){
        for(std::size_t i = 0; i < 4; ++i){
            cv::line(dest_image, bounding_v[i], bounding_v[(i+1)%4], cv::Scalar(0,0,255), 2, CV_AA);
        }

        for(std::size_t i = 1; i < 9; ++i){
            auto mul = (1.0f / 9.0f) * i;

            cv::Point2f p1(
                bounding_v[0].x + mul * (bounding_v[1].x - bounding_v[0].x),
                bounding_v[0].y + mul * (bounding_v[1].y - bounding_v[0].y));

            cv::Point2f p2(
                bounding_v[3].x + mul * (bounding_v[2].x - bounding_v[3].x),
                bounding_v[3].y + mul * (bounding_v[2].y - bounding_v[3].y));

            cv::line(dest_image, p1, p2, cv::Scalar(0,255,0), 2, CV_AA);

            cv::Point2f p3(
                bounding_v[1].x + mul * (bounding_v[2].x - bounding_v[1].x),
                bounding_v[1].y + mul * (bounding_v[2].y - bounding_v[1].y));

            cv::Point2f p4(
                bounding_v[0].x + mul * (bounding_v[3].x - bounding_v[0].x),
                bounding_v[0].y + mul * (bounding_v[3].y - bounding_v[0].y));

            cv::line(dest_image, p3, p4, cv::Scalar(0,255,0), 2, CV_AA);
        }
    }

    std::size_t tl = 0;
    cv::Point2f origin(0.0f, 0.0f);
    float min_dist = euclidean_distance(bounding_v[tl], origin);

    for(std::size_t i = 1; i < 4; ++i){
        if(euclidean_distance(bounding_v[i], origin) < min_dist){
            min_dist = euclidean_distance(bounding_v[i], origin);
            tl = i;
        }
    }

    std::size_t br = (tl + 2) % 4;

    if(SHOW_TL_BR){
        cv::putText(dest_image, "TL", bounding_v[tl], cv::FONT_HERSHEY_PLAIN, 0.5f, cv::Scalar(0,255,25));
        cv::putText(dest_image, "BR", bounding_v[br], cv::FONT_HERSHEY_PLAIN, 0.5f, cv::Scalar(0,255,25));
    }

    cv::Point2f down_vector;
    cv::Point2f right_vector;

    if(std::fabs(bounding_v[tl].y - bounding_v[(tl+1) % 4].y) > std::fabs(bounding_v[tl].y - bounding_v[(tl+3)%4].y)){
        down_vector.x = std::fabs(bounding_v[tl].x - bounding_v[(tl+1)%4].x);
        down_vector.y = std::fabs(bounding_v[tl].y - bounding_v[(tl+1)%4].y);
        right_vector.x = std::fabs(bounding_v[tl].x - bounding_v[(tl+3)%4].x);
        right_vector.y = std::fabs(bounding_v[tl].y - bounding_v[(tl+3)%4].y);
    } else {
        down_vector.x = std::fabs(bounding_v[tl].x - bounding_v[(tl+3)%4].x);
        down_vector.y = std::fabs(bounding_v[tl].y - bounding_v[(tl+3)%4].y);
        right_vector.x = std::fabs(bounding_v[tl].x - bounding_v[(tl+1)%4].x);
        right_vector.y = std::fabs(bounding_v[tl].y - bounding_v[(tl+1)%4].y);
    }

    auto cell_factor = 1.0f / 9.0f;

    std::vector<cv::RotatedRect> cells(9 * 9);

    for(std::size_t i = 0; i < 9; ++i){
        for(std::size_t j = 0; j < 9; ++j){
            cv::Point2f p_tl(
                bounding_v[tl].x + cell_factor * i * right_vector.x,
                bounding_v[tl].y + cell_factor * j * down_vector.y);

            cv::Point2f p_tr(
                bounding_v[tl].x + cell_factor * (i+1) * right_vector.x,
                bounding_v[tl].y + cell_factor * j * down_vector.y);

            cv::Point2f p_bl(
                bounding_v[tl].x + cell_factor * i * right_vector.x,
                bounding_v[tl].y + cell_factor * (j+1) * down_vector.y);

            cv::Point2f p_br(
                bounding_v[tl].x + cell_factor * (i+1) * right_vector.x,
                bounding_v[tl].y + cell_factor * (j+1) * down_vector.y);

            cv::Point2f p_center(
                (p_br.x + p_tl.x) / 2.0f - cell_factor * 0.1f * right_vector.x,
                (p_br.y + p_tl.y) / 2.0f);

            if(SHOW_CELLS){
                cv::line(dest_image, p_tl, p_tr, cv::Scalar(255,255,0));
                cv::line(dest_image, p_tl, p_bl, cv::Scalar(255,255,0));
                cv::line(dest_image, p_bl, p_br, cv::Scalar(255,255,0));
                cv::line(dest_image, p_br, p_tr, cv::Scalar(255,255,0));
            }

            auto diff_x = p_tr.x - p_tl.x;
            auto diff_y = p_tr.y - p_tl.y;
            float angle = atan(diff_x / diff_y);

            cv::Size p_size(p_tr.x - p_tl.x, p_br.y - p_tr.y);

            cells[i + j * 9] = {p_center, p_size, angle};
        }
    }

    if(SHOW_GRID_NUMBERS){
        for(size_t i = 0; i < cells.size(); ++i){
            cv::putText(dest_image, std::to_string(i + 1), cells[i].center, cv::FONT_HERSHEY_PLAIN, 0.7f, cv::Scalar(0,255,25));
        }
    }

    return cells;
}

void intersects_test(){
    cv::Vec4i a;
    cv::Vec4i b;

    a[0] = 199; a[1] = 277; a[2] = 267; a[3] = 275;
    b[0] = 0; b[1] = 318; b[2] = 0; b[3] = 197;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 0 << std::endl;

    a[0] = 0; a[1] = 318; a[2] = 0; a[3] = 197;
    b[0] = 199; b[1] = 277; b[2] = 267; b[3] = 275;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 0 << std::endl;

    a[0] = 199; a[1] = 277; a[2] = 267; a[3] = 275;
    b[0] = 200; b[1] = 318; b[2] = 200; b[3] = 197;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 1 << std::endl;

    a[0] = 100; a[1] = 100; a[2] = 300; a[3] = 300;
    b[0] = 100; b[1] = 300; b[2] = 300; b[3] = 100;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 1 << std::endl;

    a[0] = 30; a[1] = 242; a[2] = 451; a[3] = 242;
    b[0] = 440; b[1] = 346; b[2] = 440; b[3] = 5;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 1 << std::endl;

    a[0] = 440; a[1] = 346; a[2] = 440; a[3] = 5;
    b[0] = 30; b[1] = 242; b[2] = 451; b[3] = 242;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 1 << std::endl;

    a[0] = 440; a[1] = 5; a[2] = 440; a[3] = 346;
    b[0] = 30; b[1] = 242; b[2] = 451; b[3] = 242;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 1 << std::endl;

    a[0] = 440; a[1] = 346; a[2] = 440; a[3] = 5;
    b[0] = 451; b[1] = 242; b[2] = 30; b[3] = 242;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 1 << std::endl;

    a[0] = 30; a[1] = 242; a[2] = 431; a[3] = 242;
    b[0] = 440; b[1] = 346; b[2] = 440; b[3] = 5;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 0 << std::endl;

    a[0] = 100; a[1] = 200; a[2] = 500; a[3] = 200;
    b[0] = 80; b[1] = 346; b[2] = 450; b[3] = 346;
    std::cout << "Returns: " << intersects(a, b) << std::endl;
    std::cout << "Should:  " << 0 << std::endl;
}

} //end of anonymous namespace

std::vector<cv::RotatedRect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    auto lines = detect_lines(source_image, dest_image);

    auto intersections = find_intersections(lines, source_image);

    if(SHOW_INTERSECTIONS){
        draw_points(dest_image, intersections, cv::Scalar(0,0,255));
    }

    std::cout << intersections.size() << " intersections found" << std::endl;

    auto clusters = cluster(intersections);
    auto points = gravity_points(clusters);

    if(SHOW_CLUSTERED_INTERSECTIONS){
        draw_points(dest_image, points, cv::Scalar(255,0,0));
    }

    std::cout << points.size() << " clustered intersections found" << std::endl;

    //If the detected lines are optimal, the number of intersection is 100
    //In that case, no need to more post processing, just get the grid around
    //the points
    if(points.size() == 100){
        auto hull = compute_hull(points, dest_image);

        return compute_grid(hull, dest_image);
    } else {
        auto squares = detect_squares(source_image, points);

        if(SHOW_SQUARES){
            for(auto& square : squares){
                draw_square(dest_image,
                    points[std::get<0>(square)], points[std::get<1>(square)],
                    points[std::get<2>(square)], points[std::get<3>(square)]
                    );
            }
        }

        if(squares.empty()){
            std::cout << " No square found" << std::endl;

            return {};
        }

        std::cout << squares.size() << " squares found" << std::endl;

        auto max_square = find_max_square(squares, points);

        if(SHOW_MAX_SQUARES){
            for(auto& square : max_square){
                draw_square(dest_image,
                    points[std::get<0>(square)], points[std::get<1>(square)],
                    points[std::get<2>(square)], points[std::get<3>(square)]
                    );
            }
        }

        std::cout << "cluster of " << max_square.size() << " squares found" << std::endl;

        remove_unsquare(max_square, points);

        if(SHOW_FINAL_SQUARES){
            for(auto& square : max_square){
                draw_square(dest_image,
                    points[std::get<0>(square)], points[std::get<1>(square)],
                    points[std::get<2>(square)], points[std::get<3>(square)]
                    );
            }
        }

        std::cout << "Final max_square size: " << max_square.size() << std::endl;

        //Get all the points of the squares
        std::vector<std::size_t> max_square_i;
        for(auto& square : max_square){
            max_square_i.push_back(std::get<0>(square));
            max_square_i.push_back(std::get<1>(square));
            max_square_i.push_back(std::get<2>(square));
            max_square_i.push_back(std::get<3>(square));
        }

        //Removes similar points
        std::sort(max_square_i.begin(), max_square_i.end());
        max_square_i.erase(std::unique(max_square_i.begin(), max_square_i.end()), max_square_i.end());

        //Transform indexes into real points
        auto max_square_points = vector_transform(max_square_i.begin(), max_square_i.end(),
            [&points](auto& i){return points[i];});

        auto hull = compute_hull(max_square_points, dest_image);

        return compute_grid(hull, dest_image);
    }
}

std::vector<cv::Mat> split(const cv::Mat& source_image, cv::Mat& dest_image, const std::vector<cv::RotatedRect>& cells){
    if(cells.empty()){
        std::cout << "No cell provided, no splitting" << std::endl;
        return {};
    }

    std::vector<cv::Mat> cell_mats;
    for(size_t n = 0; n < cells.size(); ++n){

        //TODO In case the angle is too big, just taking the bounding rect
        //will not be good enough

        auto bounding = cells[n].boundingRect();

        bounding.x = std::max(0, bounding.x);
        bounding.y = std::max(0, bounding.y);

        bounding.width = std::min(source_image.cols - bounding.x, bounding.width);
        bounding.height = std::min(source_image.rows - bounding.y, bounding.height);

        if(SHOW_FINAL_CELLS){
            cv::rectangle(dest_image, bounding, cv::Scalar(0, 0, 255), 1, 8, 0);
        }

        cv::Mat rect_mat(source_image, bounding);

        cv::Mat cell_mat(cv::Size(CELL_SIZE, CELL_SIZE), CV_8U);

        if(CELL_EXPAND){
            cv::Mat binary_rect_mat;
            cell_binarize(rect_mat, binary_rect_mat);

            //Fill with white
            cell_mat = cv::Scalar(255,255,255);

            auto top = (CELL_SIZE - binary_rect_mat.rows) / 2;
            auto left = (CELL_SIZE - binary_rect_mat.cols) / 2;

            for(size_t i = 0; i < static_cast<size_t>(binary_rect_mat.rows); ++i){
                for(size_t j = 0; j < static_cast<size_t>(binary_rect_mat.cols); ++j){
                    cell_mat.at<unsigned char>(i+top,j+left) = binary_rect_mat.at<unsigned char>(i, j);
                }
            }
        } else {
            cv::Mat resized_mat;
            cv::resize(rect_mat, resized_mat, cell_mat.size(), 0, 0, cv::INTER_CUBIC);

            cell_binarize(resized_mat, cell_mat);
        }

        cell_mats.emplace_back(std::move(cell_mat));
    }

    if(SHOW_REGRID){
        cv::Mat remat(cv::Size(CELL_SIZE * 9, CELL_SIZE * 9), CV_8U);

        for(size_t n = 0; n < cells.size(); ++n){
            const auto& mat = cell_mats[n];

            size_t ni = n / 9;
            size_t nj = n % 9;

            for(size_t i = 0; i < CELL_SIZE; ++i){
                for(size_t j = 0; j < CELL_SIZE; ++j){
                    remat.at<char>(i+ni * CELL_SIZE,j+nj * CELL_SIZE) = mat.at<char>(i, j);
                }
            }
        }

        cv::namedWindow("Sudoku Final", cv::WINDOW_AUTOSIZE);
        cv::imshow("Sudoku Final", remat);
    }

    return cell_mats;
}