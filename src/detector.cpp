#include <opencv2/opencv.hpp>

#include <iostream>
#include <numeric>
#include <array>

#include "detector.hpp"
#include "stop_watch.hpp"
#include "algo.hpp"
#include "data.hpp"
#include "trig_utils.hpp"
#include "image_utils.hpp"

namespace {

typedef std::pair<cv::Point2f, cv::Point2f> line_t;
typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
typedef std::pair<cv::Point2f, cv::Point2f> grid_cell;

constexpr const bool DEBUG = false;

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
constexpr const bool SHOW_TL_BR = true;
constexpr const bool SHOW_GRID_NUMBERS= false;
constexpr const bool SHOW_REGRID = false;
constexpr const bool SHOW_CELLS = true;
constexpr const bool SHOW_CHAR_CELLS = true;

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

void cell_binarize(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    dest_image = gray_image.clone();
    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);

    cv::medianBlur(dest_image, dest_image, 3);
}

constexpr bool almost_equals(float a, float b, float epsilon){
    return a >= (1.0f - epsilon) * b && a <= (1.0f + epsilon) * b;
}

void draw_square(cv::Mat& dest_image, const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    cv::line(dest_image, p1, p2, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p1, p3, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p1, p4, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p2, p3, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p2, p4, cv::Scalar(255, 0, 0), 3);
    cv::line(dest_image, p3, p4, cv::Scalar(255, 0, 0), 3);
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

float angle_rad(const cv::Point2f& a, const cv::Point2f& b){
    return acos(a.dot(b) / (norm(a) * norm(b)));
}

float angle_deg(const cv::Point2f& a, const cv::Point2f& b){
    return angle_rad(a,b) * 180.0f / CV_PI;
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

    IF_DEBUG std::cout << lines.size() << " lines found" << std::endl;

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
    pairwise_foreach(lines.begin(), lines.end(), [&intersections](auto& p1, auto& p2){
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

    IF_DEBUG std::cout << "Found " << squares.size() << " squares" << std::endl;

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

    IF_DEBUG std::cout << "Biggest square set size: " << max_square.size() << std::endl;

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

    IF_DEBUG std::cout << "Hull of size " << hull.size() << " found" << std::endl;

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

    if(true || SHOW_TL_BR){
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
            auto center_x = cells[i].x + cells[i].width / 2.0f;
            auto center_y = cells[i].y + cells[i].height / 2.0f;
            cv::putText(dest_image, std::to_string(i + 1), cv::Point2f(center_x, center_y),
                cv::FONT_HERSHEY_PLAIN, 0.7f, cv::Scalar(0,255,25));
        }
    }

    return cells;
}

std::vector<cv::Point2f> to_float_points(const std::vector<cv::Point>& vec){
    return vector_transform(vec.begin(), vec.end(), [](auto& i){return cv::Point2f(i.x, i.y);});
}

std::vector<cv::Rect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image, std::vector<line_t>& lines){
    auto_stop_watch<std::chrono::microseconds> watch("detect_grid");

    dest_image = source_image.clone();

    cv::Mat dest_image_gray;
    sudoku_binarize(source_image, dest_image_gray);

    lines = detect_lines(source_image, dest_image);

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

std::vector<cv::Mat> split(const cv::Mat& source_image, cv::Mat& dest_image, const std::vector<cv::Rect>& cells, std::vector<line_t>& lines){
    auto_stop_watch<std::chrono::microseconds> watch("split");

    if(cells.empty()){
        std::cout << "No cell provided, no splitting" << std::endl;
        return {};
    }

    cv::Mat source;
    sudoku_binarize(source_image, source);

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

    std::vector<cv::Mat> cell_mats;
    for(size_t n = 0; n < cells.size(); ++n){
        cv::Mat cell_mat(cv::Size(CELL_SIZE, CELL_SIZE), CV_8U);
        cell_mat = cv::Scalar(255, 255, 255);

        auto bounding = ensure_inside(source, cells[n]);

        cv::Mat rect_image_clean(source, bounding);
        cv::Mat rect_image = rect_image_clean.clone();

        cv::Canny(rect_image, rect_image, 4, 12);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(rect_image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        IF_DEBUG std::cout << "n=" << (n+1) << std::endl;
        IF_DEBUG std::cout << contours.size() << " contours found" << std::endl;

        if(!contours.empty()){
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

            if(!candidates.empty()){
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

                    if(fill_factor(cv::Mat(rect_image_clean, rect)) > 0.95f){
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

                if(!candidates.empty()){
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

                        cv::Mat last_rect_mat(source, big_rect);

                        cv::Mat last_mat(cv::Size(dim, dim), last_rect_mat.type());
                        last_mat = cv::Scalar(255,255,255);

                        last_rect_mat.copyTo(last_mat(cv::Rect((dim - rect.width) / 2, (dim - rect.height) / 2, rect.width, rect.height)));

                        cv::resize(last_mat, cell_mat, cell_mat.size(), 0, 0, cv::INTER_CUBIC);

                        auto fill = fill_factor(cell_mat);

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

                            if(min_distance < 50.0f){
                                cell_mat = cv::Scalar(255,255,255);
                            } else {
                                cell_mat = cv::Scalar(255);

                                cv::Mat final_rect(source_image, big_rect);
                                cv::Mat final_rect_binary;
                                cell_binarize(final_rect, final_rect_binary);

                                cv::Mat final_rect_mat(cv::Size(dim, dim), final_rect_binary.type());
                                final_rect_mat = cv::Scalar(255);

                                final_rect_binary.copyTo(final_rect_mat(cv::Rect((dim - rect.width) / 2, (dim - rect.height) / 2, rect.width, rect.height)));

                                cv::resize(final_rect_mat, cell_mat, cell_mat.size(), 0, 0, cv::INTER_CUBIC);

                                if(SHOW_CHAR_CELLS){
                                    cv::rectangle(dest_image, big_rect, cv::Scalar(255, 0, 0), 2);
                                }
                            }
                        } else {
                            cell_mat = cv::Scalar(255,255,255);
                        }
                    }
                }
            }
        }

        cell_mats.emplace_back(std::move(cell_mat));
    }

    if(SHOW_REGRID){
        cv::Mat remat(cv::Size(CELL_SIZE * 9, CELL_SIZE * 9), CV_8U);

        for(size_t n = 0; n < cells.size(); ++n){
            const auto& mat = cell_mats[n];

            size_t ni = n % 9;
            size_t nj = n / 9;

            mat.copyTo(remat(cv::Rect(ni * CELL_SIZE, nj * CELL_SIZE, CELL_SIZE, CELL_SIZE)));
        }

        cv::namedWindow("Sudoku Final", cv::WINDOW_AUTOSIZE);
        cv::imshow("Sudoku Final", remat);
    }

    return cell_mats;
}

} //end of anonymous namespace

std::vector<cv::Mat> detect(const cv::Mat& source_image, cv::Mat& dest_image){
    std::vector<line_t> lines;
    auto cells = detect_grid(source_image, dest_image, lines);
    return split(source_image, dest_image, cells, lines);
}