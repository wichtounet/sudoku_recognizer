#include <opencv2/opencv.hpp>

#include <iostream>

#include "stop_watch.hpp"
#include "algo.hpp"

//TODO Use more STL algorithms
//TODO Remove unused stuff
//TODO Improve constness handling

namespace {

constexpr const bool SHOW_LINE_SEGMENTS = false;
constexpr const bool SHOW_MERGED_LINE_SEGMENTS = false;
constexpr const bool SHOW_LONG_LINES = false;
constexpr const bool SHOW_FINAL_LINES = false;

constexpr const bool SHOW_INTERSECTIONS = false;
constexpr const bool SHOW_CLUSTERED_INTERSECTIONS = true;
constexpr const bool SHOW_SQUARES = false;
constexpr const bool SHOW_MAX_SQUARES = false;
constexpr const bool SHOW_FINAL_SQUARES = false;
constexpr const bool SHOW_HULL = true;
constexpr const bool SHOW_GRID = true;

void method_1(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::threshold(gray_image, dest_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void method_2(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::Mat blurred_image;
    cv::GaussianBlur(gray_image, blurred_image, cv::Size(5,5), 0, 0);

    cv::threshold(blurred_image, dest_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void method_3(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
}

void method_4(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::Mat blurred_image = gray_image.clone();
    cv::medianBlur(gray_image, blurred_image, 3);

    cv::adaptiveThreshold(blurred_image, dest_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
}

void method_41(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::Mat blurred_image = gray_image.clone();
    cv::medianBlur(gray_image, blurred_image, 5);

    cv::Mat temp_image;
    cv::adaptiveThreshold(blurred_image, temp_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);

    cv::medianBlur(temp_image, dest_image, 5);

    auto structure_elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(dest_image, dest_image, cv::MORPH_DILATE, structure_elem);
}

struct integral_image {
    size_t* const image;
    const size_t rows;
    const size_t cols;

    integral_image(size_t rows, size_t cols) : image(new size_t[cols * rows]), rows(rows), cols(cols) {
        //Nothing
    }

    ~integral_image(){
        delete[] image;
    }

    size_t space = 0;

    size_t& operator()(int64_t i, int64_t j){
        return image[i * cols + j];
    }
};

void method_5(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat g;
    cv::cvtColor(source_image, g, CV_RGB2GRAY);

    auto cols = g.cols;
    auto rows = g.rows;

    integral_image sI(rows, cols);
    integral_image sIs(rows, cols);

    integral_image I(rows, cols);
    integral_image Is(rows, cols);

    for(int j = 0; j < cols; ++j){
        sI(0,j) = g.at<char>(0,j);
        sIs(0,j) = sI(0,j) * sI(0, j);
    }

    for(int i = 1; i < rows; i++){
        for(int j = 0; j < cols; j++){
            size_t value = g.at<char>(i,j);
            sI(i,j) = sI(i-1,j) + value;
            sIs(i,j) = sIs(i-1,j) + value * value;
        }
    }

    for(int i = 0; i < rows; ++i){
        I(i,0) = sI(i,0);
        Is(i,0) = sIs(i,0);
    }

    for(int i = 0; i < rows; i++){
        for(int j = 1; j < cols; j++){
            I(i, j) = I(i, j-1) + sI(i,j);
            Is(i, j) = Is(i, j-1) + sIs(i,j);
        }
    }

    double k = 0.2;
    int w = 22;
    size_t R = 128;

    dest_image = g.clone();

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; j++){
            auto i1 = std::max(0, i - w/2);
            auto j1 = std::max(0, j - w/2);
            auto i2 = std::min(rows-1,i + w/2);
            auto j2 = std::min(cols-1,j + w/2);
            double area = (i2-i1+1)*(j2-j1+1);

            double diff = 0;
            double sqdiff = 0;

            if(i1 == 0 && j1 == 0){
                diff = I(i2,j2);
                sqdiff = Is(i2,j2);
            } else if(i1 == 0 && j1 != 0){
                diff = I(i2,j2) - I(i2,j1-1);
                sqdiff = Is(i2,j2) - Is(i2,j1-1);
            } else if(i1 != 0 && j1 == 0){
                diff = I(i2,j2) - I(i1-1,j2);
                sqdiff = Is(i2,j2) - Is(i1-1,j2);
            } else {
                double diagsum = I(i2,j2) + I(i1-1, j1-1);
                double idiagsum = I(i2,j1 - 1) + I(i1-1, j2);
                diff = diagsum - idiagsum;
                double sqdiagsum = Is(i2,j2) + Is(i1-1, j1-1);
                double sqidiagsum = Is(i2,j1 - 1) + Is(i1-1, j2);
                sqdiff = sqdiagsum - sqidiagsum;
            }

            double m = static_cast<double>(diff) / area;
            double s = sqrt((sqdiff - diff * diff / area) / (area -1));
            double t = m * (1 + k * (s / R - 1));

            //std::cout << t << std::endl;

            if(g.at<char>(i,j) < t){
                dest_image.at<char>(i,j) = 255;
            } else {
                dest_image.at<char>(i,j) = 0;
            }
        }
    }
}

double ordered_distance(const cv::Vec2f& line){
    auto rho = line[0];
    auto theta = line[1];

    double a = cos(theta);
    double b = sin(theta);
    double x0 = a*rho;
    double y0 = b*rho;

    cv::Point pt1(cvRound(x0 + 1000 * -b), cvRound(y0 + 1000 * a));

    return sqrt(pt1.x * pt1.x + pt1.y * pt1.y);
}

double ordered_distance(const cv::Vec2f& l1, const cv::Vec2f& l2){
    double d1 = ordered_distance(l1);
    double d2 = ordered_distance(l2);
    return d2 - d1;
}

double distance(const cv::Vec2f& l1, const cv::Vec2f& l2){
    double a = cos(l1[1]);
    double b = sin(l1[1]);
    double c1 = l1[0];
    double c2 = l2[0];

    return abs(c2 - c1) / sqrt(a * a + b * b);
}

double average_distance(std::vector<cv::Vec2f>& group, std::vector<std::vector<size_t>>& distance_groups){
    double average = 0;
    for(size_t i = 0 ; i < distance_groups.size() - 1; ++i){
        auto d = distance(group[distance_groups[i][0]], group[distance_groups[i+1][0]]);

        average += d;
    }

    average /= distance_groups.size();
    return average;
}

void draw_line(cv::Mat& dest_image, const cv::Vec2f& line){
    auto rho = line[0];
    auto theta = line[1];

    double a = cos(theta);
    double b = sin(theta);
    double x0 = a*rho;
    double y0 = b*rho;

    cv::Point pt1(cvRound(x0 + 1000 * -b), cvRound(y0 + 1000 * a));
    cv::Point pt2(cvRound(x0 - 1000 * -b), cvRound(y0 - 1000 * a));

    cv::line(dest_image, pt1, pt2, cv::Scalar(0,0,255), 2, CV_AA);
}

constexpr bool almost_equals(float a, float b, float epsilon){
    return a >= (1.0f - epsilon) * b && a <= (1.0f + epsilon) * b;
}

float sq_distance(const cv::Point2f& p1, const cv::Point2f& p2){
    auto dx = p1.x - p2.x;
    auto dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

float distance(const cv::Point2f& p1, const cv::Point2f& p2){
    return sqrt(sq_distance(p1, p2));
}

cv::Rect_<float> to_rect(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    auto d12 = sq_distance(p1, p2);
    auto d13 = sq_distance(p1, p3);
    auto d14 = sq_distance(p1, p4);
    auto d23 = sq_distance(p2, p3);
    auto d24 = sq_distance(p2, p4);
    auto d34 = sq_distance(p3, p4);

    auto s = std::min(d12, std::min(d13, std::min(d14, std::min(d23, std::min(d24, d34)))));
    auto d = 2.0 * s;

    if(almost_equals(d12, d, 0.13f)){
        return {p1, p2};
    }

    if(almost_equals(d13, d, 0.13f)){
        return {p1, p3};
    }

    if(almost_equals(d14, d, 0.13f)){
        return {p1, p4};
    }

    if(almost_equals(d23, d, 0.13f)){
        return {p2, p3};
    }

    if(almost_equals(d24, d, 0.13f)){
        return {p2, p4};
    }

    return {p3, p4};
}

void enlarge(cv::Rect_<float>& rect){
    auto tl = rect.tl();
    auto br = rect.br();

    tl.x = std::max(.0f, tl.x * 0.975f);
    tl.y = std::max(.0f, tl.y * 0.975f);

    br.x *= 1.025;
    br.y *= 1.025;

    rect = cv::Rect_<float>(tl, br);
}

float square_edge(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    auto d12 = distance(p1, p2);
    auto d13 = distance(p1, p3);
    auto d14 = distance(p1, p4);
    auto d23 = distance(p2, p3);
    auto d24 = distance(p2, p4);
    auto d34 = distance(p3, p4);

    return std::min(d12, std::min(d13, std::min(d14, std::min(d23, std::min(d24, d34)))));
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

    x /= vec.size();
    y /= vec.size();

    return cv::Point2f(x, y);
}

float distance_to_gravity(const cv::Point2f& p, const std::vector<cv::Point2f>& vec){
    return distance(p, gravity(vec));
}

float angle(const cv::Point2f& p1, const cv::Point2f& p2){
    return atan(p1.cross(p2) / p1.dot(p2));
}

bool is_square_1(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    auto d12 = sq_distance(p1, p2);
    auto d13 = sq_distance(p1, p3);
    auto d14 = sq_distance(p1, p4);
    auto d23 = sq_distance(p2, p3);
    auto d24 = sq_distance(p2, p4);
    auto d34 = sq_distance(p3, p4);

    auto s = std::min(d12, std::min(d13, std::min(d14, std::min(d23, std::min(d24, d34)))));
    auto d = std::max(d12, std::max(d13, std::max(d14, std::max(d23, std::max(d24, d34)))));

    if(almost_equals(d, 2.0f * s, 0.13f)){
        auto sc = almost_equals(d12, s, 0.13f) + almost_equals(d13, s, 0.13f) + almost_equals(d14, s, 0.13f) + almost_equals(d23, s, 0.13f) + almost_equals(d24, s, 0.13f) + almost_equals(d34, s, 0.13f);
        auto sd = almost_equals(d12, d, 0.13f) + almost_equals(d13, d, 0.13f) + almost_equals(d14, d, 0.13f) + almost_equals(d23, d, 0.13f) + almost_equals(d24, d, 0.13f) + almost_equals(d34, d, 0.13f);

        return sc == 4 && sd == 2;
    }

    return false;
}

bool is_square_2(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4){
    auto d12 = sq_distance(p1, p2);
    auto d13 = sq_distance(p1, p3);
    auto d14 = sq_distance(p1, p4);
    auto d23 = sq_distance(p2, p3);
    auto d24 = sq_distance(p2, p4);
    auto d34 = sq_distance(p3, p4);

    auto s = std::min(d12, std::min(d13, std::min(d14, std::min(d23, std::min(d24, d34)))));
    auto d = std::max(d12, std::max(d13, std::max(d14, std::max(d23, std::max(d24, d34)))));

    if(almost_equals(d, 2.0f * s, 0.5f)){
        cv::Point2f g((p1.x + p2.x + p3.x + p4.x) / 4.0f, (p1.y + p2.y + p3.y + p4.y) / 4.0f);

        auto d1 = sq_distance(p1, g);
        auto d2 = sq_distance(p2, g);
        auto d3 = sq_distance(p3, g);
        auto d4 = sq_distance(p4, g);

        return
            almost_equals(d1, d2, 0.5f) && almost_equals(d1, d3, 0.5f) && almost_equals(d1, d4, 0.5f) &&
            almost_equals(d2, d3, 0.5f) && almost_equals(d2, d4, 0.5f) && almost_equals(d3, d4, 0.5f);
    }

    return false;
}

typedef std::pair<cv::Point2f, cv::Point2f> line_t;

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

bool on_same_line(cv::Vec4i& v1, cv::Vec4i& v2){
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

void detect_lines_2(std::vector<std::pair<cv::Point2f, cv::Point2f>>& final_lines, const cv::Mat& source_image, cv::Mat& dest_image){
    //1. Detect lines

    cv::Mat binary_image;
    method_41(source_image, binary_image);

    cv::Mat lines_image;
    constexpr const size_t CANNY_THRESHOLD = 60;
    cv::Canny(binary_image, lines_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 5);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(lines_image, lines, 1, CV_PI/180, 50, 50, 12);

    std::cout << lines.size() << " lines found" << std::endl;

    //2. Cluster lines

    //Enlarge a bit the lines to enhance the clusters
    for(auto& l : lines){
        cv::Point2f a(l[0], l[1]);
        cv::Point2f b(l[2], l[3]);

        cv::Vec2f u(b.x - a.x, b.y - a.y);
        u *= 0.02;

        b.x += u[0];
        b.y += u[1];

        a.x -= u[0];
        a.y -= u[1];

        l[0] = a.x;
        l[1] = a.y;
        l[2] = b.x;
        l[3] = b.y;
    }

    std::vector<std::vector<cv::Vec4i>> clusters;
    clusters.reserve(lines.size());

    for(auto& v1 : lines){
        clusters.push_back({v1});
    }

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
        auto end = max_cluster.end();

        while(it != end){
            auto& v1 = *it;

            auto nit = std::next(it);
            while(nit != end){
                auto& v2 = *nit;

                if(on_same_line(v1, v2)){
                    cv::Point2f a(v1[0], v1[1]);
                    cv::Point2f b(v1[2], v1[3]);
                    cv::Point2f c(v2[0], v2[1]);
                    cv::Point2f d(v2[2], v2[3]);

                    auto dab = distance(a, b);
                    auto dac = distance(a, c);
                    auto dad = distance(a, d);
                    auto dbc = distance(b, c);
                    auto dbd = distance(b, d);
                    auto dcd = distance(c, d);

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

                    merged = true;

                    nit = max_cluster.erase(nit);
                    end = max_cluster.end();
                } else {
                    ++nit;
                }
            }

            if(merged){
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

    for(auto& l1 : long_lines){
        std::size_t similar = 0;

        float rho_1 = std::fabs(atan((l1.second.y - l1.first.y) / (l1.second.x - l1.first.x)) * 180 / CV_PI);

        for(auto& l2 : long_lines){
            float rho_2 = std::fabs(atan((l2.second.y - l2.first.y) / (l2.second.x - l2.first.x)) * 180 / CV_PI);

            if(std::fabs(rho_2 - rho_1) <= 2.0f){
                ++similar;
            }
        }

        if(similar >= 3){
            final_lines.push_back(l1);
        }
    }

    //6. Filtere extreme outliers

    //20 is the optimal number for a Sudoku
    //If there is less, we cannot do anything to add more
    if(final_lines.size() > 20){
        std::vector<std::vector<line_t>> p_clusters;

        for(auto& l1 : final_lines){
            float rho_1 = std::fabs(atan((l1.second.y - l1.first.y) / (l1.second.x - l1.first.x)) * 180 / CV_PI);

            bool found = false;

            for(auto& cluster : p_clusters){
                for(auto& l2 : cluster){
                    float rho_2 = std::fabs(atan((l2.second.y - l2.first.y) / (l2.second.x - l2.first.x)) * 180 / CV_PI);

                    if(std::fabs(rho_2 - rho_1) <= 10.0f){
                        cluster.push_back(l1);
                        found = true;
                        break;
                    }
                }
                if(found){
                    break;
                }
            }
            if(!found){
                p_clusters.push_back({l1});
            }
        }

        bool cleaned_once = false;

        for(auto& cluster : p_clusters){
            bool cleaned;

            do {
                cleaned = false;

                //10 is the optimal size for a cluster
                if(cluster.size() > 10){
                    std::cout << "cluster " << cluster.size() << std::endl;

                    float theta = std::fabs(atan((cluster.front().second.y - cluster.front().first.y) / (cluster.front().second.x - cluster.front().first.x)) * 180 / CV_PI);

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
}

std::vector<cv::Point2f> find_intersections(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& lines, const cv::Mat& source_image){
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
    intersections.erase(std::remove_if(intersections.begin(), intersections.end(), [&source_image](const auto& p) -> bool {
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
        bool found = false;
        for(auto& cluster : clusters){
            if(distance_to_gravity(i, cluster) < 10.0f){
                cluster.push_back(i);
                found = true;
                break;
            }
        }

        if(!found){
            clusters.push_back({i});
        }
    }

    return clusters;
}

std::vector<cv::Point2f> gravity_points(const std::vector<std::vector<cv::Point2f>>& clusters){
    return vector_transform(begin(clusters), end(clusters), [](auto& cluster) -> cv::Point2f {return gravity(cluster);});
}

void draw_points(cv::Mat& dest_image, const std::vector<cv::Point2f>& points){
    for(auto& point : points){
        cv::circle(dest_image, point, 1, cv::Scalar(0, 0, 255), 3);
    }
}

typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;

//MAX Square
void sudoku_lines_2(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<std::pair<cv::Point2f, cv::Point2f>> lines;
    detect_lines_2(lines, source_image, dest_image);

    auto intersections = find_intersections(lines, source_image);

    auto clusters = cluster(intersections);

    auto points = gravity_points(clusters);

    draw_points(dest_image, points);

    std::cout << points.size() << std::endl;

    float max = 0.0;

    size_t max_i = 0;
    size_t max_j = 0;
    size_t max_k = 0;
    size_t max_l = 0;

    std::vector<square_t> squares;

    for(size_t i = 0; i < points.size(); ++i){
        for(size_t j = i + 1; j < points.size(); ++j){
            auto dij = distance(points[i], points[j]);

            if(dij < max){
                continue;
            }

            for(size_t k = 0; k < points.size(); ++k){
                if(k != j && k != i){
                    for(size_t l = 0; l < points.size(); ++l){
                        if(l != k && l != j && l != i){
                            if(is_square_1(points[i], points[j], points[k], points[l])){
                                max = dij;

                                max_i = i;
                                max_j = j;
                                max_k = k;
                                max_l = l;
                                squares.emplace_back(i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }

    draw_square(dest_image,
        points[max_i], points[max_j],
        points[max_k], points[max_l]
    );

    return;
}

//MAX square with enough inside points
void sudoku_lines_3(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<std::pair<cv::Point2f, cv::Point2f>> lines;
    detect_lines_2(lines, source_image, dest_image);

    auto intersections = find_intersections(lines, source_image);

    auto clusters = cluster(intersections);

    auto points = gravity_points(clusters);

    draw_points(dest_image, points);

    float max = 0.0;

    typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
    std::vector<square_t> squares;

    for(size_t i = 0; i < points.size(); ++i){
        for(size_t j = i + 1; j < points.size(); ++j){
            auto dij = distance(points[i], points[j]);

            if(dij < 0.3 * source_image.cols || dij < 0.3 * source_image.rows){
                continue;
            }

            if(dij < 0.7 * max){
                continue;
            }

            for(size_t k = 0; k < points.size(); ++k){
                if(k != j && k != i){
                    for(size_t l = 0; l < points.size(); ++l){
                        if(l != k && l != j && l != i){
                            if(is_square_1(points[i], points[j], points[k], points[l])){
                                max = std::max(max, dij);

                                squares.emplace_back(i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }

    std::size_t max_inside = 0;
    square_t max_square;

    for(auto& square : squares){
        if(distance(points[std::get<0>(square)], points[std::get<1>(square)]) > 0.8 * max){
            auto rect = to_rect(
                points[std::get<0>(square)], points[std::get<1>(square)],
                points[std::get<2>(square)], points[std::get<3>(square)]);

            enlarge(rect);

            std::size_t inside = 0;
            for(auto& p : points){
                if(p.inside(rect)){
                    ++inside;
                }
            }

            if(inside > max_inside){
                max_square = square;
                max_inside = inside;
            }
        }
    }

    draw_square(dest_image,
        points[std::get<0>(max_square)], points[std::get<1>(max_square)],
        points[std::get<2>(max_square)], points[std::get<3>(max_square)]
    );
}

double mse(const square_t& s, const std::vector<cv::Point2f>& points){
    auto& p1 = points[std::get<0>(s)];
    auto& p2 = points[std::get<1>(s)];
    auto& p3 = points[std::get<2>(s)];
    auto& p4 = points[std::get<3>(s)];

    auto d12 = distance(p1, p2);
    auto d13 = distance(p1, p3);
    auto d14 = distance(p1, p4);
    auto d23 = distance(p2, p3);
    auto d24 = distance(p2, p4);
    auto d34 = distance(p3, p4);

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
            auto dij = distance(points[i], points[j]);

            if(dij > limit){
                continue;
            }

            for(size_t k = j + 1; k < points.size(); ++k){
                for(size_t l = k + 1; l < points.size(); ++l){
                    if(is_square_2(points[i], points[j], points[k], points[l])){
                        squares.emplace_back(i,j,k,l);
                    }
                }
            }
        }
    }

    std::cout << "Found " << squares.size() << " squares" << std::endl;

    return squares;
}

typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
bool equals_one_of(std::size_t a, const square_t& other){
    return a == std::get<0>(other) || a == std::get<1>(other) || a == std::get<2>(other) || a == std::get<3>(other);
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
    auto it = squares.begin();
    auto end = squares.end();

    while(it != end){
        auto& square = *it;

        auto& p1 = points[std::get<0>(square)];
        auto& p2 = points[std::get<1>(square)];
        auto& p3 = points[std::get<2>(square)];
        auto& p4 = points[std::get<3>(square)];

        cv::Point2f g((p1.x + p2.x + p3.x + p4.x) / 4.0f, (p1.y + p2.y + p3.y + p4.y) / 4.0f);

        auto d1 = sq_distance(p1, g);
        auto d2 = sq_distance(p2, g);
        auto d3 = sq_distance(p3, g);
        auto d4 = sq_distance(p4, g);

        auto diffs = std::fabs(d1 - d2) + std::fabs(d1 - d3) + std::fabs(d1 - d4) + std::fabs(d2 - d3) + std::fabs(d2 - d4) + std::fabs(d3 - d4);
        auto norm = d1 + d2 + d3 + d4;

        auto squareness = diffs / norm;

        if(squareness > 0.33){
            it = squares.erase(it);
            end = squares.end();
        } else {
            ++it;
        }
    }
}

//LEGO Algorithm
void sudoku_lines_4(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<std::pair<cv::Point2f, cv::Point2f>> lines;
    detect_lines_2(lines, source_image, dest_image);

    auto intersections = find_intersections(lines, source_image);

    if(SHOW_INTERSECTIONS){
        for(auto& point : intersections){
            cv::circle(dest_image, point, 1, cv::Scalar(0, 0, 255), 3);
        }
    }

    std::cout << intersections.size() << " intersections found" << std::endl;

    auto clusters = cluster(intersections);
    auto points = gravity_points(clusters);

    if(SHOW_CLUSTERED_INTERSECTIONS){
        for(auto& point : points){
            cv::circle(dest_image, point, 1, cv::Scalar(255, 0, 0), 3);
        }
    }

    std::cout << points.size() << " clustered intersections found" << std::endl;

    //If the detected lines are optimal, the number of intersection is 100
    //In that case, no need to more post processing, just get the grid around
    //the points
    if(points.size() == 100){
        std::vector<cv::Point2f> hull;
        cv::convexHull(points, hull, false);

        if(SHOW_HULL){
            for(std::size_t i = 0; i < hull.size(); ++i){
                cv::line(dest_image, hull[i], hull[(i+1)%hull.size()], cv::Scalar(128,128,128), 2, CV_AA);
            }
        }

        auto bounding = cv::minAreaRect(hull);

        cv::Point2f bounding_v[4];
        bounding.points(bounding_v);

        if(SHOW_GRID){
            for(std::size_t i = 0; i < 4; ++i){
                cv::line(dest_image, bounding_v[i], bounding_v[(i+1)%4], cv::Scalar(0,0,255), 2, CV_AA);
            }

            for(std::size_t i = 1; i < 9; ++i){
                auto mul = (1.0f / 9.0f) * i;

                cv::Point2f p1;
                p1.x = bounding_v[0].x + mul * (bounding_v[1].x - bounding_v[0].x);
                p1.y = bounding_v[0].y + mul * (bounding_v[1].y - bounding_v[0].y);

                cv::Point2f p2;
                p2.x = bounding_v[3].x + mul * (bounding_v[2].x - bounding_v[3].x);
                p2.y = bounding_v[3].y + mul * (bounding_v[2].y - bounding_v[3].y);

                cv::line(dest_image, p1, p2, cv::Scalar(0,255,0), 2, CV_AA);

                cv::Point2f p3;
                p3.x = bounding_v[1].x + mul * (bounding_v[2].x - bounding_v[1].x);
                p3.y = bounding_v[1].y + mul * (bounding_v[2].y - bounding_v[1].y);

                cv::Point2f p4;
                p4.x = bounding_v[0].x + mul * (bounding_v[3].x - bounding_v[0].x);
                p4.y = bounding_v[0].y + mul * (bounding_v[3].y - bounding_v[0].y);

                cv::line(dest_image, p3, p4, cv::Scalar(0,255,0), 2, CV_AA);
            }
        }
    } else {
        auto squares = detect_squares(source_image, points);

        if(squares.empty()){
            std::cout << "Failed to detect squares" << std::endl;
            return;
        }

        if(SHOW_SQUARES){
            for(auto& square : squares){
                draw_square(dest_image,
                    points[std::get<0>(square)], points[std::get<1>(square)],
                    points[std::get<2>(square)], points[std::get<3>(square)]
                    );
            }
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

        return;

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
        std::vector<cv::Point2f> max_square_points;
        for(auto& i : max_square_i){
            max_square_points.push_back(points[i]);
        }

        std::vector<cv::Point2f> hull;
        cv::Point2f bounding_v[4];
        cv::RotatedRect bounding;

        bool pruned;
        do {
            pruned = false;

            cv::convexHull(max_square_points, hull, false);

            bounding = cv::minAreaRect(hull);
            bounding.points(bounding_v);

            if(hull.size() > 3){
                for(std::size_t i = 0; i < 4; ++i){
                    auto& p1 = bounding_v[i];
                    auto& p2 = bounding_v[i+1%4];

                    cv::Vec2f line(p2.x - p1.x, p2.y - p1.y);
                    line *= (1.0 / norm(line));
                    auto a = p1;

                    std::size_t n = 0;
                    std::vector<cv::Point2f> close;

                    for(auto& hull_p : max_square_points){
                        cv::Vec2f ap = a - hull_p;
                        cv::Vec2f dist_v = ap - (ap.dot(line)) * line;
                        auto dist = norm(dist_v);

                        if(dist < 5.0f){
                            ++n;
                            close.push_back(hull_p);
                        }
                    }

                    if(!close.empty() && close.size() <= 1){
                        for(auto& p : close){
                            max_square_points.erase(
                                std::remove_if(max_square_points.begin(), max_square_points.end(),
                                [&p](auto& x) -> bool {return x == p; }),
                                max_square_points.end());
                        }

                        pruned = true;
                        break;
                    }
                }
            }
        } while(pruned);

        for(std::size_t i = 0; i < hull.size(); ++i){
            cv::line(dest_image, hull[i], hull[(i+1)%hull.size()], cv::Scalar(128,128,128), 2, CV_AA);
        }

        for(std::size_t i = 0; i < 4; ++i){
            cv::line(dest_image, bounding_v[i], bounding_v[(i+1)%4], cv::Scalar(0,0,255), 2, CV_AA);
        }

        for(std::size_t i = 1; i < 9; ++i){
            auto mul = (1.0f / 9.0f) * i;

            cv::Point2f p1;
            p1.x = bounding_v[0].x + mul * (bounding_v[1].x - bounding_v[0].x);
            p1.y = bounding_v[0].y + mul * (bounding_v[1].y - bounding_v[0].y);

            cv::Point2f p2;
            p2.x = bounding_v[3].x + mul * (bounding_v[2].x - bounding_v[3].x);
            p2.y = bounding_v[3].y + mul * (bounding_v[2].y - bounding_v[3].y);

            cv::line(dest_image, p1, p2, cv::Scalar(0,255,0), 2, CV_AA);

            cv::Point2f p3;
            p3.x = bounding_v[1].x + mul * (bounding_v[2].x - bounding_v[1].x);
            p3.y = bounding_v[1].y + mul * (bounding_v[2].y - bounding_v[1].y);

            cv::Point2f p4;
            p4.x = bounding_v[0].x + mul * (bounding_v[3].x - bounding_v[0].x);
            p4.y = bounding_v[0].y + mul * (bounding_v[3].y - bounding_v[0].y);

            cv::line(dest_image, p3, p4, cv::Scalar(0,255,0), 2, CV_AA);
        }
    }
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

int main(int argc, char** argv ){
    if(argc < 2){
        std::cout << "Usage: binarize <image>..." << std::endl;
        return -1;
    }

    if(argc == 2){
        std::string source_path(argv[1]);

        cv::Mat source_image;
        source_image = cv::imread(source_path.c_str(), 1);

        if (!source_image.data){
            std::cout << "Invalid source_image" << std::endl;
            return -1;
        }

        cv::Mat dest_image = source_image.clone();
        sudoku_lines_4(source_image, dest_image);

        cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
        cv::imshow("Sudoku Grid", dest_image);

        cv::waitKey(0);
    } else {
        for(size_t i = 1; i < static_cast<size_t>(argc); ++i){
            std::string source_path(argv[i]);

            std::cout << source_path << std::endl;

            cv::Mat source_image;
            source_image = cv::imread(source_path.c_str(), 1);

            if (!source_image.data){
                std::cout << "Invalid source_image" << std::endl;
                continue;
            }

            cv::Mat dest_image;
            sudoku_lines_4(source_image, dest_image);

            source_path.insert(source_path.rfind('.'), ".lines");
            imwrite(source_path.c_str(), dest_image);
        }
    }

    return 0;
}