#include <opencv2/opencv.hpp>

#include <iostream>

#include "stop_watch.hpp"
#include "algo.hpp"

namespace {

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

void probabilistic_hough(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat binary_image;
    method_4(source_image, binary_image);

    cv::Mat lines_image;
    cv::Canny(binary_image, lines_image, 50, 200, 3);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(lines_image, lines, 1, CV_PI/360, 50, 100, 10 );

    dest_image = source_image.clone();

    for(auto& l : lines){
        cv::line( dest_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, CV_AA);
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

bool acceptLinePair(const cv::Vec2f& line1, const cv::Vec2f& line2, float minTheta){
    auto theta1 = line1[1];
    auto theta2 = line2[1];

    if(theta1 < minTheta){
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    if(theta2 < minTheta){
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    return abs(theta1 - theta2) > minTheta;
}

std::pair<cv::Point2f, cv::Point2f> lineToPointPair(const cv::Vec2f& line){
    auto cos_t = cos(line[1]);
    auto sin_t = sin(line[1]);
    auto x0 = line[0] * cos_t;
    auto y0 = line[0] * sin_t;

    return std::make_pair(
        cv::Point2f(x0 + 1000.0f * (-sin_t), y0 + 1000.0f * cos_t),
        cv::Point2f(x0 - 1000.0f * (-sin_t), y0 - 1000.0f * cos_t)
    );
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

bool detect_lines(std::vector<cv::Vec2f>& lines, const cv::Mat& source_image){
    cv::Mat binary_image;
    method_41(source_image, binary_image);

    constexpr const size_t CANNY_THRESHOLD = 60;
    cv::Canny(binary_image, binary_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 5);

    HoughLines(binary_image, lines, 1, CV_PI/270, 125, 0, 0);

    if(lines.size() > 250){
        std::cout << "Too many lines" << std::endl;

        lines.clear();

        method_5(source_image, binary_image);

        cv::Canny(binary_image, binary_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 3);

        HoughLines(binary_image, lines, 1, CV_PI/180, 125, 0, 0);

        return lines.size() < 500;
    }

    return true;
}

void draw_lines(const cv::Mat& source_image, cv::Mat& dest_image){
    std::vector<cv::Vec2f> lines;
    detect_lines(lines, source_image);

    dest_image = source_image.clone();

    for(auto& line : lines){
        draw_line(dest_image, line);
    }
}

std::vector<cv::Point2f> find_intersections(const std::vector<cv::Vec2f>& lines){
    std::vector<cv::Point2f> intersections;

    pairwise_foreach(lines.begin(), lines.end(), [&intersections](auto& line1, auto& line2){
        if(acceptLinePair(line1, line2, CV_PI / 32)){
            auto p1 = lineToPointPair(line1);
            auto p2 = lineToPointPair(line2);

            float denom = (p1.first.x - p1.second.x)*(p2.first.y - p2.second.y) - (p1.first.y - p1.second.y)*(p2.first.x - p2.second.x);
            intersections.emplace_back(
                ((p1.first.x*p1.second.y - p1.first.y*p1.second.x)*(p2.first.x - p2.second.x) -
                    (p1.first.x - p1.second.x)*(p2.first.x*p2.second.y - p2.first.y*p2.second.x)) / denom,
                ((p1.first.x*p1.second.y - p1.first.y*p1.second.x)*(p2.first.y - p2.second.y) -
                    (p1.first.y - p1.second.y)*(p2.first.x*p2.second.y - p2.first.y*p2.second.x)) / denom);
        }
    });

    return intersections;
}

std::vector<std::vector<cv::Point2f>> cluster(const std::vector<cv::Point2f>& intersections){
    std::vector<std::vector<cv::Point2f>> clusters;

    for(auto& i : intersections){
        bool found = false;
        for(auto& cluster : clusters){
            if(distance_to_gravity(i, cluster) < 10.0){
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

void filter_outer_points(std::vector<cv::Point2f>& points, const cv::Mat& image){
    points.erase(std::remove_if(points.begin(), points.end(), [&image](auto& i) -> bool {
        return i.x <= 2.0 || i.y <= 2.0 || i.x >= 0.99 * image.cols || i.y >= 0.99 * image.rows;
    }), points.end());
}

void draw_points(cv::Mat& dest_image, const std::vector<cv::Point2f>& points){
    for(auto& point : points){
        cv::circle(dest_image, point, 1, cv::Scalar(0, 0, 255), 3);
    }
}

void sudoku_lines_0(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<cv::Vec2f> lines;
    if(!detect_lines(lines, source_image)){
        return;
    }

    auto intersections = find_intersections(lines);

    auto clusters = cluster(intersections);

    auto points = gravity_points(clusters);

    draw_points(dest_image, points);

    constexpr const size_t PARALLEL_RESOLUTION = 10;
    constexpr const size_t BUCKETS = 180 / PARALLEL_RESOLUTION;
    constexpr const size_t DISTANCE_RESOLUTION = 5;

    std::vector<std::vector<cv::Vec2f>> groups(BUCKETS);

    for(auto& line : lines){
        auto theta = line[1];

        auto theta_deg = static_cast<size_t>(theta * (180 / CV_PI)) % 360;

        auto group = theta_deg / PARALLEL_RESOLUTION;
        groups[group].push_back(line);

        printf("theta:%f theta_deg=%ld group:%ld\n", theta, theta_deg, group);
    }

    std::copy(groups.back().begin(), groups.back().end(), std::back_inserter(groups.front()));
    groups.back().clear();

    for(size_t i = 0; i < groups.size(); ++i){
        auto& group = groups[i];

        if(group.size() < 9){
            continue;
        }

        auto angle_first = i * PARALLEL_RESOLUTION;
        auto angle_last = angle_first + PARALLEL_RESOLUTION - 1;

        std::cout << "size[" << angle_first << "," << angle_last << "]=" << group.size() << std::endl;

        double max_d = 0;
        for(size_t i = 0 ; i < group.size() - 1; ++i){
            for(size_t j = i + 1; j < group.size(); ++j){
                max_d = std::max(distance(group[i],group[j]), max_d);
            }
        }

        auto buckets = static_cast<size_t>(max_d) / DISTANCE_RESOLUTION + 1;
        std::vector<std::vector<std::pair<size_t, size_t>>> pairs(buckets);

        for(size_t i = 0 ; i < group.size() - 1; ++i){
            for(size_t j = i + 1; j < group.size(); ++j){
                auto d = distance(group[i], group[j]);

                pairs[d / DISTANCE_RESOLUTION].emplace_back(i, j);
            }
        }

        unsigned int min_tot = std::min(source_image.rows, source_image.cols);

        for(size_t i = 0 ; i < pairs.size() - 1; ++i){
            auto& pair_group = pairs[i];

            auto d_first = i * DISTANCE_RESOLUTION;
            auto d_last = d_first + DISTANCE_RESOLUTION - 1;

            if(pair_group.size() < 9){
                continue;
            }

            if(d_last < min_tot / 25){
                continue;
            }

            std::cout << i << std::endl;

            std::cout << "pair_group[" << d_first << ", " << d_last << "].size()=" << pair_group.size() << std::endl;

            for(auto& pair : pair_group){
                draw_line(dest_image, group[pair.first]);
                draw_line(dest_image, group[pair.second]);
            }
        }
    }
}

void sudoku_lines_1(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<cv::Vec2f> lines;
    if(!detect_lines(lines, source_image)){
        return;
    }

    auto intersections = find_intersections(lines);

    auto clusters = cluster(intersections);

    auto points = gravity_points(clusters);

    draw_points(dest_image, points);

    constexpr const size_t PARALLEL_RESOLUTION = 10;
    constexpr const size_t BUCKETS = 180 / PARALLEL_RESOLUTION;

    std::vector<std::vector<cv::Vec2f>> groups(BUCKETS);

    for(auto& line : lines){
        auto theta = line[1];

        auto theta_deg = static_cast<size_t>(theta * (180 / CV_PI)) % 360;

        auto group = theta_deg / PARALLEL_RESOLUTION;
        groups[group].push_back(line);

        printf("theta:%f theta_deg=%ld group:%ld\n", theta, theta_deg, group);
    }

    std::copy(groups.back().begin(), groups.back().end(), std::back_inserter(groups.front()));
    groups.back().clear();

    for(size_t g = 0; g < groups.size(); ++g){
        auto& group = groups[g];

        if(group.size() < 10){
            continue;
        }

        std::vector<std::vector<size_t>> distance_groups;

        for(size_t i = 0 ; i < group.size(); ++i){
            auto& line = group[i];

            bool found = false;
            for(auto& distance_group : distance_groups){
                for(auto j : distance_group){
                    if(std::abs(ordered_distance(line, group[j])) < 2){
                        distance_group.push_back(i);
                        found = true;
                        break;
                    }
                }

                if(found){
                    break;
                }
            }

            if(!found){
                distance_groups.emplace_back();
                distance_groups.back().push_back(i);
            }
        }

        if(distance_groups.size() < 10){
            continue;
        }

        size_t ei = 0;
        size_t ej = 0;
        size_t max_d = 0;

        for(size_t i = 0 ; i < distance_groups.size() - 1; ++i){
            for(size_t j = i ; j < distance_groups.size() - 1; ++j){
                auto d = distance(group[distance_groups[i][0]], group[distance_groups[j][0]]);
                if(d > max_d){
                    ei = i;
                    ej = j;
                    max_d = d;
                }
            }
        }

        auto extreme = group[distance_groups[ei][0]];
        std::sort(distance_groups.begin(), distance_groups.end(), [&group,&extreme](auto& lhs, auto& rhs){
            auto d1 = distance(extreme, group[lhs[0]]);
            auto d2 = distance(extreme, group[rhs[0]]);

            return d1 > d2;
        });

        while(true){
            double average = average_distance(group, distance_groups);

            size_t max = 0;
            size_t max_i = 0;

            for(size_t i = 0 ; i < distance_groups.size() - 1; ++i){
                auto d = distance(group[distance_groups[i][0]], group[distance_groups[i+1][0]]);
                if(d > max){
                    max_i = i;
                    max = d;
                }
            }

            std::cout << "average " << average << std::endl;
            std::cout << "max " << max << std::endl;

            if(max > average * 1.20){
                if(max_i == 0){
                    distance_groups.erase(distance_groups.begin() + max_i);
                    continue;
                } else if(max_i+1 == distance_groups.size() - 1){
                    distance_groups.erase(distance_groups.begin() + max_i + 1);
                    continue;
                } else {
                    std::cout << "here" << std::endl;
                    //TODO
                }
            }

            double min = 1111111111;
            size_t min_i = 0;

            for(size_t i = 0 ; i < distance_groups.size() - 1; ++i){
                auto d = distance(group[distance_groups[i][0]], group[distance_groups[i+1][0]]);
                if(d < min){
                    min_i = i;
                    min = d;
                }
            }

            std::cout << "average " << average << std::endl;
            std::cout << "min " << min << std::endl;

            if(min < average * 0.80){
                if(min_i == 0){
                    distance_groups.erase(distance_groups.begin() + min_i);
                    continue;
                } else if(min_i+1 == distance_groups.size() - 1){
                    distance_groups.erase(distance_groups.begin() + min_i + 1);
                    continue;
                } else {
                    auto d_to_next_next = distance(group[distance_groups[min_i][0]], group[distance_groups[min_i+2][0]]);
                    auto d_prev_to_next = distance(group[distance_groups[min_i-1][0]], group[distance_groups[min_i+1][0]]);

                    auto delete_i = d_prev_to_next;
                    auto delete_next = d_to_next_next;

                    if(std::abs(delete_i - average) > std::abs(delete_next - average)){
                        std::cout << "delete i " << std::endl;
                        distance_groups.erase(distance_groups.begin() + min_i);
                        continue;
                    } else {
                        std::cout << "delete next " << std::endl;
                        distance_groups.erase(distance_groups.begin() + min_i + 1);
                        continue;
                    }
                }
            }

            break;
        }

        for(auto& distance_group : distance_groups){
            draw_line(dest_image, group[distance_group[0]]);
        }
    }
}

//MAX Square
void sudoku_lines_2(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<cv::Vec2f> lines;
    if(!detect_lines(lines, source_image)){
        return;
    }

    auto intersections = find_intersections(lines);

    auto clusters = cluster(intersections);

    auto points = gravity_points(clusters);

    filter_outer_points(points, dest_image);

    draw_points(dest_image, points);

    std::cout << points.size() << std::endl;

    float max = 0.0;

    size_t max_i = 0;
    size_t max_j = 0;
    size_t max_k = 0;
    size_t max_l = 0;

    typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
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

    std::vector<cv::Vec2f> lines;
    if(!detect_lines(lines, source_image)){
        return;
    }

    auto intersections = find_intersections(lines);

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

typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
bool equals_one_of(std::size_t a, square_t& other){
    return a == std::get<0>(other) || a == std::get<1>(other) || a == std::get<2>(other) || a == std::get<3>(other);
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

    for(size_t i = 0; i < points.size() - 3; ++i){
        for(size_t j = i + 1; j < points.size() - 2; ++j){
            auto dij = distance(points[i], points[j]);

            if(dij > limit){
                continue;
            }

            for(size_t k = j + 1; k < points.size() - 1; ++k){
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
            auto d1 = mse(ss1, points);
            auto d2 = mse(ss2, points);

            if(almost_equals(d1, d2, 0.07f)){
                //TODO Filter general orientation
                bool found = false;

                for(auto& s1 : ss1){
                    auto result = std::find_if(begin(ss2), end(ss2), [&s1](auto& s2) -> bool {
                        return
                                (equals_one_of(std::get<0>(s1), s2) && equals_one_of(std::get<1>(s1), s2))
                            ||  (equals_one_of(std::get<0>(s1), s2) && equals_one_of(std::get<2>(s1), s2))
                            ||  (equals_one_of(std::get<0>(s1), s2) && equals_one_of(std::get<3>(s1), s2))
                            ||  (equals_one_of(std::get<1>(s1), s2) && equals_one_of(std::get<2>(s1), s2))
                            ||  (equals_one_of(std::get<1>(s1), s2) && equals_one_of(std::get<3>(s1), s2))
                            ||  (equals_one_of(std::get<2>(s1), s2) && equals_one_of(std::get<3>(s1), s2));
                    });

                    if(result != end(ss2)){
                        found = true;
                        break;
                    }
                }

                if(found){
                    std::copy(ss2.begin(), ss2.end(), std::back_inserter(ss1));
                    ss2.clear();

                    merged = true;
                }
            }
        });
    } while(merged);

    auto max_square = *std::max_element(square_set.begin(), square_set.end(), [](auto& lhs, auto& rhs){return lhs.size() < rhs.size();});

    std::cout << "Biggest square set size: " << max_square.size() << std::endl;

    return max_square;
}

void remove_unsquare(std::vector<square_t>& max_square, const std::vector<cv::Point2f>& points){
    auto it = max_square.begin();
    auto end = max_square.end();

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
            it = max_square.erase(it);
            end = max_square.end();
        } else {
            ++it;
        }
    }
}

//LEGO Algorithm
void sudoku_lines_4(const cv::Mat& source_image, cv::Mat& dest_image){
    auto_stop_watch<std::chrono::microseconds> watch("sudoku_lines");

    dest_image = source_image.clone();

    std::vector<cv::Vec2f> lines;
    if(!detect_lines(lines, source_image)){
        return;
    }

    auto intersections = find_intersections(lines);

    auto clusters = cluster(intersections);

    auto points = gravity_points(clusters);

    draw_points(dest_image, points);

    auto squares = detect_squares(source_image, points);

    auto max_square = find_max_square(squares, points);

    remove_unsquare(max_square, points);

    for(auto& square : max_square){
        draw_square(dest_image,
            points[std::get<0>(square)], points[std::get<1>(square)],
            points[std::get<2>(square)], points[std::get<3>(square)]
            );
    }

    std::vector<std::size_t> max_square_i;
    for(auto& square : max_square){
        max_square_i.push_back(std::get<0>(square));
        max_square_i.push_back(std::get<1>(square));
        max_square_i.push_back(std::get<2>(square));
        max_square_i.push_back(std::get<3>(square));
    }

    std::sort(max_square_i.begin(), max_square_i.end());
    max_square_i.erase(std::unique(max_square_i.begin(), max_square_i.end()), max_square_i.end());

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
        //draw_lines(source_image, dest_image);
        //method_41(source_image, dest_image);
        sudoku_lines_4(source_image, dest_image);

        cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
        cv::imshow("Sudoku Grid", dest_image);

        cv::waitKey(0);
    } else {
        for(size_t i = 1; i < static_cast<size_t>(argc); ++i){
            std::string source_path(argv[i]);

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