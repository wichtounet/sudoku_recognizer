#include <opencv2/opencv.hpp>

#include <iostream>

#include "stop_watch.hpp"

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

    cv::Mat blurred_image;
    cv::medianBlur(gray_image, blurred_image, 3);

    cv::adaptiveThreshold(blurred_image, dest_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
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

    auto m = [&I,w](size_t x, size_t y) -> double {
        return (I(x+w/2, y + w/2) + I(x - w/2,y - w/2) - I(x + w/2,y - w/2) - I(x - w/2, y + w/2)) / (w * w);
    };

    auto s = [&Is,w,g,&m](size_t x, size_t y) -> double {
        auto s0 = (Is(x+w/2, y + w/2) + Is(x - w/2,y - w/2) - Is(x + w/2,y - w/2) - Is(x - w/2, y + w/2)) / (w * w);
        return s0 - m(x, y) * m(x, y);
    };

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; j++){
            auto i1 = std::max(0, i - w/2);
            auto j1 = std::max(0, j - w/2);
            auto i2 = std::min(rows-1,i + w/2);
            auto j2 = std::min(cols-1,j + w/2);
            double area = (i2-i1+1)*(j2-j1+1);

            //double t = m(i,j) * (1 + k * (s(i,j) / R - 1));

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
                dest_image.at<char>(i,j) = 0;
            } else {
                dest_image.at<char>(i,j) = 255;
            }
        }
    }
}

void hough(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat binary_image;
    method_4(source_image, binary_image);

    cv::Mat lines_image;
    cv::Canny(binary_image, lines_image, 50, 200, 3);

    std::vector<cv::Vec2f> lines;
    HoughLines(lines_image, lines, 1, CV_PI/180, 125, 0, 0 );

    dest_image = source_image.clone();

    for(size_t i = 0; i < lines.size(); i++){
        auto rho = lines[i][0];
        auto theta = lines[i][1];
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a*rho;
        double y0 = b*rho;

        cv::Point pt1(cvRound(x0 + 1000 * -b), cvRound(y0 + 1000 * a));
        cv::Point pt2(cvRound(x0 - 1000 * -b), cvRound(y0 - 1000 * a));

        cv::line(dest_image, pt1, pt2, cv::Scalar(0,0,255), 3, CV_AA);
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

double ordered_average_distance(std::vector<cv::Vec2f>& group, std::vector<std::vector<size_t>>& distance_groups){
    double average = 0;
    for(size_t i = 0 ; i < distance_groups.size() - 1; ++i){
        auto d = ordered_distance(group[distance_groups[i][0]], group[distance_groups[i+1][0]]);

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

bool almost_better(float a, float b){
    return a >= 0.87f * b && a <= 1.13f * b;
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

    if(almost_better(d12, d)){
        return {p1, p2};
    }

    if(almost_better(d13, d)){
        return {p1, p3};
    }

    if(almost_better(d14, d)){
        return {p1, p4};
    }

    if(almost_better(d23, d)){
        return {p2, p3};
    }

    if(almost_better(d24, d)){
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

std::pair<cv::Point2f, cv::Point2f> lineToPointPair(cv::Vec2f line){
    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

    return std::make_pair(
        cv::Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t),
        cv::Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t)
    );
}

cv::Point2f gravity(const std::vector<cv::Point2f>& vec){
    if(vec.size() == 1){
        return vec[0];
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

    if(almost_better(d, 2.0f * s)){
        auto sc = almost_better(d12, s) + almost_better(d13, s) + almost_better(d14, s) + almost_better(d23, s) + almost_better(d24, s) + almost_better(d34, s);
        auto sd = almost_better(d12, d) + almost_better(d13, d) + almost_better(d14, d) + almost_better(d23, d) + almost_better(d24, d) + almost_better(d34, d);

        return sc == 4 && sd == 2;
    }

    return false;
}
bool almost_better_sq(float a, float b){
    /*std::cout << "sq" << std::endl;
    std::cout << "a=" << a << std::endl;
    std::cout << "b=" << b << std::endl;
    std::cout << "[]" << 0.75f * b << std::endl;
    std::cout << "[]" << 1.25f * b << std::endl;*/
    return a >= 0.8f * b && a <= 1.2f * b;
}

bool almost_better_sq_h(float a, float b){
    /*std::cout << "sq_h" << std::endl;
    std::cout << "a=" << a << std::endl;
    std::cout << "b=" << b << std::endl;
    std::cout << "[]" << 0.75f * b << std::endl;
    std::cout << "[]" << 1.25f * b << std::endl;*/
    return a >= 0.5f * b && a <= 1.5f * b;
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

    if(almost_better_sq_h(d, 2.0f * s)){
        cv::Point2f g((p1.x + p2.x + p3.x + p4.x) / 4.0f, (p1.y + p2.y + p3.y + p4.y) / 4.0f);

        auto d1 = sq_distance(p1, g);
        auto d2 = sq_distance(p2, g);
        auto d3 = sq_distance(p3, g);
        auto d4 = sq_distance(p4, g);

        return
            almost_better_sq_h(d1, d2) && almost_better_sq_h(d1, d3) && almost_better_sq_h(d1, d4) &&
            almost_better_sq_h(d2, d3) && almost_better_sq_h(d2, d4) && almost_better_sq_h(d3, d4);
    }

    return false;
}

bool detect_lines(std::vector<cv::Vec2f>& lines, const cv::Mat& source_image){
    cv::Mat binary_image;
    method_4(source_image, binary_image);

    constexpr const size_t CANNY_THRESHOLD = 50;
    cv::Canny(binary_image, binary_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 5);

    HoughLines(binary_image, lines, 1, CV_PI/180, 125, 0, 0);

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
    for( size_t i = 0; i < lines.size() - 1; i++ ){
        for(size_t j = i + 1; j < lines.size(); j++){
            auto& line1 = lines[i];
            auto& line2 = lines[j];

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
        }
    }
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
    std::vector<cv::Point2f> points;
    for(auto& cluster : clusters){
        points.push_back(gravity(cluster));
    }
    return points;

}

void filter_outer_points(std::vector<cv::Point2f>& points, const cv::Mat& image){
    auto it = points.begin();
    auto end = points.end();
    while(it != end){
        auto& i = *it;

        if(i.x <= 2.0 || i.y <= 2.0 || i.x >= 0.99 * image.cols || i.y >= 0.99 * image.rows){
            it = points.erase(it);
            end = points.end();
        } else {
            ++it;
        }
    }
}

void draw_points(cv::Mat& dest_image, const std::vector<cv::Point2f>& points){
    for(auto& point : points){
        cv::circle(dest_image, point, 1, cv::Scalar(0, 0, 255), 3);
    }
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

    /*for(size_t i = 0; i < groups.size(); ++i){
        auto& group = groups[i];

        if(group.size() < 9){
            continue;
        }

        auto angle_first = i * PARALLEL_RESOLUTION;
        auto angle_last = angle_first + PARALLEL_RESOLUTION - 1;

        //TODO TEMPORARY
        if(angle_first < 90){
            continue;
        }

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

        auto min_tot = std::min(source_image.rows, source_image.cols);

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
    }*/
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

    typedef std::tuple<std::size_t,std::size_t,std::size_t,std::size_t> square_t;
    std::vector<square_t> squares;

    auto limit = std::min(source_image.rows, source_image.cols) / 9.0f;

    for(size_t i = 0; i < points.size(); ++i){
        for(size_t j = 0; j < points.size(); ++j){
            if(i == j){
                continue;
            }
            auto dij = distance(points[i], points[j]);

            if(dij > limit){
                continue;
            }

            for(size_t k = 0; k < points.size(); ++k){
                if(k != j && k != i){
                    for(size_t l = 0; l < points.size(); ++l){
                        if(l != k && l != j && l != i){
                            if(is_square_2(points[i], points[j], points[k], points[l])){
                                squares.emplace_back(i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }

    for(auto& square : squares){
        auto d = square_edge(
                points[std::get<0>(square)], points[std::get<1>(square)],
                points[std::get<2>(square)], points[std::get<3>(square)]);

        if(d < 75.0f ){
            draw_square(dest_image,
                points[std::get<0>(square)], points[std::get<1>(square)],
                points[std::get<2>(square)], points[std::get<3>(square)]
               );
        }
    }

    return;
}

} //end of anonymous namespace

int main(int argc, char** argv ){
    if(argc < 2){
        std::cout << "Usage: binarize <image>..." << std::endl;
        return -1;
    }

    cv::Point2f p1(1.0f, 1.05f);
    cv::Point2f p2(2.0f, 2.05f);
    cv::Point2f p3(1.05f, 2.0f);
    cv::Point2f p4(2.05f, 1.05f);

    if(argc == 2){
        std::string source_path(argv[1]);

        cv::Mat source_image;
        source_image = cv::imread(source_path.c_str(), 1);

        if (!source_image.data){
            std::cout << "Invalid source_image" << std::endl;
            return -1;
        }

        cv::Mat dest_image;
        //draw_lines(source_image, dest_image);
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