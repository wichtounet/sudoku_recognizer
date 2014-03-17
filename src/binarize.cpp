#include <opencv2/opencv.hpp>

#include <iostream>

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

bool acceptLinePair(cv::Vec2f line1, cv::Vec2f line2, float minTheta){
    float theta1 = line1[1], theta2 = line2[1];

    if(theta1 < minTheta){
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    if(theta2 < minTheta){
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    return abs(theta1 - theta2) > minTheta;
}

std::vector<cv::Point2f> lineToPointPair(cv::Vec2f line){
    std::vector<cv::Point2f> points;

    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

    points.push_back(cv::Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
    points.push_back(cv::Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));

    return points;
}

cv::Point2f computeIntersect(cv::Vec2f line1, cv::Vec2f line2){
    std::vector<cv::Point2f> p1 = lineToPointPair(line1);
    std::vector<cv::Point2f> p2 = lineToPointPair(line2);

    float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
    cv::Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
                       (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
                      ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
                       (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);

    return intersect;
}

void sudoku_lines(const cv::Mat& source_image, cv::Mat& dest_image){
    dest_image = source_image.clone();

    cv::Mat binary_image;
    method_4(source_image, binary_image);

    constexpr const size_t CANNY_THRESHOLD = 50;
    cv::Canny(binary_image, binary_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 3);

    std::vector<cv::Vec2f> lines;
    HoughLines(binary_image, lines, 1, CV_PI/180, 125, 0, 0);

    std::vector<cv::Point2f> intersections;
    for( size_t i = 0; i < lines.size(); i++ ){
        for(size_t j = 0; j < lines.size(); j++){
            cv::Vec2f line1 = lines[i];
            cv::Vec2f line2 = lines[j];
            if(acceptLinePair(line1, line2, CV_PI / 32)){
                cv::Point2f intersection = computeIntersect(line1, line2);
                intersections.push_back(intersection);
            }
        }
    }

    for(auto& i : intersections){
        cv::circle(dest_image, i, 1, cv::Scalar(0, 255, 0), 3);

    }

    return;

    /*if(lines.size() > 1000000){
        cv::Mat gray_image;
        cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

        cv::Canny(gray_image, gray_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 3);

        lines.clear();
        HoughLines(gray_image, lines, 1, CV_PI/180, 100, 0, 0);
    }*/

    std::cout << lines.size() << std::endl;

    /*for(auto& line : lines){
        draw_line(dest_image, line);
    }

    return;*/

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

    for(size_t g = 0; g < groups.size(); ++g){
        auto& group = groups[g];

        if(group.size() < 10){
            continue;
        }

        if(g != 9){
            //continue;
        }

        std::cout << "group(" << g << "), size=" << group.size() << std::endl;

        auto angle_first = g * PARALLEL_RESOLUTION;
        auto angle_last = angle_first + PARALLEL_RESOLUTION - 1;

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

        for(size_t i = 0 ; i < distance_groups.size(); ++i){
            std::cout << "dgroup(" << i << "), size=" << distance_groups[i].size() << std::endl;
        }

        for(auto& line : distance_groups){
            draw_line(dest_image, group[line[0]]);
        }

        continue;

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
                    auto d_to_prev = distance(group[distance_groups[min_i-1][0]], group[distance_groups[min_i][0]]);
                    auto d_to_next_next = distance(group[distance_groups[min_i][0]], group[distance_groups[min_i+2][0]]);
                    auto d_next_to_next = distance(group[distance_groups[min_i+1][0]], group[distance_groups[min_i+2][0]]);
                    auto d_prev_to_next = distance(group[distance_groups[min_i-1][0]], group[distance_groups[min_i+1][0]]);

                    auto delete_i = d_prev_to_next;
                    auto delete_next = d_to_next_next;

                    /*if(std::abs(delete_i - average) > std::abs(delete_next - average)){
                        std::cout << "delete i " << std::endl;
                        distance_groups.erase(distance_groups.begin() + min_i);
                        continue;
                    } else {
                        std::cout << "delete next " << std::endl;
                        distance_groups.erase(distance_groups.begin() + min_i + 1);
                        continue;
                    }*/
                }
            }

            break;
        }

        for(auto& distance_group : distance_groups){
            draw_line(dest_image, group[distance_group[0]]);
        }

    /*    auto it = distance_groups.begin();
        auto end = distance_groups.end();

        while(it != end){
            if(it == distance_groups.begin()){
                auto right = it + 1;
                if(right != end){
                    auto d = distance(group[(*it)[0]], group[(*right)[0]]);

                    std::cout << "distance from right " << d << std::endl;

                    if(d < average * 0.90 || d > average * 1.10){
                        distance_groups.erase(it);

                        it = distance_groups.begin();
                        end = distance_groups.end();
                        average = average_distance(group, distance_groups);
                        std::cout << "average " << average << std::endl;
                        continue;
                    }
                }
            }

            ++it;
        }*/
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

        cv::Mat dest_image;
        sudoku_lines(source_image, dest_image);

        cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
        cv::imshow("Sudoku Grid", dest_image);

        cv::waitKey(0);
    } else {
        for(size_t i = 1; i < argc; ++i){
            std::string source_path(argv[i]);

            cv::Mat source_image;
            source_image = cv::imread(source_path.c_str(), 1);

            if (!source_image.data){
                std::cout << "Invalid source_image" << std::endl;
                continue;
            }

            cv::Mat dest_image;
            sudoku_lines(source_image, dest_image);

            source_path.insert(source_path.rfind('.'), ".lines");
            imwrite(source_path.c_str(), dest_image);
        }
    }

    return 0;
}